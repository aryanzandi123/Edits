from dotenv import load_dotenv
import os

# Load .env file - try current directory and parent directory
load_dotenv()
load_dotenv(override=True)  # Reload to ensure fresh values

# Hardcode API key as fallback (TEMPORARY - for debugging)
if not os.getenv("GOOGLE_API_KEY"):
    print("[WARN]WARNING: Using hardcoded API key. Create a .env file with GOOGLE_API_KEY for production.")

import re
import sys
import json
import time
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

from utils.pruner import (
    run_prune_job,
    pruned_filename,
    is_pruned_fresh,
    make_prune_job_id,
    parse_prune_job_id,
    HARD_MAX_KEEP_DEFAULT,
    PRUNED_DIRNAME,
    PROTEIN_RE,
)

# Import protein database for cross-query knowledge
import utils.protein_database as pdb

# --- Import your existing functions ---
# You'll need to make them importable, e.g., from runner import run_pipeline
from runner import run_full_job, run_requery_job
from visualizer import create_visualization

# --- App Setup ---
app = Flask(__name__)

# --- Database Configuration (PostgreSQL via Railway) ---
# Use DATABASE_PUBLIC_URL for local dev (accessible externally)
# Use DATABASE_URL for production (Railway internal network)
database_url = os.getenv('DATABASE_PUBLIC_URL') or os.getenv('DATABASE_URL')
if not database_url:
    print("[WARN]WARNING: DATABASE_URL not set. Using SQLite fallback (local dev only).", file=sys.stderr)
    database_url = 'sqlite:///fallback.db'
elif database_url.startswith('postgres://'):
    # Railway provides postgres:// but SQLAlchemy 1.4+ requires postgresql://
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 5,
    'pool_recycle': 3600,
    'pool_pre_ping': True,  # Verify connections before using
    'connect_args': {
        'connect_timeout': 10
    }
}

# Initialize SQLAlchemy with app
from models import db
db.init_app(app)

# Create tables and validate database connection
with app.app_context():
    print("\n" + "="*60, file=sys.stderr)
    print("[DATABASE] Initializing PostgreSQL connection...", file=sys.stderr)
    print(f"[DATABASE] URL: {database_url.split('@')[0]}@***", file=sys.stderr)  # Hide credentials

    try:
        # Test connection first
        db.session.execute(db.text('SELECT 1'))
        print("[DATABASE] [OK]Connection verified", file=sys.stderr)

        # Create tables if they don't exist
        db.create_all()

        # Verify tables exist
        from models import Protein, Interaction
        protein_count = Protein.query.count()
        interaction_count = Interaction.query.count()

        print("[DATABASE] [OK]Tables initialized", file=sys.stderr)
        print(f"[DATABASE]   • Proteins table: {protein_count} entries", file=sys.stderr)
        print(f"[DATABASE]   • Interactions table: {interaction_count} entries", file=sys.stderr)
        print(f"[DATABASE] [OK]Database ready for sync", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)

    except Exception as e:
        print(f"\n[ERROR][DATABASE] Initialization failed: {e}", file=sys.stderr)
        print(f"   Falling back to file-based cache only", file=sys.stderr)
        print(f"   All query results will be saved to cache/ directory", file=sys.stderr)
        print(f"   Run 'python sync_cache_to_db.py <PROTEIN>' to sync manually later", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
PRUNED_DIR = os.path.join(CACHE_DIR, PRUNED_DIRNAME)
os.makedirs(PRUNED_DIR, exist_ok=True)

# --- Job Tracking (Crucial for handling concurrency) ---
# This dictionary will store the status of running jobs.
# The lock is essential to prevent race conditions when multiple users access it.
jobs = {}
jobs_lock = threading.Lock()

# --- Main Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/search/<protein>')
def search_protein(protein):
    """
    Search for a protein in the database (no querying/research).

    Returns:
        JSON: {
            "status": "found" | "not_found",
            "protein": str,
            "has_interactions": bool (if found),
            "interaction_count": int (if found),
            "last_queried": str (if found),
            "query_count": int (if found)
        }
    """
    # Validate protein name
    if not re.match(r'^[a-zA-Z0-9_-]+$', protein):
        return jsonify({
            "error": "Invalid protein name format. Please use only letters, numbers, hyphens, and underscores."
        }), 400

    try:
        from models import Protein, Interaction

        # Check if protein exists in database
        protein_obj = Protein.query.filter_by(symbol=protein).first()

        if not protein_obj:
            return jsonify({
                "status": "not_found",
                "protein": protein
            })

        # Count interactions (bidirectional due to canonical ordering)
        interaction_count = db.session.query(Interaction).filter(
            (Interaction.protein_a_id == protein_obj.id) |
            (Interaction.protein_b_id == protein_obj.id)
        ).count()

        return jsonify({
            "status": "found",
            "protein": protein,
            "has_interactions": interaction_count > 0,
            "interaction_count": interaction_count,
            "last_queried": protein_obj.last_queried.isoformat() if protein_obj.last_queried else None,
            "query_count": protein_obj.query_count
        })

    except Exception as e:
        print(f"⚠️  Search failed: {e}", file=sys.stderr)
        return jsonify({"error": "Database search failed"}), 500

@app.route('/api/query', methods=['POST'])
def start_query():
    """Starts a new pipeline job in the background."""
    data = request.json
    protein_name = data.get('protein')
    if not protein_name:
        return jsonify({"error": "Protein name is required"}), 400

    if not re.match(r'^[a-zA-Z0-9_-]+$', protein_name):
        return jsonify({
            "error": "Invalid protein name format. Please use only letters, numbers, hyphens, and underscores."
        }), 400

    # Extract configuration (with defaults and validation)
    try:
        interactor_rounds = int(data.get('interactor_rounds', 3))
        function_rounds = int(data.get('function_rounds', 3))
        max_depth = int(data.get('max_depth', 3))

        # Clamp to valid range (3-8)
        interactor_rounds = max(3, min(8, interactor_rounds))
        function_rounds = max(3, min(8, function_rounds))
        # max_depth: 1-4 limited, 5+ unlimited
        max_depth = max(1, max_depth)
    except (TypeError, ValueError):
        # If invalid, use defaults
        interactor_rounds = 3
        function_rounds = 3
        max_depth = 3

    # Extract skip_validation option (default: False)
    skip_validation = bool(data.get('skip_validation', False))

    # Extract skip_deduplicator option (default: False)
    skip_deduplicator = bool(data.get('skip_deduplicator', False))

    # Extract skip_arrow_determination option (default: False)
    skip_arrow_determination = bool(data.get('skip_arrow_determination', False))

    # Extract skip_fact_checking option (default: False)
    skip_fact_checking = bool(data.get('skip_fact_checking', False))

    # No instant returns - always run pipeline
    # This allows finding NEW interactions for existing proteins
    # The pipeline has built-in history awareness via known_interactions context

    with jobs_lock:
        # Check if a job for this protein is already running
        current_job = jobs.get(protein_name)
        if current_job:
            # Only prevent starting a new job if one is actively processing
            # Allow restarting if the previous job was cancelled, errored, or is cancelling
            current_status = current_job.get("status")
            if current_status == "processing":
                # Check if the job has a cancel event that's been set (being cancelled)
                cancel_event_check = current_job.get("cancel_event")
                if cancel_event_check and cancel_event_check.is_set():
                    # Job is being cancelled, allow restart
                    pass
                else:
                    # Job is actively running
                    return jsonify({"status": "processing", "message": "Job already in progress."})

        # 4. Start a new job with configuration and cancellation event
        cancel_event = threading.Event()
        jobs[protein_name] = {
            "status": "processing",
            "progress": "Initializing pipeline...",
            "cancel_event": cancel_event
        }
        thread = threading.Thread(
            target=run_full_job,
            args=(protein_name, jobs, jobs_lock, interactor_rounds, function_rounds, max_depth, skip_validation, skip_deduplicator, skip_arrow_determination, skip_fact_checking, app)
        )
        thread.daemon = True
        thread.start()

    return jsonify({"status": "processing", "protein": protein_name})


@app.route('/api/requery', methods=['POST'])
def start_requery():
    """
    DEPRECATED: Use /api/query instead (which now handles both new and existing proteins).
    This endpoint is kept for backward compatibility only.

    Re-queries a protein with context from previous results to find NEW interactors/functions.
    """
    print("⚠️  DEPRECATED: /api/requery called. Use /api/query instead.", file=sys.stderr)

    data = request.json
    protein_name = data.get('protein')
    if not protein_name:
        return jsonify({"error": "Protein name is required"}), 400

    if not re.match(r'^[a-zA-Z0-9_-]+$', protein_name):
        return jsonify({
            "error": "Invalid protein name format. Please use only letters, numbers, hyphens, and underscores."
        }), 400

    # Extract configuration (with defaults and validation)
    # RE-QUERIES allow minimum 1 round (quick searches for missed items)
    try:
        interactor_rounds = int(data.get('interactor_rounds', 1))
        function_rounds = int(data.get('function_rounds', 1))
        max_depth = int(data.get('max_depth', 3))

        # Clamp to valid range (1-8 for re-queries)
        interactor_rounds = max(1, min(8, interactor_rounds))
        function_rounds = max(1, min(8, function_rounds))
        max_depth = max(1, max_depth)
    except (TypeError, ValueError):
        # If invalid, use defaults (1 for quick re-query)
        interactor_rounds = 1
        function_rounds = 1
        max_depth = 3

    # Extract skip options
    skip_deduplicator = bool(data.get('skip_deduplicator', False))
    skip_fact_checking = bool(data.get('skip_fact_checking', False))

    # Check if cached result exists (required for re-query)
    cache_path = os.path.join(CACHE_DIR, f"{protein_name}.json")
    if not os.path.exists(cache_path):
        return jsonify({"error": "No cached results found. Use /api/query for initial query."}), 404

    with jobs_lock:
        # Check if a job for this protein is already running
        current_job = jobs.get(protein_name)
        if current_job:
            current_status = current_job.get("status")
            if current_status == "processing":
                cancel_event_check = current_job.get("cancel_event")
                if cancel_event_check and cancel_event_check.is_set():
                    pass  # Job is being cancelled, allow restart
                else:
                    return jsonify({"status": "processing", "message": "Job already in progress."}), 409

        # Start a re-query job with context
        cancel_event = threading.Event()
        jobs[protein_name] = {
            "status": "processing",
            "progress": "Re-querying with context...",
            "cancel_event": cancel_event
        }
        thread = threading.Thread(
            target=run_requery_job,
            args=(protein_name, jobs, jobs_lock, interactor_rounds, function_rounds, max_depth, skip_deduplicator, skip_fact_checking, app)
        )
        thread.daemon = True
        thread.start()

    return jsonify({"status": "processing", "protein": protein_name})


@app.route('/api/status/<protein>')
def get_status(protein):
    """Checks the status of a running job."""
    # IMPORTANT: Check jobs dict FIRST before checking cache
    # This allows re-queries to run even when cache exists
    with jobs_lock:
        job_status = jobs.get(protein)

    # If there's an active job, return its status
    if job_status:
        # Filter out non-serializable fields (like threading.Event)
        serializable_status = {k: v for k, v in job_status.items() if k != "cancel_event"}
        return jsonify(serializable_status)

    # If no active job, check if cached result exists
    cache_path = os.path.join(CACHE_DIR, f"{protein}.json")
    if os.path.exists(cache_path):
        return jsonify({"status": "complete"})

    # No job and no cache
    return jsonify({"status": "not_found"})


# ============================================================================
# Helper Functions for Database Queries
# ============================================================================

def build_full_json_from_db(protein_symbol: str) -> dict:
    """
    Reconstruct complete JSON from PostgreSQL database.

    NEW: Returns restructured format with proteins array and interactions array.
    This enables cleaner graph rendering without function nodes cluttering the view.

    Args:
        protein_symbol: Protein to query (e.g., "ATXN3")

    Returns:
        Dict with snapshot_json and ctx_json, or None if protein not found
        Format: {
            "snapshot_json": {
                "main": "PROTEIN",
                "proteins": ["PROTEIN", "INTERACTOR1", ...],
                "interactions": [
                    {
                        "type": "direct",
                        "source": "PROTEIN",
                        "target": "INTERACTOR1",
                        "direction": "bidirectional",
                        "arrow": "binds",
                        "confidence": 0.85,
                        "functions": [...],
                        "evidence": [...],
                        ...
                    },
                    {
                        "type": "shared",
                        "source": "INTERACTOR1",
                        "target": "INTERACTOR2",
                        ...
                    }
                ]
            },
            "ctx_json": {...}
        }

    Side effects:
        Queries database via SQLAlchemy
    """
    from models import Protein, Interaction

    # Query main protein
    main_protein = Protein.query.filter_by(symbol=protein_symbol).first()
    if not main_protein:
        return None

    # Query all interactions using CANONICAL ORDERING
    # Since we enforce protein_a_id < protein_b_id, we only need to check both positions
    db_interactions = db.session.query(Interaction).filter(
        (Interaction.protein_a_id == main_protein.id) |
        (Interaction.protein_b_id == main_protein.id)
    ).all()

    # Build interactions list with explicit source/target/type
    interactions_list = []
    protein_set = {protein_symbol}  # Track all unique proteins
    interactor_proteins = []  # Track interactor Protein objects for shared link query

    # Process direct interactions (main protein ↔ interactor)
    for interaction in db_interactions:
        # Determine partner protein and perspective
        if interaction.protein_a_id == main_protein.id:
            # Main protein is stored as protein_a
            partner = interaction.protein_b
            needs_flip = False
        else:
            # Main protein is stored as protein_b (reversed in storage)
            partner = interaction.protein_a
            needs_flip = True

        # Track partner for shared link detection
        interactor_proteins.append(partner)
        protein_set.add(partner.symbol)

        # Extract FULL data from JSONB (preserves all fields: functions, evidence, PMIDs, etc.)
        interaction_data = interaction.data.copy()

        # Add explicit source/target/type fields for frontend clarity
        # CRITICAL: Set source/target based on DIRECTION (who affects whom), NOT canonical ordering

        # Step 1: Convert absolute direction to query-relative direction
        # Database stores absolute directions: "a_to_b", "b_to_a", "bidirectional"
        # Frontend expects query-relative: "main_to_primary", "primary_to_main", "bidirectional"
        stored_direction = interaction.direction

        if needs_flip:
            # Query protein is protein_b (reversed in storage)
            # Convert absolute → query-relative from protein_b's perspective
            if stored_direction == "a_to_b":
                # protein_a → protein_b means partner → query
                final_direction = "primary_to_main"
            elif stored_direction == "b_to_a":
                # protein_b → protein_a means query → partner
                final_direction = "main_to_primary"
            else:
                # bidirectional stays bidirectional
                final_direction = stored_direction or "bidirectional"
        else:
            # Query protein is protein_a (natural order)
            # Convert absolute → query-relative from protein_a's perspective
            if stored_direction == "a_to_b":
                # protein_a → protein_b means query → partner
                final_direction = "main_to_primary"
            elif stored_direction == "b_to_a":
                # protein_b → protein_a means partner → query
                final_direction = "primary_to_main"
            else:
                # bidirectional stays bidirectional
                final_direction = stored_direction or "bidirectional"

        # Step 2: Set source/target based on FINAL DIRECTION (not canonical ordering!)
        if final_direction == "main_to_primary":
            # Main protein affects interactor: query → interactor
            interaction_data["source"] = protein_symbol
            interaction_data["target"] = partner.symbol
        elif final_direction == "primary_to_main":
            # Interactor affects main protein: interactor → query
            interaction_data["source"] = partner.symbol
            interaction_data["target"] = protein_symbol
        else:
            # Bidirectional/undirected: use alphabetical order for consistency
            if protein_symbol < partner.symbol:
                interaction_data["source"] = protein_symbol
                interaction_data["target"] = partner.symbol
            else:
                interaction_data["source"] = partner.symbol
                interaction_data["target"] = protein_symbol

        interaction_data["direction"] = final_direction

        # Extract interaction_type and upstream_interactor from DB columns
        # Use interaction_type from DB column for both fields
        interaction_type_value = interaction.interaction_type or "direct"
        interaction_data["type"] = interaction_type_value  # Frontend reads this
        interaction_data["interaction_type"] = interaction_type_value  # Metadata
        if interaction.upstream_interactor:
            interaction_data["upstream_interactor"] = interaction.upstream_interactor

        # Special handling for indirect interactions:
        # For indirect interactions, source should be upstream_interactor, not main protein
        # Example: p62→KEAP1→NRF2 should render as source=KEAP1, target=NRF2
        if interaction_type_value == "indirect" and interaction.upstream_interactor:
            interaction_data["source"] = interaction.upstream_interactor
            # Target is the partner protein (already set correctly above)
            # Direction already set correctly at line 474 from database - don't override
            # IMPORTANT: Add flag to indicate direction semantics have changed
            # Direction is now LINK-ABSOLUTE (upstream→partner), not query-relative (main→partner)
            interaction_data["_direction_is_link_absolute"] = True

        # Extract chain metadata for indirect interactions
        if interaction.mediator_chain:
            interaction_data["mediator_chain"] = interaction.mediator_chain
        if interaction.depth:
            interaction_data["depth"] = interaction.depth
        if interaction.chain_context:
            interaction_data["chain_context"] = interaction.chain_context

        # Ensure required fields have defaults
        if interaction_data.get("confidence") is None:
            interaction_data["confidence"] = 0.5
        if interaction_data.get("arrow") is None:
            interaction_data["arrow"] = "binds"
        if interaction_data.get("functions") is None:
            interaction_data["functions"] = []
        if interaction_data.get("evidence") is None:
            interaction_data["evidence"] = []
        if interaction_data.get("pmids") is None:
            interaction_data["pmids"] = []

        # Auto-generate interaction_effect from arrow if not present
        if not interaction_data.get("interaction_effect"):
            arrow = interaction_data.get("arrow", "binds")
            effect_map = {
                "activates": "activation",
                "inhibits": "inhibition",
                "binds": "binding",
                "regulates": "regulation",
                "complex": "complex formation"
            }
            interaction_data["interaction_effect"] = effect_map.get(arrow, arrow)

        # Auto-generate function_effect for each function if not present
        for func in interaction_data.get("functions", []):
            if not func.get("function_effect"):
                func_arrow = func.get("arrow", "")
                if func_arrow:
                    effect_map = {
                        "activates": "activation",
                        "inhibits": "inhibition",
                        "binds": "binding",
                        "regulates": "regulation",
                        "complex": "complex formation"
                    }
                    func["function_effect"] = effect_map.get(func_arrow, func_arrow)

            # For indirect interactions with arrow_context, ensure net_effect and direct_effect
            if func.get("arrow_context"):
                arrow_ctx = func["arrow_context"]
                net_arrow = arrow_ctx.get("net_arrow", func.get("arrow", "regulates"))
                direct_arrow = arrow_ctx.get("direct_arrow", net_arrow)

                effect_map = {
                    "activates": "activation",
                    "inhibits": "inhibition",
                    "binds": "binding",
                    "regulates": "regulation",
                    "complex": "complex formation"
                }

                func["net_effect"] = effect_map.get(net_arrow, net_arrow)
                func["direct_effect"] = effect_map.get(direct_arrow, direct_arrow)

        # CRITICAL FIX: For direct mediator links, use direct_arrow from arrow_context
        # This ensures RHEB→MTOR shows "activates" (direct) instead of "inhibits" (net)
        function_context = interaction_data.get("function_context")
        if function_context == "direct":
            # Check if any function has arrow_context with direct_arrow
            for func in interaction_data.get("functions", []):
                if func.get("arrow_context") and func["arrow_context"].get("direct_arrow"):
                    correct_arrow = func["arrow_context"]["direct_arrow"]
                    if interaction_data.get("arrow") != correct_arrow:
                        # Override interaction-level arrow with direct arrow
                        interaction_data["arrow"] = correct_arrow
                        # Also update interaction_effect
                        effect_map = {
                            "activates": "activation",
                            "inhibits": "inhibition",
                            "binds": "binding",
                            "regulates": "regulation",
                            "complex": "complex formation"
                        }
                        interaction_data["interaction_effect"] = effect_map.get(correct_arrow, correct_arrow)
                    break  # Only need first function with arrow_context

        # Add differentiation flags for dual-track indirect/direct system
        # This helps frontend visually distinguish NET effects vs DIRECT effects
        if function_context == "net":
            interaction_data["_net_effect"] = True  # Frontend marker for NET chain effects
            interaction_data["_display_badge"] = "NET EFFECT"
        elif function_context == "direct" and interaction_data.get("_inferred_from_chain"):
            interaction_data["_direct_mediator_link"] = True  # Frontend marker for extracted direct links
            interaction_data["_display_badge"] = "DIRECT LINK"

        interactions_list.append(interaction_data)

    # Retrieve chain links for indirect interactions
    # For indirect interactions like p62→KEAP1→NRF2, we need to also query for KEAP1→NRF2
    chain_link_ids = set()  # Track which chain links we've already added
    for interaction in db_interactions:
        if interaction.interaction_type == "indirect":
            mediator_chain = interaction.mediator_chain
            if mediator_chain and isinstance(mediator_chain, list):
                # Get the target protein of the indirect interaction
                if interaction.protein_a_id == main_protein.id:
                    target_protein = interaction.protein_b
                else:
                    target_protein = interaction.protein_a

                # For each mediator in the chain, query for mediator→target link
                for mediator_symbol in mediator_chain:
                    mediator_protein = Protein.query.filter_by(symbol=mediator_symbol).first()
                    if not mediator_protein:
                        continue

                    # Query for chain link (mediator ↔ target)
                    # Use canonical ordering: protein_a_id < protein_b_id
                    if mediator_protein.id < target_protein.id:
                        chain_link = db.session.query(Interaction).filter(
                            Interaction.protein_a_id == mediator_protein.id,
                            Interaction.protein_b_id == target_protein.id
                        ).first()
                    else:
                        chain_link = db.session.query(Interaction).filter(
                            Interaction.protein_a_id == target_protein.id,
                            Interaction.protein_b_id == mediator_protein.id
                        ).first()

                    if chain_link and chain_link.id not in chain_link_ids:
                        chain_link_ids.add(chain_link.id)

                        # Add this chain link to interactor_proteins for shared link detection
                        if chain_link.protein_a not in interactor_proteins:
                            interactor_proteins.append(chain_link.protein_a)
                        if chain_link.protein_b not in interactor_proteins:
                            interactor_proteins.append(chain_link.protein_b)

                        # Add both proteins to protein set
                        protein_set.add(chain_link.protein_a.symbol)
                        protein_set.add(chain_link.protein_b.symbol)

                        # Build interaction data for chain link
                        chain_data = chain_link.data.copy()

                        # Determine source/target (mediator → target)
                        # Chain links are always mediator as source, target as target
                        chain_data["source"] = mediator_symbol
                        chain_data["target"] = target_protein.symbol

                        # Set type and interaction_type
                        chain_interaction_type = chain_link.interaction_type or "direct"
                        chain_data["type"] = chain_interaction_type
                        chain_data["interaction_type"] = chain_interaction_type
                        chain_data["direction"] = chain_link.direction if chain_link.direction else "bidirectional"

                        # Ensure required fields
                        if chain_data.get("confidence") is None:
                            chain_data["confidence"] = 0.5
                        if chain_data.get("arrow") is None:
                            chain_data["arrow"] = chain_link.arrow or "binds"
                        if chain_data.get("functions") is None:
                            chain_data["functions"] = []
                        if chain_data.get("evidence") is None:
                            chain_data["evidence"] = []
                        if chain_data.get("pmids") is None:
                            chain_data["pmids"] = []

                        # Auto-generate effect labels for chain links
                        if not chain_data.get("interaction_effect"):
                            arrow = chain_data.get("arrow", "binds")
                            effect_map = {
                                "activates": "activation",
                                "inhibits": "inhibition",
                                "binds": "binding",
                                "regulates": "regulation",
                                "complex": "complex formation"
                            }
                            chain_data["interaction_effect"] = effect_map.get(arrow, arrow)

                        for func in chain_data.get("functions", []):
                            if not func.get("function_effect"):
                                func_arrow = func.get("arrow", "")
                                if func_arrow:
                                    effect_map = {
                                        "activates": "activation",
                                        "inhibits": "inhibition",
                                        "binds": "binding",
                                        "regulates": "regulation",
                                        "complex": "complex formation"
                                    }
                                    func["function_effect"] = effect_map.get(func_arrow, func_arrow)

                        # CRITICAL FIX: For direct chain links, use direct_arrow from arrow_context
                        chain_function_context = chain_data.get("function_context") or chain_link.function_context
                        if chain_function_context == "direct":
                            # Check if any function has arrow_context with direct_arrow
                            for func in chain_data.get("functions", []):
                                if func.get("arrow_context") and func["arrow_context"].get("direct_arrow"):
                                    correct_arrow = func["arrow_context"]["direct_arrow"]
                                    if chain_data.get("arrow") != correct_arrow:
                                        # Override chain link arrow with direct arrow
                                        chain_data["arrow"] = correct_arrow
                                        # Also update interaction_effect
                                        effect_map = {
                                            "activates": "activation",
                                            "inhibits": "inhibition",
                                            "binds": "binding",
                                            "regulates": "regulation",
                                            "complex": "complex formation"
                                        }
                                        chain_data["interaction_effect"] = effect_map.get(correct_arrow, correct_arrow)
                                    break  # Only need first function with arrow_context

                        # Add differentiation flags for chain links (same as direct interactions)
                        if chain_function_context == "direct" and chain_data.get("_inferred_from_chain"):
                            chain_data["_direct_mediator_link"] = True
                            chain_data["_display_badge"] = "DIRECT LINK"

                        interactions_list.append(chain_data)

    # Query for shared interactions BETWEEN interactors
    # This reveals triangular relationships: HDAC6 → VCP, HDAC6 → ATXN3, VCP ↔ ATXN3
    if len(interactor_proteins) > 1:
        # Get IDs of all interactor proteins
        interactor_ids = [p.id for p in interactor_proteins]

        # Query interactions where BOTH proteins are in the interactor list
        # This finds interactions BETWEEN the interactors (not involving main protein)
        # EXCLUDE interactions involving main protein (prevents duplicate display of direct interactions)
        # INCLUDE extracted mediator links (e.g., RHEB→MTOR from ATXN3→RHEB→MTOR chain)
        shared_interactions = db.session.query(Interaction).filter(
            Interaction.protein_a_id.in_(interactor_ids),
            Interaction.protein_b_id.in_(interactor_ids),
            ~((Interaction.protein_a_id == main_protein.id) | (Interaction.protein_b_id == main_protein.id))  # Only exclude if main protein is involved
        ).all()

        # Build set of proteins involved in indirect chains for THIS query
        # (e.g., if KEAP1→NRF2 is indirect chain in p62 query, exclude KEAP1-NRF2 shared link)
        indirect_chain_pairs = set()
        for interaction in db_interactions:
            inter_data = interaction.data
            if inter_data.get('interaction_type') == 'indirect':
                upstream = inter_data.get('upstream_interactor')
                if upstream:
                    # Get partner protein
                    if interaction.protein_a_id == main_protein.id:
                        target = interaction.protein_b.symbol
                    else:
                        target = interaction.protein_a.symbol
                    # Add both orderings (KEAP1-NRF2 and NRF2-KEAP1)
                    indirect_chain_pairs.add((upstream, target))
                    indirect_chain_pairs.add((target, upstream))

        # Add shared interactions with FULL JSONB data (including functions)
        for shared_ix in shared_interactions:
            # Skip if this interaction is part of an indirect chain in THIS query
            protein_a_sym = shared_ix.protein_a.symbol
            protein_b_sym = shared_ix.protein_b.symbol
            if (protein_a_sym, protein_b_sym) in indirect_chain_pairs or (protein_b_sym, protein_a_sym) in indirect_chain_pairs:
                continue  # Don't add to shared links
            # Get both proteins involved in this shared link
            protein_a = shared_ix.protein_a
            protein_b = shared_ix.protein_b

            # Add both proteins to protein set
            protein_set.add(protein_a.symbol)
            protein_set.add(protein_b.symbol)

            # Extract FULL JSONB data (including functions, evidence, PMIDs)
            # NOTE: User confirmed they WANT functions for shared links
            shared_data = shared_ix.data.copy()

            # Add explicit source/target/type fields
            shared_data["source"] = protein_a.symbol
            shared_data["target"] = protein_b.symbol
            shared_data["type"] = "shared"  # For backward compatibility (frontend expects "type" field)
            shared_data["_is_shared_link"] = True  # Frontend marker for styling
            shared_data["interaction_type"] = shared_ix.interaction_type or "direct"
            if shared_ix.upstream_interactor:
                shared_data["upstream_interactor"] = shared_ix.upstream_interactor
            # Use DB direction if present, otherwise default to bidirectional
            shared_data["direction"] = shared_ix.direction if shared_ix.direction else "bidirectional"

            # Ensure required fields have defaults
            if shared_data.get("confidence") is None:
                shared_data["confidence"] = 0.5
            if shared_data.get("arrow") is None:
                shared_data["arrow"] = shared_ix.arrow or "binds"
            if shared_data.get("functions") is None:
                shared_data["functions"] = []
            if shared_data.get("evidence") is None:
                shared_data["evidence"] = []
            if shared_data.get("pmids") is None:
                shared_data["pmids"] = []
            if shared_data.get("intent") is None:
                shared_data["intent"] = "binding"

            # Auto-generate effect labels for shared links
            if not shared_data.get("interaction_effect"):
                arrow = shared_data.get("arrow", "binds")
                effect_map = {
                    "activates": "activation",
                    "inhibits": "inhibition",
                    "binds": "binding",
                    "regulates": "regulation",
                    "complex": "complex formation"
                }
                shared_data["interaction_effect"] = effect_map.get(arrow, arrow)

            for func in shared_data.get("functions", []):
                if not func.get("function_effect"):
                    func_arrow = func.get("arrow", "")
                    if func_arrow:
                        effect_map = {
                            "activates": "activation",
                            "inhibits": "inhibition",
                            "binds": "binding",
                            "regulates": "regulation",
                            "complex": "complex formation"
                        }
                        func["function_effect"] = effect_map.get(func_arrow, func_arrow)

            interactions_list.append(shared_data)

    # Build snapshot_json with new structure
    snapshot_json = {
        "main": protein_symbol,
        "proteins": sorted(list(protein_set)),  # All unique proteins (sorted for consistency)
        "interactions": interactions_list  # All interactions with full JSONB data
    }

    # Build ctx_json (keep for backwards compatibility, simplified)
    ctx_json = {
        "main": protein_symbol,
        "proteins": snapshot_json["proteins"],
        "interactions": interactions_list,
        "interactor_history": [p for p in snapshot_json["proteins"] if p != protein_symbol],
        "function_history": {},
        "function_batches": []
    }

    return {
        "snapshot_json": snapshot_json,
        "ctx_json": ctx_json
    }


def build_expansion_json_from_db(protein_symbol: str, visible_proteins: list = None) -> dict:
    """
    Build expansion JSON with auto-cross-linking support.

    This extends build_full_json_from_db() to discover shared interactions between
    newly-expanded proteins and existing visible proteins in the graph.

    Example:
        User has HDAC6 visible, expands VCP.
        VCP's interactors include BECN1.
        If BECN1 also interacts with HDAC6 (already visible), that link is included.

    Args:
        protein_symbol: Protein to expand (e.g., "VCP")
        visible_proteins: List of proteins currently visible in graph (e.g., ["HDAC6", "ATXN3"])

    Returns:
        Dict with snapshot_json and ctx_json, including discovered cross-links

    Side effects:
        Queries database via SQLAlchemy
    """
    from models import Protein, Interaction

    # Get base expansion data (protein's direct interactions + shared links between its interactors)
    result = build_full_json_from_db(protein_symbol)
    if not result:
        return None

    # If no visible proteins provided, return base result (no cross-linking needed)
    if not visible_proteins or not isinstance(visible_proteins, list):
        return result

    # Filter out the protein being expanded from visible list (already in base result)
    visible_proteins = [p for p in visible_proteins if p != protein_symbol]
    if not visible_proteins:
        return result

    # Get all new proteins from expansion (excluding main and visible proteins)
    snapshot = result["snapshot_json"]
    new_proteins = [
        p for p in snapshot["proteins"]
        if p != protein_symbol and p not in visible_proteins
    ]

    if not new_proteins:
        # No new proteins added, no cross-links possible
        return result

    # Query for cross-links between new proteins and visible proteins
    # Example: new_proteins = ["BECN1"], visible_proteins = ["HDAC6"]
    # Find interactions where one protein is in new_proteins and other is in visible_proteins

    # Get Protein objects for lookups
    new_protein_objs = Protein.query.filter(Protein.symbol.in_(new_proteins)).all()
    visible_protein_objs = Protein.query.filter(Protein.symbol.in_(visible_proteins)).all()

    if not new_protein_objs or not visible_protein_objs:
        return result

    new_ids = [p.id for p in new_protein_objs]
    visible_ids = [p.id for p in visible_protein_objs]

    # Query interactions where one protein is new and other is visible
    # Need to check both orderings due to canonical storage
    cross_link_interactions = db.session.query(Interaction).filter(
        db.or_(
            db.and_(
                Interaction.protein_a_id.in_(new_ids),
                Interaction.protein_b_id.in_(visible_ids)
            ),
            db.and_(
                Interaction.protein_a_id.in_(visible_ids),
                Interaction.protein_b_id.in_(new_ids)
            )
        )
    ).all()

    # Add discovered cross-links to interactions list
    interactions_list = snapshot["interactions"]
    existing_ids = {
        f"{i.get('source', '')}-{i.get('target', '')}"
        for i in interactions_list
        if i.get('source') and i.get('target')
    }
    existing_ids.update({
        f"{i.get('target', '')}-{i.get('source', '')}"
        for i in interactions_list
        if i.get('source') and i.get('target')
    })

    for cross_ix in cross_link_interactions:
        # Extract FULL JSONB data
        cross_data = cross_ix.data.copy()

        # Determine source/target
        protein_a = cross_ix.protein_a
        protein_b = cross_ix.protein_b

        cross_data["source"] = protein_a.symbol
        cross_data["target"] = protein_b.symbol
        cross_data["type"] = "cross_link"  # Mark as discovered cross-link
        cross_data["direction"] = cross_ix.direction if cross_ix.direction else "bidirectional"

        # Ensure required fields have defaults
        if cross_data.get("confidence") is None:
            cross_data["confidence"] = 0.5
        if cross_data.get("arrow") is None:
            cross_data["arrow"] = cross_ix.arrow or "binds"
        if cross_data.get("functions") is None:
            cross_data["functions"] = []
        if cross_data.get("evidence") is None:
            cross_data["evidence"] = []
        if cross_data.get("pmids") is None:
            cross_data["pmids"] = []
        if cross_data.get("intent") is None:
            cross_data["intent"] = "binding"

        # Check if link already exists (avoid duplicates)
        link_id = f"{cross_data['source']}-{cross_data['target']}"
        rev_link_id = f"{cross_data['target']}-{cross_data['source']}"

        if link_id not in existing_ids and rev_link_id not in existing_ids:
            interactions_list.append(cross_data)
            existing_ids.add(link_id)

            # Add proteins to protein set if not already present
            if protein_a.symbol not in snapshot["proteins"]:
                snapshot["proteins"].append(protein_a.symbol)
            if protein_b.symbol not in snapshot["proteins"]:
                snapshot["proteins"].append(protein_b.symbol)

    # Re-sort proteins for consistency
    snapshot["proteins"] = sorted(snapshot["proteins"])

    # Update ctx_json
    result["ctx_json"]["proteins"] = snapshot["proteins"]
    result["ctx_json"]["interactions"] = interactions_list
    result["ctx_json"]["interactor_history"] = [p for p in snapshot["proteins"] if p != protein_symbol]

    return result


@app.route('/api/results/<protein>')
def get_results(protein):
    """
    Serves complete JSON data for a protein.

    Builds from PostgreSQL database (full snapshot_json + ctx_json).

    Returns:
        JSON: {snapshot_json: {...}, ctx_json: {...}}
    """
    # Build from PostgreSQL database
    try:
        result = build_full_json_from_db(protein)
        if result:
            return jsonify(result)
        else:
            return jsonify({"error": "Protein not found"}), 404
    except Exception as e:
        print(f"❌ Database query failed: {e}", file=sys.stderr)
        return jsonify({"error": "Database query failed"}), 500

@app.route('/api/visualize/<protein>')
def get_visualization(protein):
    """
    Generates and serves HTML visualization.

    Builds from PostgreSQL database, passes dict to visualizer.

    Returns:
        HTML string
    """
    # Build from PostgreSQL database
    try:
        result = build_full_json_from_db(protein)
        if result:
            # DEBUG: Log data structure for troubleshooting
            print(f"[DEBUG] Visualization for {protein}:", file=sys.stderr)
            print(f"  Result keys: {list(result.keys())}", file=sys.stderr)
            if 'snapshot_json' in result:
                snap = result['snapshot_json']
                print(f"  snapshot_json keys: {list(snap.keys())}", file=sys.stderr)
                print(f"  Main: {snap.get('main')}", file=sys.stderr)
                print(f"  Proteins count: {len(snap.get('proteins', []))}", file=sys.stderr)
                print(f"  Interactions count: {len(snap.get('interactions', []))}", file=sys.stderr)
                print(f"  Interactors count (legacy): {len(snap.get('interactors', []))}", file=sys.stderr)

            # Pass dict to visualizer
            from visualizer import create_visualization_from_dict
            html = create_visualization_from_dict(result)

            # Add cache-busting headers
            from flask import make_response
            response = make_response(html)
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        else:
            print(f"[DEBUG] Protein {protein} not found in database", file=sys.stderr)
            return "Protein not found.", 404
    except Exception as e:
        print(f"❌ Database visualization failed for {protein}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return "Database query failed.", 500


# ---------------------------
# PRUNED EXPANSION ENDPOINTS
# ---------------------------

def _get_api_key() -> str:
    return os.getenv("GOOGLE_API_KEY", "")

@app.post('/api/expand/pruned')
def expand_pruned():
    """
    Request a pruned subgraph for an expanded interactor with auto-cross-linking.

    NEW: Accepts visible_proteins to discover cross-links between new and visible proteins.

    Body: {
        "parent": "ATXN3",
        "protein": "VCP",
        "current_nodes": [...],
        "visible_proteins": ["HDAC6", "ATXN3"],  # NEW: for cross-linking
        "parent_edge": {...},
        "max_keep": 12?
    }

    Returns: { "status": "queued"|"complete"|"needs_full", "job_id": "prune:ATXN3:VCP" }
    """
    data = request.get_json(silent=True) or {}
    parent = (data.get("parent") or "").strip()
    protein = (data.get("protein") or "").strip()
    current_nodes = data.get("current_nodes") or []
    visible_proteins = data.get("visible_proteins") or []  # NEW: for cross-linking
    parent_edge = data.get("parent_edge") or {}
    max_keep = int(data.get("max_keep") or HARD_MAX_KEEP_DEFAULT)
    max_keep = min(max_keep, HARD_MAX_KEEP_DEFAULT)  # enforce hard maximum 12

    if not parent or not PROTEIN_RE.match(parent):
        return jsonify({"error":"Invalid parent"}), 400
    if not protein or not PROTEIN_RE.match(protein):
        return jsonify({"error":"Invalid protein"}), 400

    # NEW: Try to build from PostgreSQL database first (with cross-linking support)
    try:
        from models import Protein
        protein_in_db = Protein.query.filter_by(symbol=protein).first()
        if protein_in_db and protein_in_db.total_interactions > 0:
            # Define paths
            full_path = os.path.join(CACHE_DIR, f"{protein}.json")
            pruned_name = pruned_filename(parent, protein)
            pruned_path = os.path.join(PRUNED_DIR, pruned_name)
            job_id = make_prune_job_id(parent, protein)

            # CHECK: If fresh pruned file already exists, return immediately (avoid re-pruning)
            if os.path.exists(full_path) and is_pruned_fresh(Path(full_path), Path(pruned_path), hard_max_keep=max_keep):
                with jobs_lock:
                    jobs[job_id] = {"status": "complete"}
                print(f"[PRUNE CACHE HIT] Using cached pruned data for {protein}", file=sys.stderr)
                return jsonify({"status":"complete", "job_id": job_id}), 200

            # Build expansion JSON with cross-linking support
            expansion_data = build_expansion_json_from_db(protein, visible_proteins)
            if expansion_data:
                # Write to file for pruner to process (pruner requires file path input)
                import json

                # Write expansion data to full cache (for pruner to read)
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(expansion_data, f, indent=2, ensure_ascii=False)

                # Run pruning job
                api_key = _get_api_key()

                def _run():
                    try:
                        run_prune_job(
                            full_json_path=Path(full_path),
                            pruned_json_path=Path(pruned_path),
                            parent=parent,
                            current_nodes=current_nodes,
                            parent_edge=parent_edge,
                            hard_max_keep=max_keep,
                            api_key=api_key,
                            use_llm=False,
                        )
                        with jobs_lock:
                            jobs[job_id] = {"status": "complete"}
                    except Exception as e:
                        with jobs_lock:
                            jobs[job_id] = {"status": "error", "error": str(e)}

                with jobs_lock:
                    jobs[job_id] = {"status": "processing", "text": "Pruning subgraph with cross-links..."}
                t = threading.Thread(target=_run, daemon=True)
                t.start()
                return jsonify({"status":"queued", "job_id": job_id}), 202
    except Exception as e:
        print(f"[WARN]Database expansion failed, falling back to file cache: {e}", file=sys.stderr)

    # Fallback to old file cache logic
    full_path = os.path.join(CACHE_DIR, f"{protein}.json")
    pruned_name = pruned_filename(parent, protein)
    pruned_path = os.path.join(PRUNED_DIR, pruned_name)

    # Ensure full cache exists
    if not os.path.exists(full_path):
        return jsonify({"status":"needs_full", "job_id": make_prune_job_id(parent, protein)}), 200

    # If fresh pruned exists, return immediately
    if is_pruned_fresh(Path(full_path), Path(pruned_path), hard_max_keep=max_keep):
        with jobs_lock:
            jobs[make_prune_job_id(parent, protein)] = {"status": "complete"}
        return jsonify({"status":"complete", "job_id": make_prune_job_id(parent, protein)}), 200

    job_id = make_prune_job_id(parent, protein)
    api_key = _get_api_key()

    def _run():
        try:
            run_prune_job(
                full_json_path=Path(full_path),
                pruned_json_path=Path(pruned_path),
                parent=parent,
                current_nodes=current_nodes,
                parent_edge=parent_edge,
                hard_max_keep=max_keep,
                api_key=api_key,
                use_llm=False,
            )
            with jobs_lock:
                jobs[job_id] = {"status": "complete"}
        except Exception as e:
            with jobs_lock:
                jobs[job_id] = {"status": "error", "error": str(e)}

    with jobs_lock:
        jobs[job_id] = {"status": "processing", "text": "Pruning subgraph..."}
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status":"queued", "job_id": job_id}), 202

@app.get('/api/expand/status/<job_id>')
def expand_status(job_id):
    # If pruned file already exists, return complete
    try:
        parent, protein = parse_prune_job_id(job_id)
        full_path = Path(os.path.join(CACHE_DIR, f"{protein}.json"))
        pruned_path = Path(os.path.join(PRUNED_DIR, pruned_filename(parent, protein)))
        if full_path.exists() and is_pruned_fresh(full_path, pruned_path, HARD_MAX_KEEP_DEFAULT):
            return jsonify({"status":"complete"}), 200
    except Exception:
        pass
    with jobs_lock:
        st = jobs.get(job_id)
    if not st:
        return jsonify({"status":"unknown"}), 404
    return jsonify(st), 200

@app.get('/api/expand/results/<job_id>')
def expand_results(job_id):
    try:
        parent, protein = parse_prune_job_id(job_id)
    except Exception:
        return jsonify({"error":"invalid job id"}), 400
    fname = pruned_filename(parent, protein)
    path = os.path.join(PRUNED_DIR, fname)
    if not os.path.exists(path):
        return jsonify({"error":"not found"}), 404
    return send_from_directory(PRUNED_DIR, fname)


@app.route('/api/cancel/<protein>', methods=['POST'])
def cancel_job(protein):
    """Cancel a running job by setting its cancellation event."""
    if not protein:
        return jsonify({"error": "Protein name is required"}), 400

    with jobs_lock:
        job = jobs.get(protein)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        if job.get("status") != "processing":
            return jsonify({"error": "Job is not currently processing"}), 400

        # Set the cancellation event
        cancel_event = job.get("cancel_event")
        if cancel_event:
            cancel_event.set()
            job["status"] = "cancelling"
            job["progress"] = {"text": "Cancelling..."}
            return jsonify({"status": "cancelling", "message": "Cancellation requested"}), 200
        else:
            return jsonify({"error": "Job does not support cancellation"}), 400


# ---------------------------
# CHAT ENDPOINT
# ---------------------------

def _read_cache_json(protein: str) -> dict:
    """
    Read and parse cache JSON for a protein.

    Args:
        protein: Protein symbol to read cache for

    Returns:
        Parsed JSON dict, or empty dict if file doesn't exist or is invalid
    """
    try:
        json_path = os.path.join(CACHE_DIR, f"{protein}.json")
        if not os.path.exists(json_path):
            return {}

        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        # Log error but don't crash - return empty dict
        print(f"Warning: Failed to read cache for {protein}: {e}", file=sys.stderr)
        return {}


def _normalize_arrow_value(arrow: str) -> str:
    """Normalize arrow value to standard abbreviation."""
    if not isinstance(arrow, str):
        arrow = str(arrow) if arrow else ""
    arrow_lower = arrow.lower().strip()
    if "activ" in arrow_lower:
        return "act"
    elif "inhib" in arrow_lower:
        return "inh"
    elif "regul" in arrow_lower or "modulat" in arrow_lower:
        return "reg"
    elif "bind" in arrow_lower:
        return "bind"
    else:
        return "unk"


def _normalize_direction_value(direction: str) -> str:
    """Normalize direction value to standard abbreviation."""
    if not isinstance(direction, str):
        direction = str(direction) if direction else ""
    direction_lower = direction.lower().strip()
    if "bidir" in direction_lower:
        return "bidir"
    elif "main_to_primary" in direction_lower:
        return "m2p"
    elif "primary_to_main" in direction_lower:
        return "p2m"
    else:
        return "unk"


def _extract_compact_functions(raw_functions: list) -> list:
    """
    Helper to extract compact function data from raw functions array.
    Used by both new and old format processing.

    Args:
        raw_functions: Raw functions array from interaction data

    Returns:
        List of compact function dicts
    """
    functions = []
    if not isinstance(raw_functions, list):
        return functions

    for fn in raw_functions[:5]:  # Limit to 5 functions per interaction
        if not isinstance(fn, dict):
            continue

        # Safe confidence extraction
        try:
            fn_confidence = float(fn.get("confidence", 0.0))
        except (ValueError, TypeError):
            fn_confidence = 0.0

        compact_fn = {
            "name": str(fn.get("function", "Unknown")).strip(),
            "arrow": _normalize_arrow_value(fn.get("arrow", "")),
            "confidence": fn_confidence,
            "pmids": [],
            "effect": str(fn.get("effect_description", "")).strip(),
            "biological_consequence": [],
            "specific_effects": []
        }

        # Extract function PMIDs (limit to 5)
        fn_pmids = fn.get("pmids", [])
        if isinstance(fn_pmids, list):
            compact_fn["pmids"] = [str(p) for p in fn_pmids[:5] if p]

        # Extract biological consequence (arrow chain)
        bio_cons = fn.get("biological_consequence", [])
        if isinstance(bio_cons, list):
            compact_fn["biological_consequence"] = [
                str(b).strip() for b in bio_cons[:5] if b
            ]

        # Extract specific effects
        spec_eff = fn.get("specific_effects", [])
        if isinstance(spec_eff, list):
            compact_fn["specific_effects"] = [
                str(e).strip() for e in spec_eff[:3] if e
            ]

        functions.append(compact_fn)

    return functions


def _build_compact_rich_context(parent: str, visible_proteins: list) -> dict:
    """
    Build compact rich context with all 4 key elements:
    - Biological cascades
    - Functions
    - Interaction summaries
    - Downstream effects

    NEW: Reads data for ALL visible proteins (not just parent) to capture expanded nodes.
    Reads from PostgreSQL database first (NEW format), falls back to file cache (OLD format).
    Handles both NEW format (proteins[], interactions[]) and OLD format (interactors[]).

    Args:
        parent: Root protein symbol
        visible_proteins: List of currently visible protein symbols

    Returns:
        Dict with main protein and compact interaction data
    """
    # Build visible set for filtering
    visible_set = set(visible_proteins)

    # Track all interactions using canonical key for deduplication
    # Key: "PROTEIN_A-PROTEIN_B" (alphabetically sorted to handle bidirectional)
    interactions_map = {}

    # Read data for ALL visible proteins (not just parent)
    for protein in visible_proteins:
        # Try to build from PostgreSQL database first (NEW format)
        snapshot = None
        try:
            db_result = build_full_json_from_db(protein)
            if db_result:
                snapshot = db_result.get("snapshot_json", db_result)
        except Exception as e:
            print(f"[WARN]Database query failed for {protein}: {e}", file=sys.stderr)

        # Fallback to file cache (OLD format)
        if not snapshot:
            root_data = _read_cache_json(protein)
            if root_data:
                snapshot = root_data.get("snapshot_json", root_data)

        if not snapshot or not isinstance(snapshot, dict):
            continue

        # Check format: NEW (interactions array) vs OLD (interactors array)
        raw_interactions = snapshot.get("interactions", None)

        if raw_interactions is not None and isinstance(raw_interactions, list):
            # ===== NEW FORMAT: interactions array =====
            for interaction in raw_interactions:
                if not isinstance(interaction, dict):
                    continue

                source = interaction.get("source", "")
                target = interaction.get("target", "")

                # Only include interactions where BOTH endpoints are visible
                if not source or not target:
                    continue
                if source not in visible_set or target not in visible_set:
                    continue

                # Create canonical key (alphabetically sorted)
                canonical_key = "-".join(sorted([source, target]))

                # Skip if already processed (deduplicate)
                if canonical_key in interactions_map:
                    continue

                # Build compact interaction
                try:
                    confidence = float(interaction.get("confidence", 0.0))
                except (ValueError, TypeError):
                    confidence = 0.0

                compact_inter = {
                    "source": str(source),
                    "target": str(target),
                    "type": str(interaction.get("type", "direct")),
                    "arrow": _normalize_arrow_value(interaction.get("arrow", "")),
                    "direction": _normalize_direction_value(interaction.get("direction", "")),
                    "confidence": confidence,
                    "pmids": [],
                    "summary": str(interaction.get("support_summary", "")).strip(),
                    "functions": []
                }

                # Extract PMIDs (limit to 5, store as strings)
                raw_pmids = interaction.get("pmids", [])
                if isinstance(raw_pmids, list):
                    compact_inter["pmids"] = [str(p) for p in raw_pmids[:5] if p]

                # Extract functions using helper
                compact_inter["functions"] = _extract_compact_functions(interaction.get("functions", []))

                interactions_map[canonical_key] = compact_inter

        else:
            # ===== OLD FORMAT: interactors array (transform to new) =====
            interactors = snapshot.get("interactors", [])
            if not isinstance(interactors, list):
                continue

            main_protein = snapshot.get("main", protein)

            for inter in interactors:
                if not isinstance(inter, dict):
                    continue

                primary = inter.get("primary", "")
                # Only include if primary is visible
                if not primary or primary not in visible_set:
                    continue

                # Create canonical key (alphabetically sorted)
                canonical_key = "-".join(sorted([main_protein, primary]))

                # Skip if already processed (deduplicate)
                if canonical_key in interactions_map:
                    continue

                # Build compact interaction (transform old format to new)
                try:
                    confidence = float(inter.get("confidence", 0.0))
                except (ValueError, TypeError):
                    confidence = 0.0

                compact_inter = {
                    "source": str(main_protein),
                    "target": str(primary),
                    "type": "direct",  # old format only had direct interactions
                    "arrow": _normalize_arrow_value(inter.get("arrow", "")),
                    "direction": _normalize_direction_value(inter.get("direction", "")),
                    "confidence": confidence,
                    "pmids": [],
                    "summary": str(inter.get("support_summary", "")).strip(),
                    "functions": []
                }

                # Extract PMIDs (limit to 5, store as strings)
                raw_pmids = inter.get("pmids", [])
                if isinstance(raw_pmids, list):
                    compact_inter["pmids"] = [str(p) for p in raw_pmids[:5] if p]

                # Extract functions using helper
                compact_inter["functions"] = _extract_compact_functions(inter.get("functions", []))

                interactions_map[canonical_key] = compact_inter

    # Convert map to list
    interactions = list(interactions_map.values())

    return {
        "main": str(parent),
        "interactions": interactions
    }


def _build_chat_system_prompt(parent: str, rich_context: dict) -> str:
    """
    Build system prompt with compact rich context using 2-3 letter abbreviations.

    Includes ALL 4 key elements:
    - Biological cascades (BC)
    - Functions (FN)
    - Interaction summaries (SUM)
    - Downstream effects (SE)

    Args:
        parent: Root protein symbol
        rich_context: Rich context dict from _build_compact_rich_context

    Returns:
        Complete system prompt string with legend and data
    """

    # Build abbreviation legend
    legend = """ABBREVIATION LEGEND:
SUM=summary | FN=function | EFF=effect
BC=biological_consequence | SE=specific_effects

Arrows: -> (activates), -| (inhibits), <-> (binds)"""

    main_protein = rich_context.get("main", parent)
    interactions = rich_context.get("interactions", [])

    # Build interactions section
    interactions_lines = []
    interactions_lines.append(f"ROOT PROTEIN: {main_protein}")
    interactions_lines.append("")

    if not interactions:
        interactions_lines.append("No interaction data available in current view.")
    else:
        interactions_lines.append("INTERACTIONS:")
        interactions_lines.append("")

        for i, inter in enumerate(interactions, 1):
            # NEW: Use source/target instead of primary
            source = inter.get("source", "Unknown")
            target = inter.get("target", "Unknown")
            interaction_type = inter.get("type", "direct")
            # Normalize arrow/direction values for consistent rendering
            arrow = _normalize_arrow_value(inter.get("arrow", "unk"))
            direction = _normalize_direction_value(inter.get("direction", "unk"))
            confidence = inter.get("confidence", 0.0)
            pmids = inter.get("pmids", [])
            summary = inter.get("summary", "")

            # Build SOURCE + ARROW + TARGET line based on direction
            if direction == "bidir":
                # Bidirectional - format: ATXN3 <-> VCP (binds)
                if arrow == "bind":
                    interaction_line = f"{source} <-> {target}"
                elif arrow == "act":
                    interaction_line = f"{source} <-> {target} (activates)"
                elif arrow == "inh":
                    interaction_line = f"{source} <-> {target} (inhibits)"
                elif arrow == "reg":
                    interaction_line = f"{source} <-> {target} (regulates)"
                else:
                    interaction_line = f"{source} <-> {target}"
            elif direction == "m2p":
                # Main to primary (source -> target)
                if arrow == "act":
                    interaction_line = f"{source} -> {target}"
                elif arrow == "inh":
                    interaction_line = f"{source} -| {target}"
                elif arrow == "bind":
                    interaction_line = f"{source} -> {target} (binds)"
                elif arrow == "reg":
                    interaction_line = f"{source} -> {target} (regulates)"
                else:
                    interaction_line = f"{source} -> {target}"
            elif direction == "p2m":
                # Primary to main (target -> source)
                if arrow == "act":
                    interaction_line = f"{target} -> {source}"
                elif arrow == "inh":
                    interaction_line = f"{target} -| {source}"
                elif arrow == "bind":
                    interaction_line = f"{target} -> {source} (binds)"
                elif arrow == "reg":
                    interaction_line = f"{target} -> {source} (regulates)"
                else:
                    interaction_line = f"{target} -> {source}"
            else:
                # Unknown direction - show source -> target with arrow
                if arrow == "act":
                    interaction_line = f"{source} -> {target}"
                elif arrow == "inh":
                    interaction_line = f"{source} -| {target}"
                elif arrow == "bind":
                    interaction_line = f"{source} <-> {target}"
                elif arrow == "reg":
                    interaction_line = f"{source} -> {target} (regulates)"
                else:
                    interaction_line = f"{source} - {target}"

            # Compact header line with interaction
            header = f"{i}. {interaction_line}"
            interactions_lines.append(header)

            # Summary (keep verbose for readability)
            if summary:
                interactions_lines.append(f"   SUM: {summary}")

            # Functions section
            functions = inter.get("functions", [])
            if functions:
                interactions_lines.append("   Functions:")
                for j, fn in enumerate(functions, 1):
                    fn_name = fn.get("name", "Unknown")
                    fn_arrow = fn.get("arrow", "unk")
                    fn_conf = fn.get("confidence", 0.0)
                    fn_pmids = fn.get("pmids", [])
                    fn_effect = fn.get("effect", "")
                    bio_cons = fn.get("biological_consequence", [])
                    spec_effs = fn.get("specific_effects", [])

                    # Function header
                    interactions_lines.append(f"     F{j}: {fn_name} ACT:{fn_arrow}")

                    # Effect description (verbose for biological context)
                    if fn_effect:
                        interactions_lines.append(f"         EFF: {fn_effect}")

                    # Biological consequence (arrow chain)
                    if bio_cons:
                        bc_chain = " -> ".join(bio_cons)
                        interactions_lines.append(f"         BC: {bc_chain}")

                    # Specific effects (list)
                    if spec_effs:
                        se_list = "; ".join(spec_effs)
                        interactions_lines.append(f"         SE: {se_list}")

            interactions_lines.append("")  # Blank line between interactions

    interactions_text = "\n".join(interactions_lines)

    # Build complete prompt with pipeline-style excellence
    full_prompt = f"""╔═══════════════════════════════════════════════════════════════╗
║  PROTEIN INTERACTION NETWORK ANALYSIS ASSISTANT               ║
║  Expert Molecular Biology Q&A System                          ║
╚═══════════════════════════════════════════════════════════════╝

ROLE & EXPERTISE:
You are a senior molecular biologist and biochemist providing expert consultation
on protein-protein interaction networks. Your audience consists of research scientists,
graduate students, and clinicians who need ACCURATE, EVIDENCE-BASED answers about
protein interactions, functional outcomes, and biological mechanisms.

╔═══════════════════════════════════════════════════════════════╗
║  CRITICAL OPERATIONAL RULES (ABSOLUTE OVERRIDE)              ║
╚═══════════════════════════════════════════════════════════════╝

STRICT EVIDENCE BOUNDARIES:
- Answer ONLY using the interaction data provided below
- NEVER extrapolate beyond the visible network context
- If asked about proteins/interactions NOT in the data: explicitly state "Not in current view"
- If data is ambiguous or incomplete: acknowledge uncertainty rather than speculate
- NEVER invent PMIDs, paper citations, or experimental details

ACCURACY > COMPLETENESS:
- A precise partial answer beats a comprehensive guess
- Distinguish between direct interactions and downstream consequences
- Note when evidence is human vs model organism

OUTPUT FORMATTING:
- Use clear, professional scientific prose (NOT markdown)
- NO asterisks, underscores, headers, bullets, or special formatting
- Write as if explaining to a colleague at a lab meeting
- Keep responses CONCISE (2-4 sentences) unless depth is explicitly requested

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NETWORK DATA LEGEND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{legend}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT NETWORK CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{interactions_text}

╔═══════════════════════════════════════════════════════════════╗
║  EXPERT RESPONSE FRAMEWORK                                    ║
╚═══════════════════════════════════════════════════════════════╝

WHEN ANSWERING ABOUT INTERACTIONS:
1. Identify the relevant interaction(s) from the data above
2. State directionality clearly (X activates Y, Y inhibits X, bidirectional)
3. Explain biological context only if present in the data

WHEN ANSWERING ABOUT FUNCTIONS:
1. Link function to the SPECIFIC interaction that drives it
2. Distinguish between:
   - Effect (EFF): What happens to the function immediately
   - Biological consequences (BC): Downstream signaling cascades
   - Specific effects (SE): Direct molecular outcomes
3. Use arrow notation where appropriate (e.g., "TP53 stabilization leads to BAX upregulation")

WHEN DISCUSSING BIOLOGICAL SIGNIFICANCE:
1. Integrate information across multiple interactions when asked
2. Connect interaction mechanisms to functional outcomes
3. Explain cascades step-by-step when asked about pathways
4. Relate to disease contexts only if present in the data
5. Acknowledge gaps: "Function X is documented but mechanism details are not available"

WHEN HANDLING AMBIGUOUS QUESTIONS:
- If question is too broad: "Could you clarify which aspect/protein you're interested in?"
- If query protein not in network: "That protein is not in the current network view"
- If mechanism unclear from data: "The data shows interaction but mechanism is not specified"

RESPONSE LENGTH CALIBRATION:
- Brief query (e.g., "Does X interact with Y?"): 1-2 sentences
- Mechanism query (e.g., "How does X regulate Y?"): 2-4 sentences with key details
- Comprehensive query (e.g., "Explain X's role in pathway Z"): 4-6 sentences, integrate multiple functions
- Explicit detail request (e.g., "Give me all the details"): Expand fully with all cascades and effects

DOMAIN-SPECIFIC INTELLIGENCE:
- Recognize common post-translational modifications (phosphorylation, ubiquitination, etc.)
- Understand arrow semantics: activation vs inhibition vs binding
- Distinguish between interaction directionality (who regulates whom)
- Interpret biological consequences as signaling cascades
- Translate abbreviated data (BC, SE, EFF) into prose seamlessly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLES OF EXCELLENT RESPONSES:

Q: "Does ATXN3 interact with VCP?"
A: "Yes, ATXN3 directly interacts with VCP. ATXN3 binds VCP through its ubiquitin-binding domain, supported by Co-IP and pull-down assays in human cells."

Q: "What functions does the ATXN3-VCP interaction regulate?"
A: "The ATXN3-VCP interaction regulates protein quality control and autophagy. VCP binding enhances ATXN3's deubiquitinase activity, leading to substrate stabilization. This activates autophagy pathways through mTOR signaling modulation and promotes clearance of misfolded proteins via the ERAD pathway."

Q: "Tell me about the biological consequences"
A: "The interaction triggers multiple cascades. First, ATXN3 deubiquitinates VCP substrates, preventing their proteasomal degradation and stabilizing protein levels. This stabilization activates downstream autophagy machinery through BECN1 recruitment and LC3 lipidation. Additionally, VCP-ATXN3 complexes facilitate ER-associated degradation by extracting ubiquitinated proteins from the ER membrane, which maintains ER homeostasis under proteotoxic stress."

Q: "Is there evidence for this in disease?"
A: "The current network data does not include disease-specific contexts or patient studies. The interactions and functions shown are based on cell biology experiments in human cell lines."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are now ready to answer questions. Provide accurate, evidence-based responses
using ONLY the network data shown above. Maintain expert-level rigor."""

    return full_prompt


def _build_compact_state_from_request(state_data: dict) -> dict:
    """
    Extract and validate visible protein list from frontend request.

    Args:
        state_data: Raw state dict from frontend

    Returns:
        Dict with parent protein and list of visible proteins
    """
    if not isinstance(state_data, dict):
        return {"parent": "", "visible_proteins": []}

    parent = str(state_data.get("parent", "")).strip()
    visible_proteins = state_data.get("visible_proteins", [])

    # Validate and clean visible proteins list
    clean_visible = []
    if isinstance(visible_proteins, list):
        for protein in visible_proteins:
            if protein and isinstance(protein, str):
                clean_protein = str(protein).strip()
                if clean_protein and PROTEIN_RE.match(clean_protein):
                    clean_visible.append(clean_protein)

    return {
        "parent": parent,
        "visible_proteins": clean_visible
    }


def _call_chat_llm(messages: list, system_prompt: str, max_history: int = 10) -> str:
    """
    Call Gemini LLM for chat response using runner.py patterns.

    Args:
        messages: List of {role, content} message dicts
        system_prompt: System instructions
        max_history: Maximum message history to send (default 10, configurable)

    Returns:
        LLM response text

    Raises:
        RuntimeError: If API call fails
    """
    from google import genai as google_genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not configured")

    client = google_genai.Client(api_key=api_key)

    # Trim message history to max_history (keep most recent)
    # Always keep system context fresh
    trimmed_messages = messages[-max_history:] if len(messages) > max_history else messages

    # Build config with correct camelCase parameter names
    # systemInstruction (singular, camelCase) can accept string directly
    # Note: maxOutputTokens must be large enough to accommodate system prompt tokens + output
    config = types.GenerateContentConfig(
        systemInstruction=system_prompt,
        maxOutputTokens=5096,  # Needs to be large for 40K+ char system prompts
        temperature=0.2,  # Lower for factual, deterministic answers
        topP=0.85,  # Focused sampling for consistent responses
    )

    # Convert messages to Gemini format
    # Note: Gemini expects alternating user/model turns
    gemini_contents = []
    for msg in trimmed_messages:
        role = "model" if msg.get("role") == "assistant" else "user"
        content = msg.get("content", "")
        if content.strip():
            gemini_contents.append(types.Content(
                role=role,
                parts=[types.Part(text=content)]
            ))

    # Safety check: ensure we have at least one message
    if not gemini_contents:
        raise RuntimeError("No valid messages to send to LLM")

    # Safety check: ensure last message is from user (Gemini requirement)
    # The API expects the conversation to end with a user message
    if gemini_contents[-1].role != "user":
        raise RuntimeError("Last message must be from user (Gemini API requirement)")

    # Retry logic matching runner.py pattern
    max_retries = 3
    base_delay = 1.5

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=gemini_contents,
                config=config,
            )

            # Extract text from response (matching runner.py pattern)
            if hasattr(response, 'text'):
                text = response.text
                # Handle both None and string "None"
                text_str = str(text).strip() if text else ''
                if text_str and text_str != 'None':
                    return text
                else:
                    # Check finish reason to provide better error message
                    finish_reason = 'UNKNOWN'
                    if hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
                        if hasattr(response.candidates[0], 'finish_reason'):
                            finish_reason = str(response.candidates[0].finish_reason)
                    print(f"Warning: Empty response (finish_reason: {finish_reason})", file=sys.stderr)
                    raise RuntimeError(f"LLM returned empty response (finish_reason: {finish_reason})")
            elif hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
                # Safely access first candidate
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    text = ''.join(part.text for part in parts if hasattr(part, 'text'))
                    if text and text.strip():  # Ensure we got non-empty, non-whitespace text
                        return text
                    else:
                        print(f"Warning: Extracted text is empty or whitespace: '{text}'", file=sys.stderr)
                raise RuntimeError("No text found in LLM response candidates")
            else:
                raise RuntimeError("No text in LLM response")

        except Exception as e:
            print(f"Chat LLM attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}", file=sys.stderr)
            delay = base_delay * (attempt ** 1.5)
            if attempt < max_retries:
                print(f"Retrying in {delay:.1f}s...", file=sys.stderr)
                time.sleep(delay)
            else:
                raise RuntimeError(f"Chat LLM call failed after {max_retries} attempts: {e}")

    raise RuntimeError("Unexpected error in chat LLM call")


@app.post('/api/chat')
def chat():
    """
    Handle chat messages with LLM assistance.

    Request body:
    {
        "parent": "ATXN3",           // Root protein
        "protein": "VCP",            // Optional: current focus protein
        "messages": [...],           // Chat history [{role, content}]
        "state": {...},              // Compact graph state
        "max_history": 10            // Optional: max messages to send to LLM
    }

    Response:
    {
        "reply": "LLM response text"
    }
    or
    {
        "error": "Error message"
    }
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON request"}), 400

        # Extract and validate required fields
        parent = (data.get("parent") or "").strip()
        if not parent or not PROTEIN_RE.match(parent):
            return jsonify({"error": "Invalid or missing parent protein"}), 400

        messages = data.get("messages")
        if not isinstance(messages, list) or len(messages) == 0:
            return jsonify({"error": "Invalid or empty messages list"}), 400

        # Validate message structure
        for msg in messages:
            if not isinstance(msg, dict):
                return jsonify({"error": "Invalid message format"}), 400
            if "role" not in msg or "content" not in msg:
                return jsonify({"error": "Message missing role or content"}), 400
            if msg["role"] not in ["user", "assistant"]:
                return jsonify({"error": f"Invalid message role: {msg['role']}"}), 400

        # Extract last user message must exist
        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return jsonify({"error": "Last message must be from user"}), 400

        state_data = data.get("state", {})
        if not isinstance(state_data, dict):
            return jsonify({"error": "Invalid state format"}), 400

        max_history = data.get("max_history", 10)
        if not isinstance(max_history, int) or max_history < 1 or max_history > 50:
            max_history = 10  # Safe default

        # Extract and validate state (now expects parent + visible_proteins)
        compact_state = _build_compact_state_from_request(state_data)
        state_parent = compact_state.get("parent", "")
        visible_proteins = compact_state.get("visible_proteins", [])

        # Use state_parent if provided, otherwise fall back to parent from root
        final_parent = state_parent if state_parent else parent

        # Validate we have a valid parent
        if not final_parent or not PROTEIN_RE.match(final_parent):
            return jsonify({"error": "Invalid parent protein in state"}), 400

        # Build rich context by reading from cache JSON
        rich_context = _build_compact_rich_context(final_parent, visible_proteins)
        print(f"Chat: Built context with {len(rich_context.get('interactions', []))} interactions", file=sys.stderr)

        # Build system prompt with rich context
        system_prompt = _build_chat_system_prompt(final_parent, rich_context)
        print(f"Chat: System prompt length: {len(system_prompt)} chars", file=sys.stderr)

        # Call LLM
        response_text = _call_chat_llm(messages, system_prompt, max_history=max_history)
        print(f"Chat: Got response: {len(response_text) if response_text else 0} chars", file=sys.stderr)

        if not response_text or not response_text.strip():
            print(f"Chat ERROR: Response text is empty after LLM call", file=sys.stderr)
            return jsonify({"error": "LLM returned empty response"}), 500

        return jsonify({"reply": response_text}), 200

    except RuntimeError as e:
        # LLM-specific errors
        return jsonify({"error": f"LLM error: {str(e)}"}), 500
    except Exception as e:
        # Unexpected errors
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5007, debug=False, threaded=True, use_reloader=True)

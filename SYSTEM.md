# ProPaths System Architecture

> **30-Second Summary**: Bioinformatics web app that uses Gemini LLM to research protein-protein interactions (PPIs) from literature, stores results in PostgreSQL, and visualizes them as interactive D3.js force-directed graphs with evidence, functions, and biological cascades.

---

## Tech Stack

### Backend
- **Flask** (Python 3.10+) - Web server + API endpoints
- **PostgreSQL** (Railway) - Primary database with JSONB storage
- **SQLAlchemy** - ORM for database operations
- **Google Gemini 2.5 Pro** - LLM for research pipeline (with Google Search grounding)
- **Threading** - Background job execution with status tracking

### Frontend
- **D3.js v7** - Force-directed graph visualization
- **Vanilla JS** - UI interactions, data parsing, multi-job tracking
- **HTML/CSS** - Landing page, visualization page, modals
- **SessionStorage** - Job persistence across page navigation (viz page only)

### Infrastructure
- **Local Dev**: Flask server → Railway PostgreSQL (via `DATABASE_PUBLIC_URL`)
- **Production**: Railway deployment (via `DATABASE_URL`)
- **File Cache**: `cache/` directory for backups, intermediate storage, pruned expansions

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER FLOW                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. SEARCH (Frontend)                                             │
│    templates/index.html + static/script.js                       │
│    → POST /api/query {"protein": "ATXN3"}                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. RESEARCH PIPELINE (Backend - Background Thread)               │
│    app.py: run_full_job() → runner.py: run_pipeline()            │
│                                                                   │
│    Phase 1: Interactor Discovery (3-8 rounds)                    │
│    ├─ Gemini + Google Search → Find protein names               │
│    └─ Classify direct vs indirect (cascade chains)              │
│                                                                   │
│    Phase 2: Function Discovery (3-8 rounds)                      │
│    ├─ Gemini → Find mechanisms, evidence, PMIDs                 │
│    └─ Build biological consequence chains                        │
│                                                                   │
│    Phase 3: Enrichment & Validation                              │
│    ├─ evidence_validator.py: Add quotes, validate PMIDs         │
│    ├─ claim_fact_checker.py: Verify claims (optional)           │
│    ├─ schema_validator.py: Fix structural issues                │
│    ├─ deduplicate_functions.py: Remove duplicates               │
│    └─ interaction_metadata_generator.py: Generate metadata      │
│                                                                   │
│    Phase 4: Database Sync                                        │
│    └─ db_sync.py: Write to PostgreSQL + file cache              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. DATABASE STORAGE (PostgreSQL)                                 │
│    models.py: Protein + Interaction tables                       │
│                                                                   │
│    Protein Table:                                                │
│    └─ symbol, query_count, total_interactions                   │
│                                                                   │
│    Interaction Table (JSONB payload):                            │
│    ├─ protein_a_id < protein_b_id (canonical ordering)          │
│    ├─ direction, arrow, confidence (denormalized)                │
│    └─ data: {functions[], evidence[], pmids[], ...}             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. VISUALIZATION (Frontend)                                      │
│    app.py: build_full_json_from_db() → visualizer.py            │
│                                                                   │
│    Embedded JS (visualizer.js):                                  │
│    ├─ Parse snapshot_json (proteins[], interactions[])          │
│    ├─ Build D3 force graph (nodes + links)                      │
│    ├─ Render modals (evidence, functions, cascades)             │
│    ├─ Table view (sortable, filterable, exportable)             │
│    └─ Chat interface (LLM-powered Q&A)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

### 🔹 Core Application (Root)
```
app.py                    # Flask routes, API endpoints, job orchestration (1878 LOC)
runner.py                 # Pipeline execution engine, LLM orchestration (3150 LOC)
models.py                 # SQLAlchemy ORM (Protein, Interaction tables)
visualizer.py             # HTML template generator with embedded D3.js
requirements.txt          # Python dependencies
.env                      # Secrets (GOOGLE_API_KEY, DATABASE_URL)
```

### 🔹 Pipeline Configuration
```
pipeline/
├── config_gemini_MAXIMIZED.py  # Base pipeline (7 rounds interactors, 6 rounds functions)
├── config_dynamic.py            # Dynamic config generator (supports variable rounds)
├── types.py                     # StepConfig dataclass for pipeline steps
└── pipeline.py                  # Legacy CLI wrapper (NOT used by web app)
```

### 🔹 Utils (Processing & Validation)
```
utils/
├── db_sync.py                   # Database synchronization layer (canonical ordering)
├── pruner.py                    # LLM/heuristic-based subgraph selection (expansion)
├── evidence_validator.py        # Validate/enrich evidence, add quotes, normalize
├── claim_fact_checker.py        # LLM-based fact-checking of claims
├── schema_validator.py          # Pre/post validation, fix structural issues
├── deduplicate_functions.py     # Remove duplicate function entries
├── interaction_metadata_generator.py  # Generate interaction metadata
├── clean_function_names.py      # Normalize function names
├── update_cache_pmids.py        # Update/validate PMIDs in cached results
├── pmid_extractor.py            # Standalone CLI tool for PMID lookup
├── pubmed_match.py              # Paper search and similarity matching
├── protein_database.py          # File-based protein cache (legacy, still used for helpers)
├── step_logger.py               # Comprehensive logging for pipeline steps
└── llm_response_parser.py       # Parse LLM JSON responses
```

### 🔹 Frontend (Web Interface)
```
static/
├── script.js                    # Landing page logic, JobTracker class for multi-job tracking
├── visualizer.js                # Graph rendering, VizJobTracker class, modals (6800+ LOC)
├── styles.css                   # Shared site styles, job card styles
└── viz-styles.css               # Visualization-specific styles, mini job chip styles

templates/
└── index.html                   # Landing page template
```

### 🔹 Data Storage
```
cache/
├── <PROTEIN>.json               # Full graph snapshots (fallback, intermediate)
├── pruned/                      # Pruned subgraphs for expansion
│   └── <PARENT>_for_<PROTEIN>.json
└── proteins/                    # Legacy file-based storage (being phased out)
    └── <PROTEIN>/
        ├── metadata.json
        └── interactions/
            └── <PARTNER>.json
```

### 🔹 Testing
```
tests/
├── test.py                      # Simple test runner
└── test_server.py               # Server endpoint tests
```

### 🔹 Logs
```
Logs/
└── <PROTEIN>/
    └── <TIMESTAMP>/             # Pipeline execution logs
```

### ⚠️ **LEGACY / TO REMOVE**
```
migrate_*.py                     # 12 one-time migration scripts (NO LONGER NEEDED)
├── migrate_add_arrows.py
├── migrate_add_chain_arrows.py
├── migrate_add_function_context.py
├── migrate_add_interaction_columns.py
├── migrate_add_missing_columns.py
├── migrate_cache.py
├── migrate_deduplicate.py
├── migrate_fix_direction_semantics.py
├── migrate_fix_indirect_corruption.py
├── migrate_indirect_chains.py
├── migrate_restore_functions_from_cache.py
└── migrate_to_postgres.py

sync_cache_to_db.py              # Manual sync tool (deprecated, db_sync.py is automatic)
visualizer copy.py               # Old backup (DELETE)
propaths_refactor_*.txt/md       # Refactor notes (archive or delete)
```

---

## Data Schemas

### 1. **Database Models** (`models.py`)

#### Protein Table
```python
class Protein(db.Model):
    id: int                      # Primary key
    symbol: str                  # e.g., "ATXN3" (unique, indexed)
    first_queried: datetime
    last_queried: datetime
    query_count: int
    total_interactions: int
    extra_data: JSONB            # Flexible metadata
```

#### Interaction Table
```python
class Interaction(db.Model):
    id: int                      # Primary key
    protein_a_id: int            # FK to proteins (always < protein_b_id)
    protein_b_id: int            # FK to proteins (canonical ordering)

    # Denormalized fields (for fast filtering)
    confidence: Numeric(3,2)     # 0.00 to 1.00
    direction: str               # 'a_to_b', 'b_to_a', 'bidirectional'
    arrow: str                   # 'binds', 'activates', 'inhibits', 'regulates'
    interaction_type: str        # 'direct' or 'indirect'
    upstream_interactor: str     # Upstream protein symbol (for indirect)
    mediator_chain: JSONB        # Chain path (for indirect)
    depth: int                   # 1=direct, 2+=indirect

    # FULL PAYLOAD (JSONB)
    data: JSONB                  # Complete interactor JSON (functions, evidence, PMIDs, etc.)

    # Discovery metadata
    discovered_in_query: str     # Which protein query found this
    discovery_method: str        # 'pipeline', 'requery', 'manual'
```

**Key Design Choice**: Canonical ordering (`protein_a_id < protein_b_id`) prevents duplicate storage of symmetric interactions. The `direction` field is converted to/from query-relative perspective during reads/writes.

### 2. **JSON Schemas**

#### snapshot_json (NEW FORMAT - Current)
```json
{
  "main": "ATXN3",
  "proteins": ["ATXN3", "VCP", "HDAC6", ...],
  "interactions": [
    {
      "source": "ATXN3",
      "target": "VCP",
      "type": "direct",
      "direction": "bidirectional",
      "arrow": "binds",
      "confidence": 0.85,
      "intent": "ubiquitination",
      "support_summary": "...",
      "pmids": ["17580304", ...],
      "evidence": [
        {
          "pmid": "17580304",
          "doi": "10.1074/...",
          "paper_title": "...",
          "authors": "...",
          "journal": "J Biol Chem",
          "year": 2007,
          "assay": "Co-IP",
          "species": "human",
          "relevant_quote": "..."
        }
      ],
      "functions": [
        {
          "function": "Protein Quality Control",
          "arrow": "activates",
          "cellular_process": "...",
          "effect_description": "...",
          "biological_consequence": [
            "ATXN3 → VCP → p97 ATPase → Protein degradation"
          ],
          "specific_effects": ["..."],
          "pmids": ["..."],
          "evidence": [...]
        }
      ],
      "interaction_type": "direct",
      "upstream_interactor": null,
      "mediator_chain": [],
      "depth": 1
    },
    {
      "source": "VCP",
      "target": "HDAC6",
      "type": "shared",
      "_is_shared_link": true
    }
  ]
}
```

#### snapshot_json (OLD FORMAT - Legacy Fallback)
```json
{
  "main": "ATXN3",
  "interactors": [
    {
      "primary": "VCP",
      "direction": "bidirectional",
      "arrow": "binds",
      "functions": [...],
      "evidence": [...],
      "pmids": [...]
    }
  ]
}
```

**Key Design Choice**: New format separates proteins and interactions for cleaner graph rendering. Old format is automatically transformed by frontend (`visualizer.js:190-283`).

#### ctx_json (Context / Metadata)
```json
{
  "main": "ATXN3",
  "interactors": [...],           // Same as snapshot_json.interactions
  "interactor_history": ["VCP", "HDAC6", ...],
  "function_history": {
    "VCP": ["Autophagy", "ER-associated degradation", ...]
  },
  "function_batches": ["VCP", "HDAC6", ...],
  "search_history": ["ATXN3 protein interactions", ...]
}
```

### 3. **Interaction Types**

#### Direct Interactions (Physical)
- `interaction_type: "direct"`
- `upstream_interactor: null`
- `mediator_chain: []`
- `depth: 1`

#### Indirect Interactions (Cascade Chains)
- `interaction_type: "indirect"`
- `upstream_interactor: "VCP"` (mediator protein)
- `mediator_chain: ["VCP"]` or `["VCP", "LAMP2"]` (multi-hop)
- `depth: 2+` (number of hops from query protein)

#### Shared Links (Interactor ↔ Interactor)
- `type: "shared"` or `_is_shared_link: true`
- Discovered when two interactors of the query protein also interact with each other
- Rendered as dashed lines in graph

---

## Key Components & Functions

### 🔹 **App.py** (Flask Server)
| Route | Purpose | Key Function |
|-------|---------|--------------|
| `POST /api/query` | Start research pipeline | `run_full_job()` (runner.py) |
| `GET /api/search/<protein>` | Check if protein exists in DB | Quick lookup, no research |
| `GET /api/status/<protein>` | Poll job status | Returns progress updates |
| `GET /api/results/<protein>` | Fetch full JSON | `build_full_json_from_db()` |
| `GET /api/visualize/<protein>` | Render visualization | `create_visualization_from_dict()` |
| `POST /api/expand/pruned` | Expand node (pruned subgraph) | `run_prune_job()` (utils/pruner.py) |
| `POST /api/chat` | Chat interface | `_build_chat_system_prompt()` |
| `POST /api/cancel/<protein>` | Cancel running job | Sets `cancel_event.set()` |

**Critical Functions**:
- `build_full_json_from_db(protein_symbol)`: Reconstructs complete JSON from PostgreSQL with canonical ordering conversion (app.py:353)
- `build_expansion_json_from_db(protein, visible_proteins)`: Builds expansion with auto-cross-linking (app.py:701)

### 🔹 **Runner.py** (Pipeline Engine)
| Function | Purpose | Location |
|----------|---------|----------|
| `run_full_job()` | Master orchestrator for background threads | Line 1807 |
| `run_pipeline()` | Execute LLM pipeline steps | Line 1055 |
| `run_requery_job()` | Re-query existing protein for new data | Line 2393 |
| `call_gemini_model()` | LLM API wrapper | Line 571 |
| `parse_json_output()` | Parse and merge LLM responses | Line 657 |
| `deep_merge_interactors()` | Intelligently merge new interactors into existing | Line 177 |
| `aggregate_function_arrows()` | Compute interaction-level arrow from functions | Line 382 |

**Pipeline Flow** (run_full_job):
1. Calculate total steps (for progress bar)
2. Get known interactions from DB (for exclusion context)
3. **Phase 1**: Run interactor/function discovery pipeline (3-8 rounds each)
4. **Phase 2**: Validate evidence (optional, can skip)
5. **Phase 3**: Generate metadata (optional)
6. **Phase 4**: Update PMIDs (optional)
7. **Phase 5**: Deduplicate functions (optional, can skip)
8. **Phase 6**: Clean function names (optional)
9. **Phase 7**: Fact-check claims (optional, can skip)
10. **Phase 8**: Validate schema consistency
11. **Phase 9**: Sync to PostgreSQL + file cache

### 🔹 **Utils/db_sync.py** (Database Layer)
| Function | Purpose | Location |
|----------|---------|----------|
| `sync_query_results()` | Write pipeline output to PostgreSQL | Line 75 |
| `sync_chain_relationships()` | Store indirect interaction chains | Line 333 |
| `_get_or_create_protein()` | Upsert protein entity | Line 203 |
| `_save_interaction()` | Upsert interaction with canonical ordering | Line 237 |
| `_validate_and_fix_chain()` | Detect and fix false chain assignments | Line 36 |

**Key Design Choice**: Uses SQLAlchemy transactions (`db.session.begin_nested()`) for atomic updates. Enforces canonical ordering (`protein_a_id < protein_b_id`) to prevent duplicates.

### 🔹 **Visualizer.js** (Frontend Graph)
| Function | Purpose | Location |
|----------|---------|----------|
| `VizJobTracker` (class) | Multi-job orchestration for viz page | Lines 3316-3763 |
| `initNetwork()` | Initialize D3 force simulation | Line 34 |
| `buildInitialGraph()` | Parse JSON, create nodes/links | Line 190 |
| `createSimulation()` | Configure D3 forces | Line ~800 |
| `expandNode()` | Expand interactor (fetch subgraph) | Line ~1200 |
| `collapseNode()` | Collapse expanded subgraph | Line ~1400 |
| `showInteractionModal()` | Display evidence modal | Line ~2000 |
| `showFunctionModal()` | Display function details | Line ~2200 |
| `fetchWithRetry()` | Resilient fetch with exponential backoff | Lines 3091-3105 |
| `saveToSessionStorage()` | Persist jobs across navigation | Lines 3595-3623 |
| `restoreFromSessionStorage()` | Restore jobs on page load | Lines 3684-3762 |

**Key Design Choice**: Frontend handles both NEW format (`proteins[]`, `interactions[]`) and LEGACY format (`interactors[]`) with automatic transformation (visualizer.js:208-283). Job tracking uses functional core + imperative shell pattern with sessionStorage persistence for cross-navigation reliability.

### 🔹 **Utils/pruner.py** (Subgraph Selection)
| Function | Purpose | Location |
|----------|---------|----------|
| `run_prune_job()` | Execute pruning (LLM or heuristic) | Line ~400 |
| `build_candidate_pack()` | Extract metadata for ranking | Line 82 |
| `heuristic_rank_candidates()` | Score candidates without LLM | Line ~300 |
| `is_pruned_fresh()` | Check if cached prune is valid | Line ~600 |

**Key Design Choice**: Heuristic scoring uses confidence, PMID count, recency, and mechanism overlap. LLM mode uses Gemini to rank by biological relevance (optional, slower).

---

## Critical Workflows

### 📊 **Workflow 1: Research Pipeline** (Most Important)
```
User → POST /api/query {"protein": "ATXN3"}
  ↓
app.py: start_query()
  └─ Threading.Thread(run_full_job) → Background execution
     ↓
runner.py: run_full_job()
  ├─ Generate pipeline config (3-8 rounds each)
  ├─ Get known interactions from DB (for exclusion)
  └─ run_pipeline()
     ├─ Phase 1: Interactor Discovery (7 steps)
     │   └─ Gemini + Google Search → Find protein names + classify direct/indirect
     ├─ Phase 2: Function Discovery (6 steps)
     │   └─ Gemini → Find mechanisms, evidence, PMIDs, biological cascades
     └─ Return ctx_json + snapshot_json
  ↓
Validation & Enrichment Pipeline:
  ├─ evidence_validator.py: Add quotes, validate PMIDs
  ├─ claim_fact_checker.py: Verify claims (optional)
  ├─ schema_validator.py: Fix structural issues
  ├─ deduplicate_functions.py: Remove duplicates
  └─ interaction_metadata_generator.py: Generate metadata
  ↓
db_sync.py: sync_query_results()
  ├─ Upsert Protein entity
  ├─ For each interactor:
  │   ├─ Upsert partner Protein
  │   ├─ Enforce canonical ordering (protein_a_id < protein_b_id)
  │   ├─ Store FULL JSONB payload in Interaction.data
  │   └─ Update denormalized fields (confidence, direction, arrow)
  └─ Commit transaction
  ↓
Cache to File (Fallback/Intermediate):
  └─ Write cache/<PROTEIN>.json
  ↓
Update Job Status:
  └─ jobs[protein]['status'] = 'complete'
```

**Key Files**:
- `app.py`: Routes + job orchestration
- `runner.py`: Pipeline engine + LLM calls
- `utils/db_sync.py`: PostgreSQL sync
- `utils/evidence_validator.py`: Evidence enrichment
- `pipeline/config_gemini_MAXIMIZED.py`: Pipeline configuration

### 📊 **Workflow 2: Fetching & Sending Data to Frontend**
```
User → GET /api/visualize/ATXN3
  ↓
app.py: get_visualization()
  └─ build_full_json_from_db("ATXN3")
     ├─ Query Protein table for main protein
     ├─ Query Interactions table (bidirectional due to canonical ordering)
     │   └─ WHERE protein_a_id = ATXN3.id OR protein_b_id = ATXN3.id
     ├─ For each interaction:
     │   ├─ Extract FULL JSONB payload from data column
     │   ├─ Convert canonical direction → query-relative direction
     │   │   └─ 'a_to_b' → 'main_to_primary' OR 'primary_to_main' (based on query perspective)
     │   └─ Build interaction dict (source, target, type, arrow, functions[], evidence[])
     ├─ Query shared links (interactions BETWEEN interactors)
     │   └─ WHERE protein_a_id IN (interactor_ids) AND protein_b_id IN (interactor_ids)
     └─ Return {snapshot_json: {main, proteins[], interactions[]}, ctx_json: {...}}
  ↓
visualizer.py: create_visualization_from_dict()
  └─ Embed JSON into HTML template
  └─ Return HTML with inline <script> containing SNAP = {...}
  ↓
Browser loads visualization page
  ↓
visualizer.js: initNetwork()
  ├─ Parse SNAP.proteins[] and SNAP.interactions[]
  ├─ Handle LEGACY format fallback (SNAP.interactors[] → transform to new)
  ├─ Build D3 nodes[] and links[]
  │   └─ For each interaction:
  │       ├─ Determine source/target based on direction
  │       ├─ Attach functions[], evidence[], pmids[]
  │       └─ Set visual properties (arrow markers, colors, dash patterns)
  └─ createSimulation()
     ├─ D3 force simulation (charge, collision, links)
     └─ Render SVG (nodes, links, labels)
  ↓
User clicks interaction arrow or node
  ↓
showInteractionModal() or showFunctionModal()
  └─ Display evidence, functions, biological cascades in modal
```

**Key Files**:
- `app.py`: build_full_json_from_db() (line 353)
- `models.py`: Protein, Interaction models
- `visualizer.py`: HTML template generator
- `static/visualizer.js`: D3 graph rendering (5736 LOC)

### 📊 **Workflow 3: Frontend Data Parsing & Rendering**
```
visualizer.js: buildInitialGraph() (line 190)
  ↓
Step 1: Detect format (NEW vs LEGACY)
  ├─ NEW: SNAP.proteins[] + SNAP.interactions[] exists
  └─ LEGACY: SNAP.interactors[] exists
  ↓
Step 2a: NEW format (direct use)
  ├─ proteins = SNAP.proteins
  └─ interactions = SNAP.interactions
  ↓
Step 2b: LEGACY format (transform)
  ├─ Extract proteins from SNAP.interactors[]
  └─ Transform each interactor → interaction:
     ├─ Convert query-relative direction → link-absolute
     │   └─ 'primary_to_main' → source=primary, target=main
     │   └─ 'main_to_primary' → source=main, target=primary
     ├─ Override source for indirect interactions (use upstream_interactor)
     └─ Attach all metadata (functions, evidence, pmids)
  ↓
Step 3: Build D3 graph data
  ├─ Create nodes[] (main + interactors)
  │   └─ Main node: larger radius, indigo gradient
  │   └─ Interactor nodes: smaller radius, gray gradient
  ├─ Create links[] from interactions
  │   ├─ Determine arrow type (activates, inhibits, binds, regulates)
  │   ├─ Set marker (arrow-activate, arrow-inhibit, arrow-binding, arrow-regulate)
  │   ├─ Set stroke pattern:
  │   │   ├─ Solid: direct interactions
  │   │   ├─ Dashed: indirect interactions (cascade chains)
  │   │   └─ Dotted: shared links (interactor ↔ interactor)
  │   └─ Attach interaction data (for modal display)
  └─ Calculate dynamic spacing (scales with interactor count)
  ↓
Step 4: Create D3 force simulation
  ├─ forceCenter: Pin main node at center
  ├─ forceManyBody: Repulsion between nodes (charge: -800)
  ├─ forceCollide: Prevent node overlap (radius + padding)
  └─ forceLink: Spring force between connected nodes
  ↓
Step 5: Render SVG elements
  ├─ Links (paths with markers)
  ├─ Nodes (circles with labels)
  └─ Function boxes (rounded rects attached to interactors)
  ↓
Step 6: Add interactivity
  ├─ Click node → expandNode() or collapseNode()
  ├─ Click link → showInteractionModal()
  ├─ Click function box → showFunctionModal()
  └─ Zoom/pan with d3.zoom()
```

**Key Files**:
- `static/visualizer.js`: All graph logic (6800+ LOC)
- `static/viz-styles.css`: Graph styles

### 📊 **Workflow 4: Multi-Job Tracking** (New System)
```
USER ACTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Index Page (Full Cards):
  User starts query for PROTEIN1
    ↓
  JobTracker.addJob("PROTEIN1", config)
    ├─ Create job state (processing, 0%, startTime)
    ├─ Render full job card (70-80px height)
    ├─ Start independent polling (5s interval)
    └─ Auto-navigate to viz page on completion
    ↓
  User starts query for PROTEIN2 (while PROTEIN1 running)
    ↓
  JobTracker.addJob("PROTEIN2", config)
    ├─ Check duplicate (shows confirm dialog if already exists)
    ├─ Add second job card to tracker
    └─ Both jobs poll independently
    ↓
  User clicks "−" button (remove from tracker)
    ↓
  JobTracker.removeFromTracker()
    └─ Job continues in background (backend keeps running)

Viz Page (Compact Chips):
  Page loads with SNAP.main = "PROTEIN1"
    ↓
  Auto-resume check:
    └─ fetch(/api/status/PROTEIN1) → if processing, add to VizJobTracker
    ↓
  SessionStorage restore:
    ├─ Read vizActiveJobs from sessionStorage
    ├─ For each saved job:
    │   ├─ Skip if stale (>1 hour old)
    │   ├─ Skip if already tracked (from auto-resume)
    │   ├─ fetch(/api/status/{protein}) with retry
    │   └─ If still processing → add to tracker
    └─ Clean sessionStorage (keep only active jobs)
    ↓
  User searches for protein not in DB with running job:
    ↓
  handleQuery():
    ├─ fetch(/api/search/{protein}) → not_found
    ├─ fetch(/api/status/{protein}) → processing
    └─ VizJobTracker.addJob() (stay on page, show notification)
    ↓
  User navigates to different protein:
    ↓
  VizJobTracker.saveToSessionStorage()
    ├─ Read existing saved jobs (merge, don't overwrite)
    ├─ Filter current processing jobs
    └─ Save merged list to sessionStorage
    ↓
  Page unload:
    └─ beforeunload event → clear all polling intervals


POLLING LIFECYCLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every 5 seconds (per job):
  fetchWithRetry(/api/status/{protein})
    ├─ Timeout: 30s
    ├─ Retries: 3 (exponential backoff 1s, 2s, 4s)
    └─ On success:
       ├─ Update progress (current/total)
       ├─ Update UI (progress bar, percentage)
       └─ Handle status:
          ├─ complete → stop polling, navigate/reload
          ├─ error → show error, auto-remove after 5s
          └─ cancelled → show cancelled, auto-remove after 2s


CANCEL OPERATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User clicks "✕" button:
  JobTracker.cancelJob(protein)
    ├─ Stop polling FIRST (prevent race condition)
    ├─ Disable cancel button
    ├─ POST /api/cancel/{protein}
    ├─ On success:
    │   ├─ Mark job as cancelled
    │   ├─ Update UI
    │   └─ Remove after 2s delay
    └─ On error:
       ├─ Re-enable cancel button
       ├─ Show error in UI
       └─ Restart polling (job still running)


EDGE CASES HANDLED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Multi-tab navigation → Jobs merge in sessionStorage
✅ Duplicate job detection → Confirm dialog with cancel option
✅ Stale job cleanup → Auto-remove >1 hour old from storage
✅ Network failures → 3 retries with exponential backoff
✅ Request timeouts → 30s limit prevents hanging
✅ Race conditions → Polling stops before cancel request
✅ Page unload → All intervals cleared (no memory leaks)
✅ Parallel restores → Guard flag prevents concurrent execution
✅ Auto-resume conflicts → Skip jobs already tracked
```

**Key Files**:
- `static/script.js`: JobTracker class (lines 285-590)
- `static/visualizer.js`: VizJobTracker class (lines 3316-3763)
- `EDGE_CASE_FIXES.md`: Complete documentation of 9 fixes

**Architecture Pattern**:
- **Functional Core**: Pure state transformers (createJobState, updateJobProgress, etc.)
- **Imperative Shell**: DOM manipulation (createJobCard, updateJobCard, etc.)
- **Composition**: JobTracker orchestrates core + shell
- **Persistence**: SessionStorage (viz page only) with smart merge logic

---

## Development Setup

### Local Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file
GOOGLE_API_KEY=your_key_here
DATABASE_PUBLIC_URL=postgresql://...  # Railway external URL for local dev
DATABASE_URL=postgresql://...         # (Railway sets this automatically in prod)

# 3. Run Flask server
python app.py  # Starts on http://127.0.0.1:5000

# 4. Database tables auto-created on first run
# (db.create_all() in app.py startup)
```

### Production (Railway)
```bash
# Railway auto-sets DATABASE_URL (internal network)
# Push to main branch → Railway auto-deploys
git push origin main
```

### File Cache vs Database
- **PostgreSQL**: Primary storage (canonical source of truth)
- **File cache (`cache/`)**: Used for:
  - Intermediate storage during pipeline execution
  - Fallback when database fails
  - Pruned expansions (`cache/pruned/`)
  - Local backups/snapshots

**Rule**: All queries write to BOTH PostgreSQL and file cache. Reads prioritize database, fall back to file.

---

## Where to Look: Quick Reference

### "Where is X defined/handled?"

| What | File | Function/Line |
|------|------|---------------|
| **Flask API routes** | app.py | Lines 114-1877 |
| **Research pipeline** | runner.py | run_full_job (1807), run_pipeline (1055) |
| **Database models** | models.py | Protein (18), Interaction (69) |
| **PostgreSQL sync** | utils/db_sync.py | sync_query_results (75) |
| **LLM calls** | runner.py | call_gemini_model (571) |
| **Evidence validation** | utils/evidence_validator.py | validate_and_enrich_evidence (~200) |
| **Fact checking** | utils/claim_fact_checker.py | fact_check_json (~100) |
| **Function deduplication** | utils/deduplicate_functions.py | deduplicate_payload (~50) |
| **Pruning/expansion** | utils/pruner.py | run_prune_job (~400) |
| **Frontend graph rendering** | static/visualizer.js | buildInitialGraph (190), createSimulation (~800) |
| **Graph interactions** | static/visualizer.js | expandNode (~1200), showInteractionModal (~2000) |
| **Table view** | static/visualizer.js | buildTableView (~3500) |
| **Chat interface** | app.py | POST /api/chat (1382) |
| **Job status tracking (backend)** | app.py | jobs dict + jobs_lock |
| **Multi-job tracking (index)** | static/script.js | JobTracker class (285-590) |
| **Multi-job tracking (viz)** | static/visualizer.js | VizJobTracker class (3316-3763) |
| **Job persistence** | static/visualizer.js | saveToSessionStorage (3595), restoreFromSessionStorage (3684) |
| **Fetch with retry** | static/script.js, visualizer.js | fetchWithRetry (lines 33-47, 3091-3105) |

### "I need to debug X"

| Issue | Files to Check | What to Look For |
|-------|----------------|------------------|
| **Pipeline fails** | runner.py, Logs/<PROTEIN>/ | LLM errors, JSON parse errors, cancellation |
| **Database errors** | app.py, utils/db_sync.py | Transaction rollbacks, unique constraint violations |
| **Graph not rendering** | static/visualizer.js (190-400) | Console errors, SNAP data format, empty proteins/interactions |
| **Shared links missing** | app.py:build_full_json_from_db (600-677) | Shared link query logic |
| **Indirect chains wrong** | utils/db_sync.py:_validate_and_fix_chain (36-73) | False chain detection |
| **Modal data incorrect** | static/visualizer.js (~2000-2500) | Modal rendering, evidence parsing |
| **Table view bugs** | static/visualizer.js (~3500-4500) | Table building, filtering, sorting |
| **Chat not working** | app.py:_build_chat_system_prompt (1382-1500) | Context building, LLM prompts |
| **Performance issues** | static/visualizer.js | Console.log statements (REMOVE), shared link rendering |
| **Jobs disappearing** | static/visualizer.js (3595-3762) | SessionStorage save/restore, merge logic |
| **Job polling stuck** | static/script.js, visualizer.js (_startPolling) | Retry logic, timeout, interval clearing |
| **Multi-tab job loss** | static/visualizer.js (3595-3623) | SessionStorage merge (not overwrite) |

### "I need to add a new feature"

| Feature | Primary Files | Secondary Files |
|---------|---------------|-----------------|
| **New pipeline step** | pipeline/config_gemini_MAXIMIZED.py | runner.py (add step handler) |
| **New API endpoint** | app.py | Add route, update static/script.js |
| **New validation** | utils/ (create new file) | runner.py (call in pipeline) |
| **New graph feature** | static/visualizer.js | static/viz-styles.css |
| **New modal** | static/visualizer.js | Add modal HTML + event handlers |
| **New DB column** | models.py | utils/db_sync.py (update sync logic) |

### "I need to understand X"

| Concept | Read These Files | Key Sections |
|---------|------------------|--------------|
| **JSON schemas** | CLAUDE.md (lines 52-166) | snapshot_json, ctx_json, interactor format |
| **Canonical ordering** | models.py (130-138), utils/db_sync.py (237-332) | Unique constraint, direction conversion |
| **Direct vs indirect** | CLAUDE.md (lines 87-92) | interaction_type, upstream_interactor, depth |
| **Arrow determination** | runner.py (382-550) | aggregate_function_arrows logic |
| **Pruning algorithm** | utils/pruner.py (82-200, ~300-400) | Candidate scoring, LLM ranking |
| **Evidence structure** | cache/LC3B.json (29-80) | Example evidence objects |

---

## Known Issues & Areas for Improvement

### 🐛 Bugs (High Priority)
1. **Performance**: Heavy console.log statements in shared link rendering causing slowdowns
2. **UI Polish**: Some buttons not working, UI tweaks needed
3. **Table/Chat views**: Need verification that all graph data is displayed correctly

### 🧹 Code Cleanup (After Bug Fixes)
1. **Remove migration files**: 12 `migrate_*.py` files in root (no longer needed)
2. **Remove legacy backups**: `visualizer copy.py`, `propaths_refactor_*.txt/md`
3. **Extract frontend code**: `visualizer.py` should only be template generator, move JS to separate file
4. **Deduplicate functions**: Several repeated utility functions across backend files
5. **Better documentation**: Some complex functions lack docstrings

### 🔮 Future Enhancements
1. **Automated tests**: Currently only 2 unused tests in `tests/`
2. **Code organization**: Root directory has too many files
3. **Legacy cleanup**: Fully deprecate `cache/proteins/` file-based storage (keep `protein_database.py` helpers only)

---

## Additional Resources

- **CLAUDE.md**: Detailed project documentation (data contracts, API reference, development workflow)
- **ARYAN.md**: Co-founder's documentation
- **DEPLOYMENT.md**: Railway deployment guide
- **Logs/**: Pipeline execution logs (useful for debugging LLM behavior)
- **Railway Dashboard**: https://railway.app/ (production database, deployment logs)

---

## Quick Commands

```bash
# Start local server
python app.py

# Check database status
# (Look at startup logs for protein/interaction counts)

# Run migration (DEPRECATED - don't use unless specifically needed)
# python migrate_*.py

# View logs for a protein
ls -la Logs/<PROTEIN>/

# Clear cache (nuclear option - forces re-query)
rm -rf cache/<PROTEIN>.json
```

---

**Last Updated**: 2025-01-17
**Recent Changes**: Added multi-job tracking system with sessionStorage persistence, edge case fixes, and resilient fetch utilities
**Maintainers**: Kazi
**Production**: Railway (PostgreSQL + Flask deployment)
**LLM**: Google Gemini 2.5 Pro with Google Search grounding

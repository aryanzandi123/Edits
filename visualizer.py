"""
Visualizer with original styling & behavior restored, plus:
- De-densified layout (spacing, charge, collision)
- Header/search matches index styles (title centered, round search bar)
- Nodes: dark circles + WHITE labels (as before)
- Legend restored
- Modals match original styling; two distinct modal paths:
  (1) Interaction (main ↔ interactor) when clicking the interactor link/ circle
  (2) Function (interactor → function) when clicking the function link/box
- Function confidence labels on boxes (as before)
- Arrows: pointer on hover + thicker on hover
- Function boxes connect ONLY to their interactor (never to main)
- Progress bar on viz page updated using your exact IDs
- Snapshot hydrated with ctx_json for complete function/evidence details
- Expand-on-click preserved; depth limit = 3
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import tempfile

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>ProPath - PLACEHOLDER_MAIN</title>
  <link rel="icon" type="image/png" href="/static/logo.png">
  <link rel="apple-touch-icon" href="/static/logo.png">
  <link rel="stylesheet" href="/static/styles.css"/>
  <script src="/static/script.js"></script>
  <script src="https://cdn.sheetjs.com/xlsx-0.20.1/package/dist/xlsx.full.min.js"></script>
  <link rel="stylesheet" href="/static/viz-styles.css"/>
</head>
<body class="dark-mode">
<div class="container">
  <!-- Invisible hover trigger -->
  <div class="header-trigger"></div>

  <!-- Header (classes/IDs aligned with index page) -->
  <div class="header">
    <h1 class="title" id="networkTitle">PLACEHOLDER_MAIN Interaction Network</h1>

    <div class="header-search-container">
      <div class="input-container">
        <input type="text" id="protein-input" placeholder="Search another protein..."/>
        <button id="query-button">Generate</button>
      </div>
    </div>

    <!-- Inline Controls Row: View Tabs + Research Settings -->
    <div class="header-controls-row">
      <div class="view-tabs">
        <button class="header-btn tab-btn active" onclick="switchView('graph')">Graph View</button>
        <button class="header-btn tab-btn" onclick="switchView('table')">Table View</button>
        <button class="header-btn tab-btn" onclick="switchView('chat')">Chat</button>
      </div>

      <div class="interaction-mode-tabs" style="display: flex; gap: 4px; margin-left: 16px;">
        <button class="header-btn mode-btn active" onclick="switchInteractionMode('direct')" id="mode-direct" title="Show direct protein-protein interactions (default)">Direct</button>
        <button class="header-btn mode-btn" onclick="switchInteractionMode('net')" id="mode-net" title="Show net effects through mediator chains">Net Effects</button>
        <button class="header-btn mode-btn" onclick="switchInteractionMode('both')" id="mode-both" title="Show all interactions (may overlap)">Both</button>
        <span id="view-mode-counter"></span>
      </div>

      <details class="config-details-inline">
        <summary class="header-btn config-summary-inline">Research Settings</summary>
        <div class="config-content-inline">
          <div class="config-presets">
            <button class="preset-btn" onclick="setPreset(3,3)">Quick</button>
            <button class="preset-btn" onclick="setPreset(5,5)">Standard</button>
            <button class="preset-btn" onclick="setPreset(8,8)">Thorough</button>
          </div>
          <div class="config-inputs">
            <label class="config-label">
              <span class="config-label-text">Interactor Discovery Rounds:</span>
              <input type="number" id="interactor-rounds" class="config-input" min="3" max="8" value="3">
            </label>
            <label class="config-label">
              <span class="config-label-text">Function Mapping Rounds:</span>
              <input type="number" id="function-rounds" class="config-input" min="3" max="8" value="3">
            </label>
          </div>
        </div>
      </details>

      <button class="header-btn theme-toggle-btn" onclick="toggleTheme()" title="Toggle Light/Dark Mode" id="theme-toggle">
        <span id="theme-icon">☀️</span>
      </button>
    </div>

    <div id="job-notification" class="job-notification">
      <!-- Multi-job tracker container -->
      <div id="mini-job-container" class="mini-job-container"></div>
      <!-- Notification message for non-job updates -->
      <p id="notification-message" style="display: none;"></p>
    </div>
  </div>

  <div id="network" class="view-container">
    <div class="controls">
      <button class="control-btn" onclick="zoomIn()" title="Zoom In">+</button>
      <button class="control-btn" onclick="zoomOut()" title="Zoom Out">−</button>
      <div class="control-divider"></div>
      <button class="control-btn" onclick="refreshVisualization()" title="Reset Graph">⟳</button>
      <div class="control-divider"></div>
      <div class="filter-label">Show:</div>
      <button class="graph-filter-btn activates active" onclick="toggleGraphFilter('activates')" title="Show/Hide Activation">
        <span class="filter-dot activates"></span> Activates
      </button>
      <button class="graph-filter-btn inhibits active" onclick="toggleGraphFilter('inhibits')" title="Show/Hide Inhibition">
        <span class="filter-dot inhibits"></span> Inhibits
      </button>
      <button class="graph-filter-btn binds active" onclick="toggleGraphFilter('binds')" title="Show/Hide Binding">
        <span class="filter-dot binds"></span> Binds
      </button>
      <button class="graph-filter-btn regulates active" onclick="toggleGraphFilter('regulates')" title="Show/Hide Regulation">
        <span class="filter-dot regulates"></span> Regulates
      </button>
      <div class="control-divider"></div>
      <div class="filter-label">Depth:</div>
      <button class="graph-filter-btn depth-filter active" data-depth="1" onclick="toggleDepthFilter(1)" title="Show Depth 1 (Direct interactors)">1</button>
      <button class="graph-filter-btn depth-filter active" data-depth="2" onclick="toggleDepthFilter(2)" title="Show Depth 2 (Indirect interactors)">2</button>
      <button class="graph-filter-btn depth-filter active" data-depth="3" onclick="toggleDepthFilter(3)" title="Show Depth 3 (Extended network)">3</button>
    </div>

    <div class="info-panel"><strong>TIPS:</strong> Click arrows & nodes for details</div>

    <!-- Legend restored -->
    <div class="legend">
      <div class="legend-title">INTERACTION TYPES</div>
      <div class="legend-item">
        <div class="legend-arrow">
          <svg width="30" height="20"><line x1="0" y1="10" x2="20" y2="10" stroke="#059669" stroke-width="2"/><polygon points="20,10 26,10 23,7 23,13" fill="#059669"/></svg>
        </div>Activates
      </div>
      <div class="legend-item">
        <div class="legend-arrow">
          <svg width="30" height="20"><line x1="0" y1="10" x2="20" y2="10" stroke="#dc2626" stroke-width="2"/><line x1="23" y1="6" x2="23" y2="14" stroke="#dc2626" stroke-width="3"/></svg>
        </div>Inhibits
      </div>
      <div class="legend-item">
        <div class="legend-arrow">
          <svg width="30" height="20"><line x1="0" y1="8" x2="26" y2="8" stroke="#7c3aed" stroke-width="2"/><line x1="0" y1="12" x2="26" y2="12" stroke="#7c3aed" stroke-width="2"/></svg>
        </div>Binding
      </div>
      <div class="legend-item" style="margin-top:12px;padding-top:8px;border-top:1px solid #e5e7eb">
        <div class="legend-arrow">
          <svg width="30" height="20"><line x1="0" y1="10" x2="26" y2="10" stroke="#6b7280" stroke-width="2"/></svg>
        </div>Direct (physical)
      </div>
      <div class="legend-item">
        <div class="legend-arrow">
          <svg width="30" height="20"><line x1="0" y1="10" x2="26" y2="10" stroke="#6b7280" stroke-width="2" stroke-dasharray="8,4"/></svg>
        </div>Indirect (cascade)
      </div>
      <div class="legend-item">
        <div class="legend-arrow">
          <svg width="30" height="20"><line x1="0" y1="10" x2="26" y2="10" stroke="#ff8c00" stroke-width="2" stroke-dasharray="5,5"/></svg>
        </div>Incomplete (mediator missing)
      </div>
    </div>

    <svg id="svg"></svg>
  </div>

  <div id="table-view" class="view-container" style="display:none;">
    <div class="table-controls">
      <div class="table-controls-main">
        <div class="table-controls-left">
          <div class="search-container">
            <input type="text" id="table-search" class="table-search-input" placeholder="Search interactions..." oninput="handleSearchInput(event)">
            <button class="search-clear-btn" id="search-clear-btn" onclick="clearSearch()" style="display:none;">×</button>
          </div>
          <div class="filter-chips">
            <button class="filter-chip filter-active activates" onclick="toggleFilter('activates')">Activates</button>
            <button class="filter-chip filter-active inhibits" onclick="toggleFilter('inhibits')">Inhibits</button>
            <button class="filter-chip filter-active binds" onclick="toggleFilter('binds')">Binds</button>
            <button class="filter-chip filter-active regulates" onclick="toggleFilter('regulates')">Regulates</button>
          </div>
        </div>
        <div class="export-dropdown">
          <button class="export-btn" onclick="toggleExportDropdown()">Export ▼</button>
          <div class="export-dropdown-menu" id="export-dropdown-menu">
            <button class="export-option" onclick="exportToCSV(); closeExportDropdown();">Export as CSV</button>
            <button class="export-option" onclick="exportToExcel(); closeExportDropdown();">Export as Excel (.xlsx)</button>
          </div>
        </div>
      </div>
      <div id="filter-results" class="filter-results"></div>
    </div>
    <div class="table-wrapper">
      <table id="interactions-table" class="data-table">
        <thead>
          <tr>
            <th class="col-expand"><span class="expand-header-icon">▼</span></th>
            <th class="col-interaction resizable sortable" data-sort="interaction" onclick="sortTable('interaction')">Interaction <span class="sort-indicator"></span><span class="resize-handle"></span></th>
            <th class="col-effect resizable sortable" data-sort="effect" onclick="sortTable('effect')">Type <span class="sort-indicator"></span><span class="resize-handle"></span></th>
            <th class="col-function resizable sortable" data-sort="function" onclick="sortTable('function')">Function Affected <span class="sort-indicator"></span><span class="resize-handle"></span></th>
            <th class="col-effect-type resizable sortable" data-sort="effectType" onclick="sortTable('effectType')">Effect <span class="sort-indicator"></span><span class="resize-handle"></span></th>
            <th class="col-mechanism resizable sortable" data-sort="mechanism" onclick="sortTable('mechanism')">Mechanism <span class="sort-indicator"></span><span class="resize-handle"></span></th>
          </tr>
        </thead>
        <tbody id="table-body">
          <!-- Populated by buildTableView() -->
        </tbody>
      </table>
    </div>
  </div>

  <div id="chat-view" class="view-container" style="display:none;">
    <div class="chat-container">
      <div class="chat-header">
        <h2 class="chat-title">Network Assistant</h2>
        <p class="chat-subtitle">Ask questions about the protein interaction network</p>
      </div>
      <div id="chat-messages" class="chat-messages">
        <div class="chat-message system-message">
          <div class="message-content">
            👋 Hello! I'm here to help you understand this protein interaction network. Ask me anything about the visible proteins, their interactions, or biological functions.
          </div>
        </div>
      </div>
      <div class="chat-input-wrapper">
        <textarea
          id="chat-input"
          class="chat-input"
          placeholder="Ask about this network (e.g., 'What proteins interact with ATXN3?')..."
          rows="3"
        ></textarea>
        <button id="chat-send-btn" class="chat-send-btn" onclick="sendChatMessage()">
          <span id="chat-send-text">Send</span>
          <span id="chat-send-loading" style="display:none;">Thinking...</span>
        </button>
      </div>
    </div>
  </div>
</div>

<!-- Modal restored -->
<div id="modal" class="modal">
  <div class="modal-content">
    <div class="modal-header">
      <h2 class="modal-title" id="modalTitle">Details</h2>
      <button class="close-btn" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div id="modalBody"></div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
/* ===== Robust data load & hydration ===== */
let RAW, SNAP, CTX;

try {
  // Validate PLACEHOLDER was replaced
  RAW = PLACEHOLDER_JSON;

  if (!RAW || typeof RAW !== 'object') {
    throw new Error('Invalid data structure: RAW is not an object');
  }

  // Check if PLACEHOLDER wasn't replaced (safety check)
  const rawStr = JSON.stringify(RAW).substring(0, 100);
  if (rawStr.includes('PLACEHOLDER')) {
    throw new Error('Data embedding failed: template placeholder not replaced');
  }

  console.log('✅ Step 1: RAW data loaded', {
    keys: Object.keys(RAW),
    hasSnapshot: !!RAW.snapshot_json,
    hasCtx: !!RAW.ctx_json
  });

  SNAP = (RAW && RAW.snapshot_json && typeof RAW.snapshot_json === 'object') ? RAW.snapshot_json : (RAW || {});
  CTX  = (RAW && RAW.ctx_json && typeof RAW.ctx_json === 'object') ? RAW.ctx_json : {};
  SNAP.interactors = Array.isArray(SNAP.interactors) ? SNAP.interactors : [];
  if (!SNAP.main) SNAP.main = RAW.main || RAW.primary || 'Unknown';

  console.log('✅ Step 2: SNAP extracted', {
    main: SNAP.main,
    keys: Object.keys(SNAP),
    hasProteins: !!SNAP.proteins,
    hasInteractors: !!SNAP.interactors,
    hasInteractions: !!SNAP.interactions
  });

} catch (error) {
  console.error('❌ Data initialization failed:', error);
  document.getElementById('network').innerHTML =
    `<div style="padding: 60px 40px; text-align: center; color: #ef4444; font-family: system-ui, sans-serif;">
      <h2 style="font-size: 24px; margin-bottom: 16px;">⚠️ Failed to Load Visualization</h2>
      <p style="font-size: 16px; color: #6b7280; margin-bottom: 8px;">Error: ${error.message}</p>
      <p style="font-size: 14px; color: #9ca3af;">Check the browser console for details, then try refreshing the page.</p>
    </div>`;
  throw error; // Stop execution
}

console.log('✅ Step 3: Data loaded (hydration removed - using direct database format)');

document.getElementById('networkTitle').textContent = `${SNAP.main} Interaction Network`;
</script>
<script src="/static/visualizer.js?v=785"></script>
</body>
</html>
"""

def _load_json(obj):
    if isinstance(obj, (str, bytes, Path)):
        return json.loads(Path(obj).read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        return obj
    raise TypeError("json_data must be path or dict")

# JSON helper functions for data cleaning and validation
def _resolve_symbol(entry):
    """Resolves protein symbol from various field names"""
    for key in ('primary', 'hgnc_symbol', 'symbol', 'gene', 'name'):
        value = entry.get(key) if isinstance(entry, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    placeholder = None
    if isinstance(entry, dict):
        placeholder = entry.get('id') or entry.get('interactor_id') or entry.get('mechanism_id')
    if placeholder:
        return f"MISSING_{placeholder}"
    return None

def _build_interactor_key(interactor):
    """Creates unique key for interactor matching"""
    if not isinstance(interactor, dict):
        return None
    pmids = interactor.get('pmids')
    if isinstance(pmids, list) and pmids:
        normalized_pmids = tuple(sorted(str(pmid) for pmid in pmids))
        return ('pmids', normalized_pmids)
    summary = interactor.get('support_summary')
    if isinstance(summary, str) and summary.strip():
        return ('summary', summary.strip())
    mechanism = interactor.get('mechanism_details')
    if isinstance(mechanism, list) and mechanism:
        return ('mechanism', tuple(sorted(mechanism)))
    return None

# Function name shortening map - REMOVED to preserve AI-generated specificity
# Previous NAME_FIXES was making specific names vague:
#   "ATXN3 Degradation" → "Degradation" (loses what's being degraded!)
#   "RNF8 Stability & DNA Repair" → "DNA repair" (loses the protein!)
#   "Apoptosis Inhibition" → "Apoptosis" (loses the arrow direction!)
# The AI prompts now generate specific, arrow-compatible names - preserve them!
NAME_FIXES = {}

def validate_function_name(name: str) -> tuple[bool, str]:
    """
    Check if function name is specific enough.
    Returns (is_valid, error_message)
    """
    if not name or not isinstance(name, str):
        return (False, "Function name is missing or invalid")

    name_lower = name.lower().strip()

    # Too short
    if len(name) < 5:
        return (False, f"Function name '{name}' is too short (< 5 chars)")

    # Check for overly generic terms without specifics
    generic_patterns = [
        ('regulation', 30),   # "Regulation" is vague unless part of longer specific name
        ('control', 25),      # "Control" is vague
        ('response', 25),     # "Response" is vague (unless specific like "DNA Damage Response")
        ('metabolism', 20),   # "Metabolism" alone is too vague
        ('signaling', 20),    # "Signaling" alone is too vague
        ('pathway', 20),      # "Pathway" alone is too vague
    ]

    for term, min_length in generic_patterns:
        if term in name_lower and len(name) < min_length:
            return (False, f"Function name '{name}' is too generic (contains '{term}' but too short)")

    # Check for very generic standalone terms
    very_generic = [
        'function', 'process', 'activity', 'mechanism', 'role',
        'involvement', 'participation', 'interaction'
    ]
    if name_lower in very_generic:
        return (False, f"Function name '{name}' is extremely generic")

    return (True, "")


def validate_interactor_quality(interactor: dict) -> list[str]:
    """
    Check for data quality issues in an interactor.
    Returns list of warning messages.
    """
    issues = []
    primary = interactor.get('primary', 'Unknown')

    # Check interactor-level confidence
    interactor_conf = interactor.get('confidence')
    if interactor_conf is not None and interactor_conf == 0:
        issues.append(f"{primary}: interaction confidence is 0 (likely data error)")

    # Check functions
    for idx, func in enumerate(interactor.get('functions', [])):
        func_name = func.get('function', f'Function #{idx}')

        # Validate function name specificity
        is_valid, msg = validate_function_name(func_name)
        if not is_valid:
            issues.append(f"{primary}/{func_name}: {msg}")

        # Validate function confidence
        fn_conf = func.get('confidence')
        if fn_conf is not None and fn_conf == 0:
            issues.append(f"{primary}/{func_name}: function confidence is 0 (likely data error)")

        # Check if arrow and function name are compatible
        arrow = func.get('arrow', '')
        if arrow in ['activates', 'inhibits']:
            # Function name should describe a process that can be activated/inhibited
            # This is a heuristic check
            incompatible_terms = ['interaction', 'binding', 'association']
            if any(term in func_name.lower() for term in incompatible_terms):
                issues.append(f"{primary}/{func_name}: arrow='{arrow}' may not match function name")

    return issues


def create_visualization(json_data, output_path=None):
    # PMID refresh disabled: PMIDs are already updated during pipeline execution (runner.py STAGE 5)
    # This eliminates 10-40 second blocking delays on visualization requests
    data = _load_json(json_data)

    # NEW FORMAT: Use proteins + interactions arrays directly from database
    # No normalization or deduplication needed - database returns clean data
    if 'snapshot_json' in data:
        viz_data = data['snapshot_json']
    elif 'main' in data:
        # Direct snapshot format (rare, but possible)
        viz_data = data
    else:
        raise ValueError("Invalid JSON structure: expected 'snapshot_json' or 'main' field")

    # Validate new format structure
    if not isinstance(viz_data.get('proteins'), list) or not isinstance(viz_data.get('interactions'), list):
        raise ValueError("Invalid data structure: expected 'proteins' (list) and 'interactions' (list)")

    # Get main protein name (with fallback logic)
    main = viz_data.get('main', 'Unknown')
    if not main or main == 'UNKNOWN':
        main = 'Unknown'

    # Validate data quality and log warnings
    all_issues = []
    for interaction in viz_data.get('interactions', []):
        issues = validate_interactor_quality(interaction)
        all_issues.extend(issues)

    if all_issues:
        print(f"\n⚠️  Data Quality Warnings for {main}:")
        for issue in all_issues[:10]:  # Limit to first 10 to avoid spam
            print(f"  - {issue}")
        if len(all_issues) > 10:
            print(f"  ... and {len(all_issues) - 10} more warnings")
        print()

    # Prepare final data for embedding
    raw = data  # Keep original structure for backwards compatibility

    # Title uses snapshot_json.main or fallback
    try:
        main = (raw.get('snapshot_json') or {}).get('main') or raw.get('main') or raw.get('primary') or 'Protein'
    except Exception:
        main = raw.get('main') or raw.get('primary') or 'Protein'

    html = HTML.replace('PLACEHOLDER_MAIN', str(main))
    html = html.replace('PLACEHOLDER_JSON', json.dumps(raw, ensure_ascii=False))

    if output_path:
        # If output_path provided, write to file and return path
        p = Path(output_path)
        p.write_text(html, encoding='utf-8')
        return str(p.resolve())
    else:
        # If no output_path, return HTML content directly (for web endpoints)
        return html

def create_visualization_from_dict(data_dict, output_path=None):
    """
    Create visualization from dict (not file).

    NEW: Accepts dict directly from database (PostgreSQL).
    This maintains compatibility with existing frontend while enabling
    database-backed visualization.

    Args:
        data_dict: Dict with {snapshot_json: {...}, ctx_json: {...}}
        output_path: Optional output file path. If None, returns HTML content.

    Returns:
        HTML string if output_path is None, else path to saved HTML file

    Note:
        Internally calls create_visualization() which supports both
        dict input (via _load_json) and returns HTML or file path based on output_path.
    """
    if not isinstance(data_dict, dict):
        raise TypeError("data_dict must be a dict")

    # create_visualization already supports dict input via _load_json
    return create_visualization(data_dict, output_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <json_file> [output_html]"); raise SystemExit(2)
    src = sys.argv[1]; dst = sys.argv[2] if len(sys.argv)>2 else None
    out = create_visualization(src, dst); print("Wrote:", out)

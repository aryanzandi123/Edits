
/* ===== Globals ===== */
let svg, g, width, height, simulation, zoomBehavior;
let graphInitialFitDone = false;
let fitToViewTimer = null;

let nodes = [], links = [];

// PERFORMANCE: Throttle link path updates to screen refresh rate (max 60fps)
let linkUpdatePending = false;
let linkUpdateRAF = null;

let currentZoom = 1;
let mainNodeRadius = 60;            // Bigger than interactors but not too fat
let interactorNodeRadius = 32;      // Standard size for interactor nodes
let expandedNodeRadius = 45;        // Expanded nodes (midway between main and interactor)
let interactorR = 950;              // Interactor ring radius (recalculated in buildInitialGraph)
let linkGroup, nodeGroup;            // D3 selections for links and nodes

// PERFORMANCE: Cache main node to avoid O(N) search on every tick for shared links
let cachedMainNode = null;

// PERFORMANCE: Node lookup map for O(1) access instead of O(N) find operations
let nodeMap = new Map(); // Map<nodeId, node>

/**
 * Rebuilds the node lookup map for O(1) access
 * Call this after any operation that modifies the nodes array
 */
function rebuildNodeMap() {
  nodeMap.clear();
  nodes.forEach(n => nodeMap.set(n.id, n));
}

function initNetwork(){
  const container = document.getElementById('network');
  if (!container) return;

  const fallbackWidth = Math.max(window.innerWidth * 0.75, 960);
  const fallbackHeight = Math.max(window.innerHeight * 0.65, 640);
  width = container.clientWidth || fallbackWidth;
  height = container.clientHeight || fallbackHeight;

  svg = d3.select('#svg').attr('width', width).attr('height', height);

  graphInitialFitDone = false;
  if (fitToViewTimer) {
    clearTimeout(fitToViewTimer);
    fitToViewTimer = null;
  }

  zoomBehavior = d3.zoom()
    .scaleExtent([0.35, 2.8])
    .on('zoom', (ev) => {
      if (g) {
        g.attr('transform', ev.transform);
      }
      currentZoom = ev.transform.k;
    });

  svg.call(zoomBehavior);
  g = svg.append('g');

  // Arrowheads
  const defs = svg.append('defs');
  ['activate','inhibit','binding'].forEach(type=>{
    const color = type==='activate' ? '#059669' : type==='inhibit' ? '#dc2626' : '#7c3aed';
    if (type==='activate'){
      defs.append('marker').attr('id','arrow-activate').attr('viewBox','0 -5 10 10').attr('refX',10).attr('refY',0)
          .attr('markerWidth',10).attr('markerHeight',10).attr('orient','auto')
          .append('path').attr('d','M0,-5L10,0L0,5L3,0Z').attr('fill',color);
    } else if (type==='inhibit'){
      defs.append('marker').attr('id','arrow-inhibit').attr('viewBox','0 -5 10 10').attr('refX',10).attr('refY',0)
          .attr('markerWidth',10).attr('markerHeight',10).attr('orient','auto')
          .append('rect').attr('x',6).attr('y',-4).attr('width',3).attr('height',8).attr('fill',color);
    } else {
      const m = defs.append('marker').attr('id','arrow-binding').attr('viewBox','0 -5 10 10').attr('refX',10).attr('refY',0)
          .attr('markerWidth',10).attr('markerHeight',10).attr('orient','auto');
      m.append('rect').attr('x',4).attr('y',-4).attr('width',2).attr('height',8).attr('fill',color);
      m.append('rect').attr('x',7).attr('y',-4).attr('width',2).attr('height',8).attr('fill',color);
    }
  });
  // Distinct marker for 'regulates' (amber diamond)
  const reg = defs.append('marker')
    .attr('id','arrow-regulate')
    .attr('viewBox','0 -5 10 10')
    .attr('refX',10)
    .attr('refY',0)
    .attr('markerWidth',10)
    .attr('markerHeight',10)
    .attr('orient','auto');
  reg.append('path')
    .attr('d','M0,0 L5,-4 L10,0 L5,4 Z')
    .attr('fill','#d97706');

  // Node Gradients - Light Mode
  const mainGrad = defs.append('radialGradient').attr('id', 'mainGradient');
  mainGrad.append('stop').attr('offset', '0%').attr('stop-color', '#6366f1');
  mainGrad.append('stop').attr('offset', '100%').attr('stop-color', '#4338ca');

  const interactorGrad = defs.append('radialGradient').attr('id', 'interactorGradient');
  interactorGrad.append('stop').attr('offset', '0%').attr('stop-color', '#525252');
  interactorGrad.append('stop').attr('offset', '100%').attr('stop-color', '#404040');

  // Node Gradients - Dark Mode
  const mainGradDark = defs.append('radialGradient').attr('id', 'mainGradientDark');
  mainGradDark.append('stop').attr('offset', '0%').attr('stop-color', '#818cf8');
  mainGradDark.append('stop').attr('offset', '100%').attr('stop-color', '#6366f1');

  const interactorGradDark = defs.append('radialGradient').attr('id', 'interactorGradientDark');
  interactorGradDark.append('stop').attr('offset', '0%').attr('stop-color', '#404040');
  interactorGradDark.append('stop').attr('offset', '100%').attr('stop-color', '#262626');

  // Expanded Node Gradients - Distinct from main, darker glow
  const expandedGrad = defs.append('radialGradient').attr('id', 'expandedGradient');
  expandedGrad.append('stop').attr('offset', '0%').attr('stop-color', '#c7d2fe'); // Light indigo (indigo-200)
  expandedGrad.append('stop').attr('offset', '100%').attr('stop-color', '#a5b4fc'); // Light indigo (indigo-300)

  const expandedGradDark = defs.append('radialGradient').attr('id', 'expandedGradientDark');
  expandedGradDark.append('stop').attr('offset', '0%').attr('stop-color', '#a5b4fc'); // Light indigo (indigo-300)
  expandedGradDark.append('stop').attr('offset', '100%').attr('stop-color', '#818cf8'); // Light indigo (indigo-400)

   buildInitialGraph();
   // snapshot base graph ids (non-removable)
   baseNodes = new Set(nodes.map(n => n.id));
   baseLinks = new Set(links.map(l => l.id));
   // PERFORMANCE: Cache main node reference for O(1) lookup in calculateLinkPath
   cachedMainNode = nodes.find(n => n.type === 'main');
   // PERFORMANCE: Build node lookup map for O(1) access
   rebuildNodeMap();
   createSimulation();
}

// calculateSpacing function removed - logic now inline in buildInitialGraph()

function arrowKind(rawArrow, intent, direction){
  const arrowValue = (rawArrow || '').toString().trim().toLowerCase();
  const intentValue = (intent || '').toString().trim().toLowerCase();

  // Comprehensive activation terms
  const activateTerms = ['activate','activates','activation','enhance','enhances','promote','promotes','upregulate','upregulates','stabilize','stabilizes'];
  // Comprehensive inhibition terms
  const inhibitTerms = ['inhibit','inhibits','inhibition','suppress','suppresses','repress','represses','downregulate','downregulates','block','blocks','reduce','reduces'];

  // Check arrow value for activation
  if (activateTerms.some(term => arrowValue.includes(term))) {
    return 'activates';
  }
  // Check arrow value for inhibition
  if (inhibitTerms.some(term => arrowValue.includes(term))) {
    return 'inhibits';
  }
  // Regulation/modulation normalization
  if (arrowValue === 'regulates' || arrowValue.includes('regulat') || arrowValue === 'modulates' || arrowValue.includes('modulat')) {
    return 'regulates';
  }
  // Exact binding match
  if (arrowValue === 'binds' || arrowValue === 'binding') {
    return 'binds';
  }
  // Additional arrow value checks
  if (arrowValue === 'activator' || arrowValue === 'positive') {
    return 'activates';
  }
  if (arrowValue === 'negative') {
    return 'inhibits';
  }
  // If arrow is undirected/unknown, check intent
  if (!arrowValue || ['undirected','unknown','none','na','n/a','bidirectional','both','reciprocal','neutral'].includes(arrowValue)) {
    if (intentValue === 'activation' || intentValue === 'activates') return 'activates';
    if (intentValue === 'inhibition' || intentValue === 'inhibits') return 'inhibits';
    if (intentValue === 'regulation' || intentValue === 'modulation' || intentValue === 'regulates' || intentValue === 'modulates') return 'regulates';
    if (intentValue === 'binding') return 'binds';
    return 'binds';
  }
  // Check intent as fallback
  if (intentValue === 'binding') {
    return 'binds';
  }
  if (intentValue === 'activation') {
    return 'activates';
  }
  if (intentValue === 'inhibition') {
    return 'inhibits';
  }
  // Final fallback
  return ['activates','inhibits','binds','regulates'].includes(arrowValue) ? arrowValue : 'binds';
}

function isBiDir(dir){
  const v = (dir||'').toLowerCase();
  return v==='bidirectional'||v==='undirected'||v==='both'||v==='reciprocal';
}

/**
 * Calculate node depths using breadth-first search from main protein.
 * Ignores backend metadata (depth, interaction_type) - uses only graph structure.
 *
 * @param {Array} interactions - Array of interaction objects with source/target
 * @param {string} mainProtein - ID of the main protein node
 * @returns {Map<string, number>} Map of nodeId → depth (distance from main)
 */
function calculateDepthsFromGraph(interactions, mainProtein) {
  const depthMap = new Map();
  const queue = [];
  const visited = new Set();

  // Start BFS from main protein
  depthMap.set(mainProtein, 0);
  queue.push(mainProtein);
  visited.add(mainProtein);

  while (queue.length > 0) {
    const currentNode = queue.shift();
    const currentDepth = depthMap.get(currentNode);

    // Find all neighbors of currentNode
    interactions.forEach(interaction => {
      const source = interaction.source;
      const target = interaction.target;

      // Determine neighbor (the other endpoint)
      let neighbor = null;
      if (source === currentNode) {
        neighbor = target;
      } else if (target === currentNode) {
        neighbor = source;
      } else {
        return; // This interaction doesn't involve currentNode
      }

      // Skip if already visited (first visit = shortest path)
      if (visited.has(neighbor)) {
        return;
      }

      // Set depth and mark as visited
      const newDepth = currentDepth + 1;
      depthMap.set(neighbor, newDepth);
      visited.add(neighbor);
      queue.push(neighbor);
    });
  }

  return depthMap;
}

/**
 * Filter interactions based on view mode to handle duplicates (NET vs DIRECT)
 */
function filterInteractionsByViewMode(interactions) {
  const mode = getCurrentViewMode ? getCurrentViewMode() : 'direct';

  if (mode === 'both') {
    return interactions;  // Show everything
  }

  // Group interactions by source-target pair (normalized alphabetically)
  const pairMap = new Map();  // "PROTEIN1::PROTEIN2" -> [interactions]

  interactions.forEach(int => {
    const src = int.source;
    const tgt = int.target;
    if (!src || !tgt) return;

    // Create normalized key (alphabetical order for consistency)
    const pairKey = src < tgt ? `${src}::${tgt}` : `${tgt}::${src}`;

    if (!pairMap.has(pairKey)) {
      pairMap.set(pairKey, []);
    }
    pairMap.get(pairKey).push(int);
  });

  // Filter each group based on view mode
  const filtered = [];
  pairMap.forEach((group, pairKey) => {
    if (group.length === 1) {
      // Only one interaction for this pair - always include it
      filtered.push(group[0]);
      return;
    }

    // Multiple interactions for same pair - apply filtering
    if (mode === 'direct') {
      // Prefer DIRECT mediator links over NET effects
      const directLink = group.find(int =>
        int._direct_mediator_link ||
        int.function_context === 'direct' ||
        (int.data && int.data.function_context === 'direct')
      );

      if (directLink) {
        filtered.push(directLink);  // Show DIRECT link only
      } else {
        // No direct link found - show first NET effect or regular interaction
        const netEffect = group.find(int =>
          int._net_effect ||
          int.function_context === 'net' ||
          (int.data && int.data.function_context === 'net')
        );
        filtered.push(netEffect || group[0]);
      }
    } else if (mode === 'net') {
      // Prefer NET effects over DIRECT links
      const netEffect = group.find(int =>
        int._net_effect ||
        int.function_context === 'net' ||
        (int.data && int.data.function_context === 'net')
      );

      if (netEffect) {
        filtered.push(netEffect);  // Show NET effect only
      } else {
        // No net effect found - show first DIRECT link or regular interaction
        const directLink = group.find(int =>
          int._direct_mediator_link ||
          int.function_context === 'direct' ||
          (int.data && int.data.function_context === 'direct')
        );
        filtered.push(directLink || group[0]);
      }
    }
  });

  console.log(`🔍 View mode: ${mode} - Filtered ${interactions.length} → ${filtered.length} interactions`);
  return filtered;
}

function buildInitialGraph(){
  // Clear arrays to prevent duplicates on refresh
  nodes = [];
  links = [];

  // NEW: Use proteins array for node creation, interactions array for links
  let proteins = SNAP.proteins || [];
  let interactions = SNAP.interactions || [];

  // Filter interactions based on view mode (NET vs DIRECT)
  interactions = filterInteractionsByViewMode(interactions);

  if (!SNAP.main || proteins.length === 0) {
    console.error('❌ buildInitialGraph: Missing data');
    const networkDiv = document.getElementById('network');
    if(networkDiv) {
        networkDiv.innerHTML = `
          <div style="padding: 60px 40px; text-align: center; color: #ef4444; font-family: system-ui, sans-serif;">
            <h2 style="font-size: 24px; margin-bottom: 16px;">⚠️ No Interaction Data Available</h2>
            <p style="font-size: 16px; color: #6b7280; margin-bottom: 8px;">
              ${SNAP.main ? `Protein: <strong>${SNAP.main}</strong>` : 'Unknown protein'}
            </p>
          </div>
        `;
    }
    return;
  }

  // Create main protein node
  nodes.push({
    id: SNAP.main,
    label: SNAP.main,
    type: 'main',
    radius: mainNodeRadius,
    // Initial position at center
    x: width/2,
    y: height/2,
    fx: width/2,
    fy: height/2
  });

  // Create interactor nodes
  proteins.forEach(p => {
      if (p !== SNAP.main) {
          nodes.push({
              id: p,
              label: p,
              type: 'interactor',
              radius: interactorNodeRadius,
              x: width/2 + (Math.random() - 0.5) * 100,
              y: height/2 + (Math.random() - 0.5) * 100
          });
      }
  });

  // Rebuild node map for O(1) access
  rebuildNodeMap();

  // Create links
  const linkIds = new Set();
  interactions.forEach(interaction => {
      const source = interaction.source;
      const target = interaction.target;

      if (!source || !target) return;
      if (!nodeMap.has(source) || !nodeMap.has(target)) return;

      const arrow = arrowKind(interaction.arrow, interaction.intent, interaction.direction);
      const linkId = `${source}-${target}-${arrow}`;

      if (linkIds.has(linkId)) return;

      links.push({
          id: linkId,
          source: source,
          target: target,
          type: 'interaction',
          arrow: arrow,
          direction: interaction.direction,
          data: interaction,
          isBidirectional: false // Simplification: Handle bidirectional visuals in CSS/marker logic if needed
      });
      linkIds.add(linkId);
  });
  
  // Calculate depths (bfs)
  depthMap.clear();
  const calculatedDepths = calculateDepthsFromGraph(interactions, SNAP.main);
  calculatedDepths.forEach((depth, nodeId) => depthMap.set(nodeId, depth));
}

/**
 * Creates standard D3 force simulation
 */
function createSimulation(){
  // Simple force layout
  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d=>d.id).distance(350)) // Reasonable link distance
    .force('charge', d3.forceManyBody().strength(-800)) // Repel nodes
    .force('center', d3.forceCenter(width / 2, height / 2)) // Keep in center
    .force('collide', d3.forceCollide().radius(d => (d.type==='main' ? mainNodeRadius : interactorNodeRadius) + 30).iterations(2));

  simulation.alpha(1).restart();

  // LINKS
  const link = g.append('g').selectAll('path')
    .data(links).enter().append('path')
    .attr('class', d=>{
      const arrow = d.arrow||'binds';
      let classes = 'link';
      if (arrow==='binds') classes += ' link-binding';
      else if (arrow==='activates') classes += ' link-activate';
      else if (arrow==='inhibits') classes += ' link-inhibit';
      else if (arrow==='regulates') classes += ' link-regulate';
      else classes += ' link-binding';
      return classes;
    })
    .attr('marker-end', d=>{
       // Simple marker logic
       const a=d.arrow||'binds';
       if (a==='activates') return 'url(#arrow-activate)';
       if (a==='inhibits') return 'url(#arrow-inhibit)';
       if (a==='regulates') return 'url(#arrow-regulate)';
       return 'url(#arrow-binding)';
    })
    .attr('fill','none')
    .on('mouseover', function(){ d3.select(this).style('stroke-width','3.5'); svg.style('cursor','pointer'); })
    .on('mouseout',  function(){ d3.select(this).style('stroke-width',null);  svg.style('cursor',null); })
    .on('click', handleLinkClick);

  // NODES
  const node = g.append('g').selectAll('g')
    .data(nodes).enter().append('g')
    .attr('class','node-group')
    .call(d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended));

  node.each(function(d){
    const group = d3.select(this);
    if (d.type==='main'){
      group.append('circle')
        .attr('class','node main-node')
        .attr('r', mainNodeRadius)
        .style('cursor','pointer')
        .on('click', (ev)=>{ ev.stopPropagation(); handleNodeClick(d); });
      group.append('text').attr('class','node-label main-label').attr('dy',5).text(d.label);
    } else {
      group.append('circle')
        .attr('class', 'node interactor-node')
        .attr('r', interactorNodeRadius)
        .style('cursor','pointer')
        .on('click', (ev)=>{ ev.stopPropagation(); handleNodeClick(d); });
      group.append('text').attr('class','node-label').attr('dy',5).text(d.label);
    }
  });

  // Tick handler
  simulation.on('tick', ()=>{
    node.attr('transform', d=> `translate(${d.x},${d.y})`);
    link.attr('d', calculateLinkPath);
  });

  // Store selections
  linkGroup = link;
  nodeGroup = node;
}

function dragstarted(ev, d){
  if (!ev.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(ev, d){
  d.fx = ev.x;
  d.fy = ev.y;
}

function dragended(ev, d){
  if (!ev.active) simulation.alphaTarget(0);
  if (d.type !== 'main') { // Keep main fixed if desired, or release
     d.fx = null;
     d.fy = null;
  }
}

/**
 * Simple link path calculation
 */
function calculateLinkPath(d) {
  const sourceNode = typeof d.source === 'object' ? d.source : nodeMap.get(d.source);
  const targetNode = typeof d.target === 'object' ? d.target : nodeMap.get(d.target);

  if (!sourceNode || !targetNode) return 'M 0 0';

  const x1 = sourceNode.x;
  const y1 = sourceNode.y;
  const x2 = targetNode.x;
  const y2 = targetNode.y;

  return `M ${x1} ${y1} L ${x2} ${y2}`;
}

function forceClusterBounds() { return () => {}; } // No-op
function forceIndirectClustering() { return () => {}; } // No-op
function initializeClusterLayout() {} // No-op


// Drag handlers removed - static layout with fixed positions
// User can zoom/pan the entire graph, but nodes don't move individually

/* ===============================================================
   MODAL SYSTEM
   =============================================================== */

function openModal(titleHTML, bodyHTML){
  document.getElementById('modalTitle').innerHTML = titleHTML;
  document.getElementById('modalBody').innerHTML = bodyHTML;
  document.getElementById('modal').classList.add('active');

  // Wire up expandable rows after modal opens
  setTimeout(() => {
    // Function expandable rows
    document.querySelectorAll('.function-expandable-row').forEach(row => {
      const header = row.querySelector('.function-row-header');
      if (header) {
        header.addEventListener('click', () => {
          row.classList.toggle('expanded');
        });
      }
    });

    // Interaction expandable rows
    document.querySelectorAll('.interaction-expandable-row').forEach(row => {
      const header = row.querySelector('.interaction-row-header');
      const content = row.querySelector('.interaction-expanded-content');
      const icon = row.querySelector('.interaction-expand-icon');
      if (header && content) {
        header.addEventListener('click', () => {
          const isExpanded = row.classList.contains('expanded');
          if (isExpanded) {
            row.classList.remove('expanded');
            content.style.maxHeight = '0';
            content.style.opacity = '0';
            if (icon) icon.style.transform = 'rotate(0deg)';
          } else {
            row.classList.add('expanded');
            content.style.maxHeight = '2000px';
            content.style.opacity = '1';
            if (icon) icon.style.transform = 'rotate(180deg)';
          }
        });
      }
    });
  }, 100);
}

function closeModal(){
  document.getElementById('modal').classList.remove('active');
}

document.getElementById('modal').addEventListener('click', (e)=>{
  if (e.target.id==='modal') closeModal();
});

/* Helper: Render an expandable function row */
function renderExpandableFunction(fn, mainProtein, interactorProtein, defaultInteractionEffect, parentDirection){
  const functionName = escapeHtml(fn.function || 'Function');

  // --- DATA PREPARATION (From Table View Logic) ---
  const fnDirection = fn.interaction_direction || parentDirection || fn.direction || 'main_to_primary';
  let sourceProtein, targetProtein, arrowSymbol;
  if (fnDirection === 'primary_to_main') {
    sourceProtein = interactorProtein;
    targetProtein = mainProtein;
    arrowSymbol = '→';
  } else if (fnDirection === 'bidirectional') {
    sourceProtein = mainProtein;
    targetProtein = interactorProtein;
    arrowSymbol = '↔';
  } else {
    // main_to_primary
    sourceProtein = mainProtein;
    targetProtein = interactorProtein;
    arrowSymbol = '→';
  }

  // Interaction Effect (Protein Level)
  // Use defaultInteractionEffect (from Link) as the source of truth for the protein-protein effect
  let interactionEffect = defaultInteractionEffect || 'binds';
  
  // Function Effect (Function Level)
  // Use fn.arrow as the source of truth for the function effect
  let fnArrow = fn.arrow || 'binds';

  // Context override: if link is "binds" but function is specific in a chain context, upgrade link effect for display context
  if (interactionEffect === 'binds' && fn._context && fn._context.type === 'chain') {
    if (fnArrow === 'activates' || fnArrow === 'inhibits') {
      interactionEffect = fnArrow;
    }
  }

  // Calculate Classes & Text
  const interactionArrowClass = arrowKind(interactionEffect, fn.intent, fnDirection);
  const interactionEffectBadgeText = formatArrow(interactionEffect);
  const interactionEffectBadge = `<span class="effect-badge effect-${interactionArrowClass}">${interactionEffectBadgeText}</span>`;

  const functionArrowClass = arrowKind(fnArrow, fn.intent, fnDirection);
  const functionEffectBadgeText = formatArrow(fnArrow);
  const functionEffectBadge = `<span class="effect-badge effect-${functionArrowClass}">${functionEffectBadgeText}</span>`;

  // Helper Data
  const contextBadge = fn._context ? (fn._context.type === 'chain' ? '<span class="context-badge" style="background: #f59e0b; color: white; font-size: 9px; padding: 2px 6px; border-radius: 3px; margin-left: 6px;">CHAIN CONTEXT</span>' : '<span class="context-badge" style="background: #10b981; color: white; font-size: 9px; padding: 2px 6px; border-radius: 3px; margin-left: 6px;">DIRECT PAIR</span>') : '';

  // Interaction Display (Header)
  const interactionDisplay = `
    <span class="detail-interaction">
      ${escapeHtml(sourceProtein)}
      <span class="detail-arrow">${arrowSymbol}</span>
      ${escapeHtml(targetProtein)}
    </span>
    ${interactionEffectBadge}
  `;

  // --- CONTENT CONSTRUCTION (Restoring "Pretty" Layout) ---
  let expandedSections = '';

  // 1. Effects Summary (Enhanced with Table View Data)
  expandedSections += `
    <div class="function-detail-section section-effects-summary section-highlighted" style="background: var(--color-bg-secondary); border-left: 3px solid var(--color-primary);">
      <div class="function-section-title">🎯 Effects Summary</div>
      <div class="function-section-content">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
          <div>
            <div style="font-size: 0.75rem; text-transform: uppercase; color: var(--color-text-secondary); margin-bottom: 4px;">Interaction</div>
            <div style="font-size: 0.9rem; margin-bottom: 4px;">
              ${escapeHtml(sourceProtein)} ${arrowSymbol} ${escapeHtml(targetProtein)}
            </div>
            ${interactionEffectBadge}
          </div>
          <div>
            <div style="font-size: 0.75rem; text-transform: uppercase; color: var(--color-text-secondary); margin-bottom: 4px;">Function</div>
            <div style="font-size: 0.9rem; margin-bottom: 4px;">
              ${escapeHtml(functionName)}
            </div>
            ${functionEffectBadge}
          </div>
        </div>
      </div>
    </div>
  `;

  // 2. Mechanism (from cellular_process)
  if (fn.cellular_process) {
    expandedSections += `
      <div class="function-detail-section section-mechanism section-highlighted">
        <div class="function-section-title">⚙️ Mechanism</div>
        <div class="function-section-content">
          <div style="margin-bottom: 8px;">${escapeHtml(fn.cellular_process)}</div>
        </div>
      </div>
    `;
  }

  // 3. Effect Description
  if (fn.effect_description) {
    expandedSections += `
      <div class="function-detail-section section-effect section-highlighted effect-${functionArrowClass}">
        <div class="function-section-title">💡 Effect</div>
        <div class="function-section-content">${escapeHtml(fn.effect_description)}</div>
      </div>
    `;
  }

  // 4. Biological Cascade
  if (Array.isArray(fn.biological_consequence) && fn.biological_consequence.length > 0) {
    const cascadesHTML = fn.biological_consequence.map((cascade, idx) => {
        const text = (cascade == null ? '' : cascade).toString().trim();
        if (!text) return '';
        const steps = text.split('→').map(s => s.trim()).filter(s => s.length > 0);
        if (steps.length === 0) return '';
        return `
          <div class="cascade-scenario">
            <div class="cascade-scenario-label">Scenario ${idx + 1}</div>
            <div class="cascade-flow-container">
              ${steps.map(step => `<div class="cascade-flow-item">${escapeHtml(step)}</div>`).join('')}
            </div>
          </div>
        `;
      }).join('');

    if (cascadesHTML) {
      expandedSections += `
        <div class="function-detail-section">
          <div class="function-section-title">Biological Cascade</div>
          ${cascadesHTML}
        </div>
      `;
    }
  }

  // 5. Specific Effects
  if (Array.isArray(fn.specific_effects) && fn.specific_effects.length > 0) {
    expandedSections += `
      <div class="function-detail-section section-specific-effects section-highlighted">
        <div class="function-section-title">⚡ Specific Effects</div>
        <ul style="margin: 0; padding-left: 1.5em;">
          ${fn.specific_effects.map(eff => `<li class="function-section-content">${escapeHtml(eff)}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  // 6. Evidence (Pretty Card Style)
  if (Array.isArray(fn.evidence) && fn.evidence.length > 0) {
    expandedSections += `
      <div class="function-detail-section">
        <div class="function-section-title">Evidence & Publications</div>
        ${fn.evidence.map(ev => {
          const title = ev.paper_title || (ev.pmid ? `PMID: ${ev.pmid}` : 'Untitled');
          const metaParts = [];
          if (ev.journal) metaParts.push(escapeHtml(ev.journal));
          if (ev.year) metaParts.push(escapeHtml(ev.year));
          const meta = metaParts.join(' · ');
          
          let pmidLinks = '';
          if (ev.pmid) pmidLinks += `<a href="https://pubmed.ncbi.nlm.nih.gov/${escapeHtml(ev.pmid)}" target="_blank" class="pmid-badge" onclick="event.stopPropagation();">PMID: ${escapeHtml(ev.pmid)}</a>`;
          if (ev.doi) pmidLinks += `<a href="https://doi.org/${escapeHtml(ev.doi)}" target="_blank" class="pmid-badge" onclick="event.stopPropagation();">DOI</a>`;

          return `
            <div class="evidence-card">
              <div class="evidence-title">${escapeHtml(title)}</div>
              ${meta ? `<div class="evidence-meta">${meta}</div>` : ''}
              ${ev.relevant_quote ? `<div class="evidence-quote">"${escapeHtml(ev.relevant_quote)}"</div>` : ''}
              ${pmidLinks ? `<div style="margin-top: var(--space-2);">${pmidLinks}</div>` : ''}
            </div>
          `;
        }).join('')}
      </div>
    `;
  } else if (fn.pmids && fn.pmids.length > 0) {
    expandedSections += `
      <div class="function-detail-section">
        <div class="function-section-title">References</div>
        <div>
          ${fn.pmids.map(pmid => `<a href="https://pubmed.ncbi.nlm.nih.gov/${escapeHtml(pmid)}" target="_blank" class="pmid-badge">PMID: ${escapeHtml(pmid)}</a>`).join('')}
        </div>
      </div>
    `;
  }

  // Build Final Row HTML
  return `
    <div class="function-expandable-row">
      <div class="function-row-header">
        <div class="function-row-left">
          <div class="function-expand-icon">▼</div>
          <div class="function-name-with-effect">
            <div class="function-name-display">${functionName}</div>
            ${functionEffectBadge}
          </div>
          <span class="function-separator" style="margin: 0 8px; color: var(--color-text-secondary);">||</span>
          ${interactionDisplay}
          ${contextBadge}
        </div>
      </div>
      <div class="function-expanded-content">
        ${expandedSections || '<div class="function-section-content" style="color: var(--color-text-secondary);">No additional details available</div>'}
      </div>
    </div>
  `;
}

function handleLinkClick(ev, d){
  ev.stopPropagation();
  if (!d) return;
  if (d.type==='function'){
    showFunctionModalFromLink(d);
  } else if (d.type==='interaction'){
    showInteractionModal(d);
  }
}

/* ===============================================================
   Interaction Modal: NEW DESIGN with Expandable Functions
   =============================================================== */
function showInteractionModal(link, clickedNode = null){
  const L = link.data || link;  // Link properties are directly on link object or in data

  // Use semantic source/target (biological direction) instead of D3's geometric source/target
  // Semantic fields preserve the biological meaning, while link.source/target are D3 node references
  const srcName = L.semanticSource || ((link.source && link.source.id) ? link.source.id : link.source);
  const tgtName = L.semanticTarget || ((link.target && link.target.id) ? link.target.id : link.target);
  const safeSrc = escapeHtml(srcName || '-');
  const safeTgt = escapeHtml(tgtName || '-');

  // Determine which protein was clicked (if any)
  // If called from node click, use clickedNode; otherwise determine from link
  let clickedProteinId = null;
  if (clickedNode) {
    clickedProteinId = clickedNode.id;
  }

  // Determine arrow direction
  // IMPORTANT: Direction field has different semantics for direct vs indirect interactions
  // - Direct: direction is QUERY-RELATIVE (main_to_primary = query→interactor)
  // - Indirect: direction is LINK-ABSOLUTE (main_to_primary = source→target after transformation)
  const direction = L.direction || link.direction || 'main_to_primary';
  const isIndirect = L.interaction_type === 'indirect';
  const directionIsLinkAbsolute = L._direction_is_link_absolute || isIndirect;

  let arrowSymbol = '↔';
  if (direction === 'bidirectional' || direction === 'undirected') {
    arrowSymbol = '↔';
  } else {
    // For all directed links (main_to_primary, primary_to_main, a_to_b, b_to_a),
    // the link source is the actor, so the arrow is always source -> target.
    arrowSymbol = '→';
  }

  // === BUILD INTERACTION METADATA SECTION ===

  let functionTypeBadge = '';
  if (isSharedInteraction) {
    functionTypeBadge = '<span class="mechanism-badge" style="background: #9333ea; color: white; font-size: 9px; padding: 2px 6px;">SHARED</span>';
  } else if (isIndirectInteraction) {
    // Build full chain path display for INDIRECT label
    // Try to extract chain from first function with chain context
    let chainDisplay = '';
    const firstChainFunc = functions.find(f => f._context && f._context.type === 'chain' && f._context.chain);
    if (firstChainFunc && firstChainFunc._context.chain) {
      chainDisplay = buildFullChainPath(SNAP.main, firstChainFunc._context.chain, L);
    }

    // Fallback: use upstream_interactor if no chain found
    if (!chainDisplay && L.upstream_interactor) {
      chainDisplay = `${escapeHtml(SNAP.main)} → ${escapeHtml(L.upstream_interactor)} → ${escapeHtml(L.primary)}`;
    }

    functionTypeBadge = chainDisplay
      ? `<span class="mechanism-badge" style="background: #f59e0b; color: white; font-size: 9px; padding: 2px 6px;">${chainDisplay}</span>`
      : `<span class="mechanism-badge" style="background: #f59e0b; color: white; font-size: 9px; padding: 2px 6px;">INDIRECT</span>`;
  } else {
    functionTypeBadge = '<span class="mechanism-badge" style="background: #10b981; color: white; font-size: 9px; padding: 2px 6px;">DIRECT</span>';
  }

  if (functions.length > 0) {
    if (isIndirectInteraction) {
      // For indirect interactions: Don't group by direction - show all together
      // Direction is no longer query-relative, so grouping would be confusing
      const arrows = L.arrows || {};
      const arrowCount = Object.values(arrows).flat().filter((v, i, a) => a.indexOf(v) === i).length;

      functionsHTML = `<div class="modal-functions-header">Functions (${functions.length})${arrowCount > 1 ? ` <span style="background:#f59e0b;color:white;padding:2px 6px;border-radius:10px;font-size:10px;margin-left:8px;">${arrowCount} arrows</span>` : ''}</div>`;

      // Display all functions without direction grouping
      functionsHTML += `<div style="margin:16px 0;">
        ${functions.map(f => {
          const effectArrow = f.arrow || 'complex';
          // For indirect interactions, always render relative to Main -> Indirect Target
          return renderExpandableFunction(f, SNAP.main, tgtName, interactionArrow, direction);
        }).join('')}
      </div>`;

    } else {
      // For direct interactions: Group by INTERACTION DIRECTION
      // Functions should be grouped by which protein acts on which, showing the directionality
      const grp = {
        main_to_primary: [],
        primary_to_main: [],
        bidirectional: []
      };
      functions.forEach(f => grp[(f.direction || 'main_to_primary')].push(f));

      const arrows = L.arrows || {};
      const arrowCount = Object.values(arrows).flat().filter((v, i, a) => a.indexOf(v) === i).length;

      // Determine protein names for direction labels
      const queryProtein = SNAP.main;
      const interactorProtein = safeSrc === queryProtein ? safeTgt : safeSrc;

      functionsHTML = `<div class="modal-functions-header">Functions (${functions.length})${arrowCount > 1 ? ` <span style="background:#f59e0b;color:white;padding:2px 6px;border-radius:10px;font-size:10px;margin-left:8px;">${arrowCount} arrows</span>` : ''}</div>`;

      // Direction labels with arrow symbols based on interaction type
      const directionConfig = {
        main_to_primary: {
          source: queryProtein,
          target: interactorProtein,
          arrowSymbol: '→',
          color: '#3b82f6',  // Blue
          bg: '#dbeafe'
        },
        primary_to_main: {
          source: interactorProtein,
          target: queryProtein,
          arrowSymbol: '→',
          color: '#9333ea',  // Purple
          bg: '#f3e8ff'
        },
        bidirectional: {
          source: queryProtein,
          target: interactorProtein,
          arrowSymbol: '↔',
          color: '#059669',  // Green
          bg: '#d1fae5'
        }
      };

      ['main_to_primary', 'primary_to_main', 'bidirectional'].forEach(dir => {
        if (grp[dir].length) {
          const config = directionConfig[dir];
          functionsHTML += `<div style="">
            <div style="">
              <span class="detail-interaction">
                ${escapeHtml(config.source)}
                <span class="detail-arrow">${config.arrowSymbol}</span>
                ${escapeHtml(config.target)}
              </span> (${grp[dir].length})
            </div>
            ${grp[dir].map(f => {
              // Within each direction, show effect type badge
              const effectArrow = f.arrow || 'complex';
              // Pass SNAP.main and interactorName to ensure correct direction resolution
              // FIX: Pass interactionArrow as defaultInteractionEffect, NOT effectArrow
              return renderExpandableFunction(f, SNAP.main, interactorName, interactionArrow, direction);
            }).join('')}
          </div>`;
        }
      });
    }
  } else {
    const emptyMessage = isSharedInteraction
      ? 'Shared interactions may not include context-specific functions.'
      : 'No functions associated with this interaction.';
    functionsHTML = `
      <div class="modal-functions-header">Functions</div>
      <div style="padding: var(--space-4); color: var(--color-text-secondary); font-style: italic;">
        ${emptyMessage}
      </div>
    `;
  }

  // === BUILD EXPAND/COLLAPSE FOOTER (if called from node click) ===
  let footerHTML = '';
  if (clickedProteinId) {
    const proteinLabel = clickedProteinId;
    const isMainProtein = clickedProteinId === SNAP.main;
    const isExpanded = expanded.has(clickedProteinId);
    const canExpand = (depthMap.get(clickedProteinId) ?? 1) < MAX_DEPTH;
    const hasInteractions = true; // Always true for showInteractionModal (single link exists)

    if (isMainProtein) {
      // Main protein: show single "Find New Interactions" button
      footerHTML = `
        <div class="modal-footer" style="border-top: 1px solid var(--color-border); padding: 16px; background: var(--color-bg-secondary);">
          <button onclick="handleQueryFromModal('${clickedProteinId}')" class="btn-primary" style="padding: 8px 20px; background: #10b981; color: white; border: none; border-radius: 6px; font-weight: 500; cursor: pointer; font-size: 14px; font-family: var(--font-sans); transition: background 0.2s;">
            Find New Interactions
          </button>
        </div>
      `;
    } else {
      // Interactor: show conditional Expand + Query buttons
      footerHTML = `
        <div class="modal-footer" style="border-top: 1px solid var(--color-border); padding: 16px; background: var(--color-bg-secondary);">
          <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
            ${canExpand && !isExpanded && hasInteractions ? `
              <button onclick="handleExpandFromModal('${clickedProteinId}')" class="btn-primary" style="padding: 8px 20px; background: #3b82f6; color: white; border: none; border-radius: 6px; font-weight: 500; cursor: pointer; font-size: 14px; font-family: var(--font-sans); transition: background 0.2s;">
                Expand
              </button>
            ` : ''}
            ${canExpand && !isExpanded && !hasInteractions ? `
              <button disabled style="padding: 8px 20px; background: #d1d5db; color: #6b7280; border: none; border-radius: 6px; font-weight: 500; font-size: 14px; cursor: not-allowed; font-family: var(--font-sans);">
                Expand (No data)
              </button>
            ` : ''}
            ${isExpanded ? `
              <button onclick="handleCollapseFromModal('${clickedProteinId}')" class="btn-secondary" style="padding: 8px 20px; background: #ef4444; color: white; border: none; border-radius: 6px; font-weight: 500; cursor: pointer; font-size: 14px; font-family: var(--font-sans); transition: background 0.2s;">
                Collapse
              </button>
            ` : ''}
            <button onclick="handleQueryFromModal('${clickedProteinId}')" class="btn-primary" style="padding: 8px 20px; background: #10b981; color: white; border: none; border-radius: 6px; font-weight: 500; cursor: pointer; font-size: 14px; font-family: var(--font-sans); transition: background 0.2s;">
              Query
            </button>
            ${!canExpand && !isExpanded ? `
              <div style="padding: 8px 20px; background: #f3f4f6; color: #6b7280; border-radius: 6px; font-size: 13px; font-family: var(--font-sans); font-style: italic;">
                Max depth reached (${MAX_DEPTH})
              </div>
            ` : ''}
          </div>
          <div style="margin-top: 12px; font-size: 12px; color: var(--color-text-secondary); font-family: var(--font-sans);">
            Expand uses existing data • Query finds new interactions
          </div>
        </div>
      `;
    }
  }

  // === BUILD MODAL TITLE WITH TYPE BADGE ===
  // Determine interaction type and create badge
  const isShared = L._is_shared_link || false;
  // isIndirect already declared at line 5518 - reuse that variable
  const mediatorChain = L.mediator_chain || [];
  const chainDepth = L.depth || 1;

  // Check if THIS interaction's target is a mediator for OTHER indirect interactions
  // (e.g., KEAP1 is mediator in p62→KEAP1→NRF2)
  const isMediator = (tgtName === L.upstream_interactor || srcName === L.upstream_interactor);

  let typeBadge = '';
  if (isShared) {
    typeBadge = '<span class="mechanism-badge" style="background: #9333ea; color: white; font-size: 10px; padding: 3px 8px; margin-left: 12px;">SHARED</span>';
  } else if (isIndirect) {
    // Build full chain path display for INDIRECT label
    // Try to extract chain from first function with chain context
    let chainDisplay = '';
    const firstChainFunc = functions.find(f => f._context && f._context.type === 'chain' && f._context.chain);
    if (firstChainFunc && firstChainFunc._context.chain) {
      chainDisplay = buildFullChainPath(SNAP.main, firstChainFunc._context.chain, L);
    }

    // Fallback: use upstream_interactor if no chain found
    if (!chainDisplay && L.upstream_interactor) {
      chainDisplay = `${escapeHtml(SNAP.main)} → ${escapeHtml(L.upstream_interactor)} → ${escapeHtml(L.primary)}`;
    }

    typeBadge = chainDisplay
      ? `<span class="mechanism-badge" style="background: #f59e0b; color: white; font-size: 10px; padding: 3px 8px; margin-left: 12px;">${chainDisplay}</span>`
      : `<span class="mechanism-badge" style="background: #f59e0b; color: white; font-size: 10px; padding: 3px 8px; margin-left: 12px;">INDIRECT</span>`;
  } else if (isMediator) {
    // This protein is a mediator in indirect chains AND this link is direct
    typeBadge = `<span class="mechanism-badge" style="background: #10b981; color: white; font-size: 10px; padding: 3px 8px; margin-left: 12px;">DIRECT</span>
                 <span class="mechanism-badge" style="background: #6366f1; color: white; font-size: 10px; padding: 3px 8px; margin-left: 4px;">MEDIATOR</span>`;
  } else {
    typeBadge = '<span class="mechanism-badge" style="background: #10b981; color: white; font-size: 10px; padding: 3px 8px; margin-left: 12px;">DIRECT</span>';
  }

  let modalTitle = `
    <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
      <span style="font-size: 18px; font-weight: 600;">${safeSrc} ${arrowSymbol} ${safeTgt}</span>
      ${typeBadge}
    </div>
  `;

  // Add full chain display for ALL indirect interactions
  if (isIndirect) {
    let fullChainText = '';
    if (mediatorChain.length > 0) {
      // CRITICAL FIX (Issue #2): Use chain_with_arrows if available for typed arrows
      const chainWithArrows = L.chain_with_arrows || [];

      if (chainWithArrows.length > 0) {
        // CRITICAL FIX (Issue #1): For shared links, use correct protein perspective
        // Check if this is a shared link and reconstruct chain from shared interactor's perspective
        if (isShared && L._shared_between && L._shared_between.length >= 2) {
          // Find the shared interactor (not the main query protein)
          const sharedInteractor = L._shared_between.find(p => p !== SNAP.main);

          if (sharedInteractor) {
            // Filter chain segments to show only those starting from shared interactor
            const relevantSegments = chainWithArrows.filter(seg =>
              seg.from === sharedInteractor || chainWithArrows.indexOf(seg) > chainWithArrows.findIndex(s => s.from === sharedInteractor)
            );

            if (relevantSegments.length > 0) {
              const arrowSymbols = {
                'activates': ' <span style="color:#059669;font-weight:700;">--&gt;</span> ',
                'inhibits': ' <span style="color:#dc2626;font-weight:700;">--|</span> ',
                'binds': ' <span style="color:#7c3aed;font-weight:700;">---</span> ',
                'complex': ' <span style="color:#f59e0b;font-weight:700;">--=</span> '
              };

              fullChainText = relevantSegments.map((segment, i) => {
                const arrow = arrowSymbols[segment.arrow] || ' → ';
                if (i === relevantSegments.length - 1) {
                  return escapeHtml(segment.from) + arrow + escapeHtml(segment.to);
                } else {
                  return escapeHtml(segment.from) + arrow;
                }
              }).join('');
            } else {
              // Fallback: shared interactor → target
              fullChainText = `${escapeHtml(sharedInteractor)} → ${escapeHtml(tgtName)}`;
            }
          } else {
            // Couldn't find shared interactor, use default
            fullChainText = chainWithArrows.map((segment, i) => {
              const arrow = arrowSymbols[segment.arrow] || ' → ';
              return i === chainWithArrows.length - 1
                ? escapeHtml(segment.from) + arrow + escapeHtml(segment.to)
                : escapeHtml(segment.from) + arrow;
            }).join('');
          }
        } else {
          // NOT a shared link: Display full chain with typed arrows
          const arrowSymbols = {
            'activates': ' <span style="color:#059669;font-weight:700;">--&gt;</span> ',
            'inhibits': ' <span style="color:#dc2626;font-weight:700;">--|</span> ',
            'binds': ' <span style="color:#7c3aed;font-weight:700;">---</span> ',
            'complex': ' <span style="color:#f59e0b;font-weight:700;">--=</span> '
          };

          fullChainText = chainWithArrows.map((segment, i) => {
            const arrow = arrowSymbols[segment.arrow] || ' → ';
            if (i === chainWithArrows.length - 1) {
              // Last segment: show "from arrow to"
              return escapeHtml(segment.from) + arrow + escapeHtml(segment.to);
            } else {
              // Middle segments: only show "from arrow" (to avoid duplication)
              return escapeHtml(segment.from) + arrow;
            }
          }).join('');
        }
      } else {
        // FALLBACK: Generic arrows (old data or no chain_with_arrows)
        // CRITICAL FIX (Issue #1): For shared links, start chain from shared interactor
        let startProtein = SNAP.main;

        if (isShared && L._shared_between && L._shared_between.length >= 2) {
          const sharedInteractor = L._shared_between.find(p => p !== SNAP.main);
          if (sharedInteractor) {
            startProtein = sharedInteractor;
          }
        }

        const fullChain = [startProtein, ...mediatorChain, tgtName];
        fullChainText = fullChain.map(p => escapeHtml(p)).join(' → ');
      }
    } else if (L.upstream_interactor && L.upstream_interactor !== SNAP.main) {
      // Indirect with single upstream (no chain array but has upstream)
      // TODO: Could enhance to look up arrow types here too
      fullChainText = `${escapeHtml(SNAP.main)} → ${escapeHtml(L.upstream_interactor)} → ${escapeHtml(tgtName)}`;
    } else {
      // First-ring indirect: no mediator specified (pathway incomplete)
      fullChainText = `${escapeHtml(SNAP.main)} → ${escapeHtml(tgtName)} <span style="font-style: italic; color: #f59e0b;">(direct mediator unknown)</span>`;
    }

    modalTitle = `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
          <span style="font-size: 18px; font-weight: 600;">${safeSrc} ${arrowSymbol} ${safeTgt}</span>
          ${typeBadge}
        </div>
        <div style="font-size: 13px; color: var(--color-text-secondary); font-weight: normal; padding: 4px 8px; background: var(--color-bg-tertiary); border-radius: 4px; border-left: 3px solid #f59e0b;">
          <strong>Full Chain:</strong> ${fullChainText}
        </div>
      </div>
    `;
  }

  // === COMBINE SECTIONS AND DISPLAY ===
  const fullModalContent = interactionMetadataHTML + functionsHTML + footerHTML;
  openModal(modalTitle, fullModalContent);
}

/* DEPRECATED: Old interactor modal - now using unified interaction modal for both arrows and nodes */
// showInteractorModal removed - nodes now use showInteractionModal with expand/collapse footer

/* Handle node click - show interaction modal with expand/collapse controls */
function handleNodeClick(node){
  // Find ALL links involving this node
  const nodeLinks = links.filter(l => {
    const src = (l.source && l.source.id) ? l.source.id : l.source;
    const tgt = (l.target && l.target.id) ? l.target.id : l.target;
    return src === node.id || tgt === node.id;
  });

  if (nodeLinks.length === 0) {
    // Fallback: show error message
    openModal(`Protein: ${escapeHtml(node.label || node.id)}`,
      '<div style="color:#6b7280; padding: 20px; text-align: center;">No interactions found for this protein.</div>');
  } else {
    // Use aggregated modal for consistent formatting (1+ interactions)
    // This ensures all modals have color-coded section headers and bordered boxes
    showAggregatedInteractionsModal(nodeLinks, node);
  }
}

/* Show aggregated modal for nodes with multiple interactions */
function showAggregatedInteractionsModal(nodeLinks, clickedNode) {
  const nodeId = clickedNode.id;
  const nodeLabel = clickedNode.label || nodeId;

  // Group links by type (direct, indirect, shared)
  const directLinks = [];
  const indirectLinks = [];
  const sharedLinks = [];

  nodeLinks.forEach(link => {
    const L = link.data || {};
    if (L._is_shared_link) {
      sharedLinks.push(link);
    } else if (L.interaction_type === 'indirect') {
      indirectLinks.push(link);
    } else {
      directLinks.push(link);
    }
  });

  // Build sections HTML
  let sectionsHTML = '';

  // Helper to render a single interaction section
  function renderInteractionSection(link, sectionType) {
    const L = link.data || link;  // Link properties are directly on link object or in data

    // Use semantic source/target (biological direction) instead of D3's geometric source/target
    const srcName = L.semanticSource || ((link.source && link.source.id) ? link.source.id : link.source);
    const tgtName = L.semanticTarget || ((link.target && link.target.id) ? link.target.id : link.target);
    const safeSrc = escapeHtml(srcName || '-');
    const safeTgt = escapeHtml(tgtName || '-');

    // Determine interactor name for function rendering
    const interactorName = srcName === SNAP.main ? tgtName : srcName;

    // Determine arguments for renderExpandableFunction
    // For indirect OR detached links (not involving main): use source/target directly
    // For direct links involving main: use Canonical Main and Interactor
    const involvesMain = (srcName === SNAP.main || tgtName === SNAP.main);
    const useOwnEndpoints = (sectionType === 'indirect') || !involvesMain;
    
    const renderArg1 = useOwnEndpoints ? srcName : SNAP.main;
    const renderArg2 = useOwnEndpoints ? tgtName : interactorName;

    // Determine arrow symbol
    // Support both query-relative AND absolute directions
    const direction = L.direction || link.direction || 'main_to_primary';
    let arrowSymbol = '↔';
    if (direction === 'bidirectional' || direction === 'undirected') {
        arrowSymbol = '↔';
    } else {
        // For all directed links (main_to_primary, primary_to_main, a_to_b, b_to_a), 
        // the link source is the actor, so the arrow is always source -> target.
        arrowSymbol = '→';
    }

    // Type badge
    let typeBadgeHTML = '';
    if (sectionType === 'shared') {
      typeBadgeHTML = '<span class="mechanism-badge" style="background: #9333ea; color: white;">SHARED</span>';
    } else if (sectionType === 'indirect') {
      // Build full chain path display for INDIRECT label
      // Try to extract chain from first function with chain context
      let chainDisplay = '';
      const functions = L.functions || [];
      const firstChainFunc = functions.find(f => f._context && f._context.type === 'chain' && f._context.chain);
      if (firstChainFunc && firstChainFunc._context.chain) {
        chainDisplay = buildFullChainPath(SNAP.main, firstChainFunc._context.chain, L);
      }

      // Fallback: use upstream_interactor if no chain found
      if (!chainDisplay && L.upstream_interactor) {
        chainDisplay = `${escapeHtml(SNAP.main)} → ${escapeHtml(L.upstream_interactor)} → ${escapeHtml(L.primary)}`;
      }

      typeBadgeHTML = chainDisplay
        ? `<span class="mechanism-badge" style="background: #f59e0b; color: white;">${chainDisplay}</span>`
        : `<span class="mechanism-badge" style="background: #f59e0b; color: white;">INDIRECT</span>`;
    } else {
      typeBadgeHTML = '<span class="mechanism-badge" style="background: #10b981; color: white;">DIRECT</span>';
    }

    // Interaction title
    const interactionTitle = `${safeSrc} ${arrowSymbol} ${safeTgt}`;

    // Arrow type badge
    const arrow = L.arrow || link.arrow || 'binds';
    const normalizedArrow = arrow === 'activates' || arrow === 'activate' ? 'activates'
                          : arrow === 'inhibits' || arrow === 'inhibit' ? 'inhibits'
                          : 'binds';
    const isDarkMode = document.body.classList.contains('dark-mode');
    const arrowColors = isDarkMode ? {
      'activates': { bg: '#065f46', text: '#a7f3d0', border: '#047857', label: 'ACTIVATES' },
      'inhibits': { bg: '#991b1b', text: '#fecaca', border: '#b91c1c', label: 'INHIBITS' },
      'binds': { bg: '#5b21b6', text: '#ddd6fe', border: '#6d28d9', label: 'BINDS' }
    } : {
      'activates': { bg: '#d1fae5', text: '#047857', border: '#059669', label: 'ACTIVATES' },
      'inhibits': { bg: '#fee2e2', text: '#b91c1c', border: '#dc2626', label: 'INHIBITS' },
      'binds': { bg: '#ede9fe', text: '#6d28d9', border: '#7c3aed', label: 'BINDS' }
    };
    const colors = arrowColors[normalizedArrow];

    // Functions
    function deduplicateFunctions(functionArray) {
      const seen = new Set();
      return functionArray.filter(fn => {
        const key = `${fn.function || ''}|${fn.arrow || ''}|${fn.cellular_process || ''}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    }

    const rawFunctions = Array.isArray(L.functions) ? L.functions : [];
    const functions = deduplicateFunctions(rawFunctions);

    let functionsHTML = '';
    if (functions.length > 0) {
      functionsHTML = functions.map(fn => {
        // Pass SNAP.main and interactorName to ensure correct direction resolution
        return renderExpandableFunction(fn, renderArg1, renderArg2, arrow, direction);
      }).join('');
    } else {
      const emptyMessage = sectionType === 'shared'
        ? 'Shared interactions may not include context-specific functions.'
        : 'No functions associated with this interaction.';
      functionsHTML = `
        <div style="padding: var(--space-4); color: var(--color-text-secondary); font-style: italic;">
          ${emptyMessage}
        </div>
      `;
    }

    return `
      <div class="interaction-expandable-row" style="margin-bottom: 16px; border: 1px solid var(--color-border); border-radius: 8px; overflow: hidden; transition: all 0.2s ease;">
        <div class="interaction-row-header" style="padding: 12px 16px; background: var(--color-bg-secondary); display: flex; align-items: center; justify-content: space-between; gap: 12px; cursor: pointer; transition: background 0.2s;">
          <div style="display: flex; align-items: center; gap: 12px;">
            <div class="interaction-expand-icon" style="font-size: 12px; color: var(--color-text-secondary); width: 20px; transition: transform 0.2s;">▼</div>
            <span style="font-weight: 600; font-size: 14px;">${interactionTitle}</span>
            ${typeBadgeHTML}
            <span class="interaction-type-badge" style="display: inline-block; padding: 2px 8px; background: ${colors.bg}; color: ${colors.text}; border: 1px solid ${colors.border}; border-radius: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px;">
              ${colors.label}
            </span>
          </div>
        </div>
        <div class="interaction-expanded-content" style="max-height: 0; opacity: 0; overflow: hidden; transition: max-height 0.3s ease, opacity 0.2s ease;">
          <div style="padding: 16px; border-top: 1px solid var(--color-border);">
            ${L.support_summary ? `
              <div style="margin-bottom: 16px;">
                <div class="modal-detail-label">SUMMARY</div>
                <div class="modal-detail-value">${escapeHtml(L.support_summary)}</div>
              </div>
            ` : ''}
            <div class="modal-functions-header" style="font-size: 16px; margin-bottom: 12px;">Biological Functions (${functions.length})</div>
            ${functionsHTML}
          </div>
        </div>
      </div>
    `;
  }

  // CRITICAL FIX (Issue #6): Enhanced section headers for visual distinction
  // Render all sections with prominent, color-coded headers
  if (directLinks.length > 0) {
    sectionsHTML += `<div class="modal-section-divider" style="margin: 24px 0 16px 0; padding: 12px 16px; background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%); border-left: 6px solid #3b82f6; border-radius: 8px; box-shadow: 0 2px 4px rgba(59,130,246,0.1);">
      <h3 style="margin: 0; font-size: 16px; font-weight: 700; color: #1e40af; text-transform: uppercase; letter-spacing: 1px; display: flex; align-items: center; gap: 8px;">
        <span style="display: inline-block; width: 8px; height: 8px; background: #3b82f6; border-radius: 50%;"></span>
        DIRECT INTERACTIONS (${directLinks.length})
      </h3>
    </div>`;
    directLinks.forEach(link => {
      sectionsHTML += renderInteractionSection(link, 'direct');
    });
  }

  if (indirectLinks.length > 0) {
    sectionsHTML += `<div class="modal-section-divider" style="margin: 24px 0 16px 0; padding: 12px 16px; background: linear-gradient(135deg, #fef3c7 0%, #fed7aa 100%); border-left: 6px solid #f59e0b; border-radius: 8px; box-shadow: 0 2px 4px rgba(245,158,11,0.1);">
      <h3 style="margin: 0; font-size: 16px; font-weight: 700; color: #92400e; text-transform: uppercase; letter-spacing: 1px; display: flex; align-items: center; gap: 8px;">
        <span style="display: inline-block; width: 8px; height: 8px; background: #f59e0b; border-radius: 50%;"></span>
        INDIRECT INTERACTIONS (${indirectLinks.length})
      </h3>
    </div>`;
    indirectLinks.forEach(link => {
      sectionsHTML += renderInteractionSection(link, 'indirect');
    });
  }

  if (sharedLinks.length > 0) {
    sectionsHTML += `<div class="modal-section-divider" style="margin: 24px 0 16px 0; padding: 12px 16px; background: linear-gradient(135deg, #f3e8ff 0%, #fae8ff 100%); border-left: 6px solid #9333ea; border-radius: 8px; box-shadow: 0 2px 4px rgba(147,51,234,0.1);">
      <h3 style="margin: 0; font-size: 16px; font-weight: 700; color: #581c87; text-transform: uppercase; letter-spacing: 1px; display: flex; align-items: center; gap: 8px;">
        <span style="display: inline-block; width: 8px; height: 8px; background: #9333ea; border-radius: 50%;"></span>
        SHARED INTERACTIONS (${sharedLinks.length})
      </h3>
    </div>`;
    sharedLinks.forEach(link => {
      sectionsHTML += renderInteractionSection(link, 'shared');
    });
  }

  // Expand/collapse footer
  const isMainProtein = nodeId === SNAP.main;
  const isExpanded = expanded.has(nodeId);
  const canExpand = (depthMap.get(nodeId) ?? 1) < MAX_DEPTH;
  const hasInteractions = nodeLinks.length > 0;

  let footerHTML = '';
  if (isMainProtein) {
    // Main protein: show single "Find New Interactions" button
    footerHTML = `
      <div class="modal-footer" style="border-top: 1px solid var(--color-border); padding: 16px; background: var(--color-bg-secondary);">
        <button onclick="handleQueryFromModal('${nodeId}')" class="btn-primary" style="padding: 8px 20px; background: #10b981; color: white; border: none; border-radius: 6px; font-weight: 500; cursor: pointer; font-size: 14px; font-family: var(--font-sans); transition: background 0.2s;">
          Find New Interactions
        </button>
      </div>
    `;
  } else {
    // Interactor: show conditional Expand + Query buttons
    footerHTML = `
      <div class="modal-footer" style="border-top: 1px solid var(--color-border); padding: 16px; background: var(--color-bg-secondary);">
        <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
          ${canExpand && !isExpanded && hasInteractions ? `
            <button onclick="handleExpandFromModal('${nodeId}')" class="btn-primary" style="padding: 8px 20px; background: #3b82f6; color: white; border: none; border-radius: 6px; font-weight: 500; cursor: pointer; font-size: 14px; font-family: var(--font-sans); transition: background 0.2s;">
              Expand
            </button>
          ` : ''}
          ${canExpand && !isExpanded && !hasInteractions ? `
            <button disabled style="padding: 8px 20px; background: #d1d5db; color: #6b7280; border: none; border-radius: 6px; font-weight: 500; font-size: 14px; cursor: not-allowed; font-family: var(--font-sans);">
              Expand (No data)
            </button>
          ` : ''}
          ${isExpanded ? `
            <button onclick="handleCollapseFromModal('${nodeId}')" class="btn-secondary" style="padding: 8px 20px; background: #ef4444; color: white; border: none; border-radius: 6px; font-weight: 500; cursor: pointer; font-size: 14px; font-family: var(--font-sans); transition: background 0.2s;">
              Collapse
            </button>
          ` : ''}
          <button onclick="handleQueryFromModal('${nodeId}')" class="btn-primary" style="padding: 8px 20px; background: #10b981; color: white; border: none; border-radius: 6px; font-weight: 500; cursor: pointer; font-size: 14px; font-family: var(--font-sans); transition: background 0.2s;">
            Query
          </button>
          ${!canExpand && !isExpanded ? `
            <div style="padding: 8px 20px; background: #f3f4f6; color: #6b7280; border-radius: 6px; font-size: 13px; font-family: var(--font-sans); font-style: italic;">
              Max depth reached (${MAX_DEPTH})
            </div>
          ` : ''}
        </div>
        <div style="margin-top: 12px; font-size: 12px; color: var(--color-text-secondary); font-family: var(--font-sans);">
          Expand uses existing data • Query finds new interactions
        </div>
      </div>
    `;
  }

  const modalTitle = `${escapeHtml(nodeLabel)} - All Interactions (${nodeLinks.length})`;
  const modalContent = sectionsHTML + footerHTML;

  openModal(modalTitle, modalContent);
}

/* Helper functions for expand/collapse from modal */
function handleExpandFromModal(proteinId){
  closeModal();
  const node = nodeMap.get(proteinId); // PERFORMANCE: O(1) lookup
  if (node) {
    expandInteractor(node);
  }
}

function handleCollapseFromModal(proteinId){
  closeModal();
  collapseInteractor(proteinId);
}

async function handleQueryFromModal(proteinId) {
  closeModal();

  // Get configuration from localStorage
  const queryConfig = {
    interactor_rounds: parseInt(localStorage.getItem('interactor_rounds')) || 3,
    function_rounds: parseInt(localStorage.getItem('function_rounds')) || 3,
    max_depth: parseInt(localStorage.getItem('max_depth')) || 3,
    skip_validation: localStorage.getItem('skip_validation') === 'true',
    skip_deduplicator: localStorage.getItem('skip_deduplicator') === 'true',
    skip_arrow_determination: localStorage.getItem('skip_arrow_determination') === 'true'
  };

  try {
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        protein: proteinId,
        ...queryConfig
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      showNotificationMessage(`<span style="color: #ef4444;">Query failed: ${errorData.error || 'Unknown error'}</span>`);
      return;
    }

    const data = await response.json();

    if (data.status === 'processing') {
      // Add job to tracker with reload callback
      vizJobTracker.addJob(proteinId, {
        ...queryConfig,
        onComplete: () => {
          // Reload page to show updated data
          vizJobTracker.saveToSessionStorage(); // Persist jobs before reload
          window.location.reload();
        }
      });
    } else if (data.status === 'complete') {
      // Already complete - reload immediately
      showNotificationMessage(`<span>Query complete! Reloading...</span>`);
      vizJobTracker.saveToSessionStorage(); // Persist jobs before reload
      setTimeout(() => { window.location.reload(); }, 500);
    } else {
      showNotificationMessage(`<span style="color: #ef4444;">Unexpected status: ${data.status}</span>`);
    }
  } catch (error) {
    console.error('[ERROR] Query from modal failed:', error);
    showNotificationMessage(`<span style="color: #ef4444;">Failed to start query</span>`);
  }
}

// Search protein from visualizer page
async function searchProteinFromVisualizer(proteinName) {
  showNotificationMessage(`<span>Searching for ${proteinName}...</span>`);

  try {
    const response = await fetch(`/api/search/${encodeURIComponent(proteinName)}`);

    if (!response.ok) {
      const errorData = await response.json();
      showNotificationMessage(`<span style="color: #ef4444;">${errorData.error || 'Search failed'}</span>`);
      return;
    }

    const data = await response.json();

    if (data.status === 'found') {
      // Protein exists - navigate to it
      showNotificationMessage(`<span>Found! Loading ${proteinName}...</span>`);
      setTimeout(() => {
        window.location.href = `/api/visualize/${encodeURIComponent(proteinName)}?t=${Date.now()}`;
      }, 500);
    } else {
      // Not found - show query prompt
      showNotificationMessage(`<span>${proteinName} not found. <button onclick="startQueryFromVisualizer('${proteinName}')" style="padding: 4px 12px; background: #3b82f6; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; margin-left: 8px;">Start Query</button></span>`);
    }
  } catch (error) {
    console.error('[ERROR] Search failed:', error);
    showNotificationMessage(`<span style="color: #ef4444;">Search failed</span>`);
  }
}

// Start query from visualizer page
async function startQueryFromVisualizer(proteinName) {
  // IMMEDIATELY hide notification message when starting query
  const msg = document.getElementById('notification-message');
  if (msg) {
    msg.style.display = 'none';
    msg.innerHTML = '';
  }

  const queryConfig = {
    interactor_rounds: parseInt(localStorage.getItem('interactor_rounds')) || 3,
    function_rounds: parseInt(localStorage.getItem('function_rounds')) || 3,
    max_depth: parseInt(localStorage.getItem('max_depth')) || 3,
    skip_validation: localStorage.getItem('skip_validation') === 'true',
    skip_deduplicator: localStorage.getItem('skip_deduplicator') === 'true',
    skip_arrow_determination: localStorage.getItem('skip_arrow_determination') === 'true'
  };

  try {
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        protein: proteinName,
        ...queryConfig
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      showNotificationMessage(`<span style="color: #ef4444;">Query failed: ${errorData.error || 'Unknown error'}</span>`);
      return;
    }

    const data = await response.json();

    if (data.status === 'processing') {
      // Add job to tracker with completion callback
      vizJobTracker.addJob(proteinName, {
        ...queryConfig,
        onComplete: () => {
          // Navigate to visualization
          vizJobTracker.saveToSessionStorage(); // Persist jobs before navigation
          window.location.href = `/api/visualize/${encodeURIComponent(proteinName)}?t=${Date.now()}`;
        }
      });
    } else if (data.status === 'complete') {
      // Already complete - navigate immediately
      showNotificationMessage(`<span>Query complete! Loading visualization...</span>`);
      vizJobTracker.saveToSessionStorage(); // Persist jobs before navigation
      setTimeout(() => {
        window.location.href = `/api/visualize/${encodeURIComponent(proteinName)}?t=${Date.now()}`;
      }, 500);
    } else {
      showNotificationMessage(`<span style="color: #ef4444;">Unexpected status: ${data.status}</span>`);
    }
  } catch (error) {
    console.error('[ERROR] Query failed:', error);
    showNotificationMessage(`<span style="color: #ef4444;">Failed to start query</span>`);
  }
}

function showFunctionModalFromNode(fnNode){
  // Find the corresponding link to get the normalized arrow
  const linkId = `${fnNode.parent}-${fnNode.id}`;
  const correspondingLink = links.find(l => l.id === linkId);

  // Leverage the same renderer as link, but pass the fields explicitly
  showFunctionModal({
    fn: fnNode.data,
    interactor: fnNode.interactorData,
    affected: fnNode.parent,
    label: fnNode.label,
    linkArrow: correspondingLink ? correspondingLink.arrow : undefined
  });
}

/* Function modal (from function link click) */
function showFunctionModalFromLink(link){
  const payload = link.data || {};
  showFunctionModal({
    fn: payload.fn || {},
    interactor: payload.interactor || {},
    affected: (payload.interactor && payload.interactor.primary) || '—',
    label: (payload.fn && payload.fn.function) || 'Function',
    linkArrow: link.arrow  // Pass the link's already-normalized arrow
  });
}

/* Render function modal (interactor → fn) */
function showFunctionModal({ fn, interactor, affected, label, linkArrow }){

  // Format references with full paper details from evidence using beautiful wrappers
  const evs = Array.isArray(fn.evidence) ? fn.evidence : [];
  const evHTML = evs.length ? `<div class="expanded-evidence-list">${evs.map(ev=>{
    const primaryLink = ev.pmid ? `https://pubmed.ncbi.nlm.nih.gov/${ev.pmid}` : (ev.doi ? `https://doi.org/${ev.doi}` : null);
    return `<div class="expanded-evidence-wrapper">
      <div class="expanded-evidence-card" data-evidence-link="${primaryLink || ''}" data-has-link="${primaryLink ? 'true' : 'false'}">
        <div class="expanded-evidence-title">${ev.paper_title || 'Title not available'}</div>
        <div class="expanded-evidence-meta">
          ${ev.authors ? `<div class="expanded-evidence-meta-item"><strong>Authors:</strong> ${ev.authors}</div>` : ''}
          ${ev.journal ? `<div class="expanded-evidence-meta-item"><strong>Journal:</strong> ${ev.journal}</div>` : ''}
          ${ev.year ? `<div class="expanded-evidence-meta-item"><strong>Year:</strong> ${ev.year}</div>` : ''}
        </div>
        ${ev.relevant_quote ? `<div class="expanded-evidence-quote">"${ev.relevant_quote}"</div>` : ''}
        <div class="expanded-evidence-pmids" style="margin-top:8px;">
          ${ev.pmid ? `<a href="https://pubmed.ncbi.nlm.nih.gov/${ev.pmid}" target="_blank" class="expanded-pmid-badge" onclick="event.stopPropagation();">PMID: ${ev.pmid}</a>` : ''}
          ${ev.doi ? `<a href="https://doi.org/${ev.doi}" target="_blank" class="expanded-pmid-badge" onclick="event.stopPropagation();">DOI: ${ev.doi}</a>` : ''}
        </div>
      </div>
    </div>`;
  }).join('')}</div>` : (Array.isArray(fn.pmids) && fn.pmids.length
      ? fn.pmids.map(p=> `<a class="pmid-link" target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/${p}">PMID: ${p}</a>`).join(', ')
      : '<div class="expanded-empty">No references available</div>');

  // Format specific effects with 3D wrappers
  let effectsHTML = '';
  if (Array.isArray(fn.specific_effects) && fn.specific_effects.length) {
    const effectChips = fn.specific_effects.map(s=>`
      <div class="expanded-effect-chip-wrapper">
        <div class="expanded-effect-chip">${s}</div>
      </div>`).join('');
    effectsHTML = `
      <tr class="info-row">
        <td class="info-label">SPECIFIC EFFECTS</td>
        <td class="info-value">
          <div class="expanded-effects-grid">${effectChips}</div>
        </td>
      </tr>`;
  }

  // Format biological cascade - NORMALIZED VERTICAL FLOWCHART
  const createCascadeHTML = (value) => {
    const segments = Array.isArray(value) ? value : (value ? [value] : []);
    if (segments.length === 0) {
      return '<div class="expanded-empty">Cascading biological effects not specified</div>';
    }

    // Normalize: flatten all segments and split by arrow (→)
    const allSteps = [];
    segments.forEach(segment => {
      const text = (segment == null ? '' : segment).toString().trim();
      if (!text) return;

      // Split by arrow and clean each step
      const steps = text.split('→').map(s => s.trim()).filter(s => s.length > 0);
      allSteps.push(...steps);
    });

    if (allSteps.length === 0) {
      return '<div class="expanded-empty">Cascading biological effects not specified</div>';
    }

    // Create vertical flowchart blocks
    const items = allSteps.map(step =>
      `<div class="cascade-flow-item">${escapeHtml(step)}</div>`
    ).join('');

    return `<div class="cascade-wrapper"><div class="cascade-flow-container">${items}</div></div>`;
  };
  const biologicalConsequenceHTML = createCascadeHTML(fn.biological_consequence);

  const mechanism = interactor && interactor.intent ? (interactor.intent[0].toUpperCase()+interactor.intent.slice(1)) : 'Not specified';

  // EFFECT TYPE: Use the link's already-normalized arrow
  // The link was created with the normalized arrow, so we MUST use that for consistency
  const normalizedArrow = linkArrow || 'binds';  // Default to binds if no link arrow provided
  const arrowColor = normalizedArrow === 'activates' ? '#059669' : (normalizedArrow === 'inhibits' ? '#dc2626' : '#7c3aed');
  const arrowStr = fn.effect_description ?
    `<strong style="color:${arrowColor};">${fn.effect_description}</strong>` :
    (normalizedArrow === 'activates' ?
      '<strong style="color:#059669;">✓ Function is enhanced or activated</strong>' :
      (normalizedArrow === 'inhibits' ?
        '<strong style="color:#dc2626;">✗ Function is inhibited or disrupted</strong>' :
        '<strong style="color:#7c3aed;">⊕ Binds/Interacts</strong>'));

  // Check for validity field (from fact-checker)
  const validity = fn.validity || 'TRUE';
  const validationNote = fn.validation_note || '';
  const isConflicting = validity === 'CONFLICTING';
  const isFalse = validity === 'FALSE';

  // Build conflict warning HTML if needed
  let conflictWarningHTML = '';
  if (isConflicting || isFalse) {
    const warningType = isFalse ? 'Invalid Claim' : 'Conflicting Evidence';
    const warningIcon = isFalse ? '❌' : '⚠️';
    const warningColor = isFalse ? '#dc2626' : '#f59e0b';
    conflictWarningHTML = `
      <tr class="info-row">
        <td colspan="2">
          <div style="background:${isFalse ? '#fee2e2' : '#fff3cd'};border-left:4px solid ${warningColor};padding:12px 16px;margin:8px 0;border-radius:4px;">
            <div style="font-weight:600;color:${warningColor};margin-bottom:4px;">
              ${warningIcon} <strong>${warningType}</strong>
            </div>
            <div style="color:#374151;font-size:13px;">${validationNote}</div>
          </div>
        </td>
      </tr>`;
  }

  // Update function label to show asterisk for conflicting claims
  const functionLabel = isConflicting ? `⚠ ${label} *` : label;

  // Wrap mechanism with beautiful wrapper
  const mechanismHTML = mechanism !== 'Not specified'
    ? `<div class="expanded-mechanism-wrapper"><span class="mechanism-badge">${mechanism}</span></div>`
    : '<span class="muted-text">Not specified</span>';

  // Wrap cellular process with beautiful wrapper
  const cellularHTML = fn.cellular_process
    ? `<div class="expanded-cellular-wrapper"><div class="expanded-cellular-process"><div class="expanded-cellular-process-text">${fn.cellular_process}</div></div></div>`
    : '<div class="expanded-empty">Molecular mechanism not specified</div>';

  // Wrap effect type with beautiful wrapper
  const effectTypeColor = normalizedArrow === 'activates' ? 'activates' : (normalizedArrow === 'inhibits' ? 'inhibits' : 'binds');
  const effectTypeText = fn.effect_description || (normalizedArrow === 'activates' ? '✓ Function is enhanced or activated' : (normalizedArrow === 'inhibits' ? '✗ Function is inhibited or disrupted' : '⊕ Binds/Interacts'));
  const effectTypeHTML = `<div class="expanded-effect-type ${effectTypeColor}"><span class="effect-type-badge ${effectTypeColor}">${effectTypeText}</span></div>`;

  // Wrap function and protein names prominently
  const functionHTML = `<div class="function-name-wrapper ${effectTypeColor}"><span class="function-name ${effectTypeColor}" style="font-size: 18px;">${functionLabel}</span></div>`;
  const affectedHTML = `<div class="interaction-name-wrapper"><div class="interaction-name" style="font-size: 16px;">${affected}</div></div>`;

  const body = `
    <table class="info-table">
      ${conflictWarningHTML}
      <tr class="info-row"><td class="info-label">FUNCTION</td><td class="info-value">${functionHTML}</td></tr>
      <tr class="info-row"><td class="info-label">AFFECTED PROTEIN</td><td class="info-value">${affectedHTML}</td></tr>
      <tr class="info-row"><td class="info-label">EFFECT TYPE</td><td class="info-value">${effectTypeHTML}</td></tr>
      <tr class="info-row"><td class="info-label">MECHANISM</td><td class="info-value">${mechanismHTML}</td></tr>
      <tr class="info-row"><td class="info-label">CELLULAR PROCESS</td><td class="info-value">${cellularHTML}</td></tr>
      <tr class="info-row"><td class="info-label">BIOLOGICAL CASCADE</td><td class="info-value">${biologicalConsequenceHTML}</td></tr>
      ${effectsHTML}
      <tr class="info-row"><td class="info-label">REFERENCES</td><td class="info-value">${evHTML}</td></tr>
    </table>`;
  openModal(`Function: ${label}`, body);
}

/* ===== Progress helpers (viz page) ===== */
// Custom error for cancellations (to distinguish from other errors)
class CancellationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'CancellationError';
  }
}

// ============================================================================
// UTILITY FUNCTIONS - Fetch with timeout and retry
// ============================================================================

/**
 * Fetch with timeout to prevent hanging requests
 * FIXED: Added 30s timeout for all HTTP requests
 */
async function fetchWithTimeout(url, options = {}, timeout = 30000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    if (error.name === 'AbortError') {
      throw new Error('Request timeout');
    }
    throw error;
  }
}

/**
 * Fetch with exponential backoff retry
 * FIXED: Added retry logic for failed status checks
 */
async function fetchWithRetry(url, options = {}, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetchWithTimeout(url, options);
      return response;
    } catch (error) {
      if (i === maxRetries - 1) throw error;

      // Exponential backoff: 1s, 2s, 4s
      const delay = 1000 * Math.pow(2, i);
      console.log(`[Fetch] Retry ${i + 1}/${maxRetries} after ${delay}ms for ${url}`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

// ============================================================================
// FUNCTIONAL CORE - Pure State Management (No Side Effects)
// ============================================================================

/**
 * Calculate percentage from current/total progress
 * @pure
 */
function calculateJobPercent(current, total) {
  if (typeof current !== 'number' || typeof total !== 'number') return 0;
  if (total <= 0) return 0;
  if (current >= total) return 100;
  return Math.round((current / total) * 100);
}

/**
 * Format job status into display metadata
 * @pure
 */
function formatVizJobStatus(status) {
  const statusMap = {
    processing: { color: '#3b82f6', icon: '⏳', text: 'Running' },
    complete: { color: '#10b981', icon: '✓', text: 'Complete' },
    error: { color: '#ef4444', icon: '✕', text: 'Failed' },
    cancelled: { color: '#6b7280', icon: '⊘', text: 'Cancelled' }
  };
  return statusMap[status] || statusMap.processing;
}

/**
 * Create new job state object
 * @pure
 */
function createVizJobState(protein, config = {}) {
  return {
    protein,
    status: 'processing',
    progress: {
      current: 0,
      total: 100,
      text: 'Initializing...'
    },
    config,
    startTime: Date.now()
  };
}

/**
 * Update job progress (returns new object)
 * @pure
 */
function updateVizJobProgress(job, progressData) {
  return {
    ...job,
    progress: {
      current: progressData.current || job.progress.current,
      total: progressData.total || job.progress.total,
      text: progressData.text || job.progress.text
    }
  };
}

/**
 * Mark job as complete (returns new object)
 * @pure
 */
function markVizJobComplete(job) {
  return {
    ...job,
    status: 'complete',
    progress: {
      current: 100,
      total: 100,
      text: 'Complete!'
    }
  };
}

/**
 * Mark job as error (returns new object)
 * @pure
 */
function markVizJobError(job, errorText) {
  return {
    ...job,
    status: 'error',
    progress: {
      ...job.progress,
      text: errorText || 'Error occurred'
    }
  };
}

/**
 * Mark job as cancelled (returns new object)
 * @pure
 */
function markVizJobCancelled(job) {
  return {
    ...job,
    status: 'cancelled',
    progress: {
      ...job.progress,
      text: 'Cancelled by user'
    }
  };
}

// ============================================================================
// IMPERATIVE SHELL - DOM Manipulation (Thin I/O Layer)
// ============================================================================

/**
 * Create a mini job card DOM element (for viz page header)
 * Compact chip layout: NAME - XX% [=====___] [−][×]
 * @returns {Object} { container, bar, text, percent, removeBtn, cancelBtn }
 */
function createMiniJobCard(protein) {
  const container = document.createElement('div');
  container.className = 'mini-job-card';
  container.id = `mini-job-${protein}`;

  container.innerHTML = `
    <span class="mini-job-protein">${protein}</span>
    <span class="mini-job-separator">−</span>
    <span class="mini-job-progress-percent">0%</span>
    <div class="mini-job-progress-bar-outer">
      <div class="mini-job-progress-bar-inner"></div>
    </div>
    <div class="mini-job-actions">
      <button class="mini-job-btn mini-job-remove" title="Remove from tracker (job continues in background)" aria-label="Remove from tracker">
        <span class="mini-job-btn-icon">−</span>
      </button>
      <button class="mini-job-btn mini-job-cancel" title="Cancel job" aria-label="Cancel job">
        <span class="mini-job-btn-icon">✕</span>
      </button>
    </div>
  `;

  return {
    container,
    bar: container.querySelector('.mini-job-progress-bar-inner'),
    text: null, // Not used in compact layout
    percent: container.querySelector('.mini-job-progress-percent'),
    removeBtn: container.querySelector('.mini-job-remove'),
    cancelBtn: container.querySelector('.mini-job-cancel')
  };
}

/**
 * Update mini job card UI with current job state
 */
function updateMiniJobCard(elements, job) {
  if (!elements || !job) return;

  const { bar, text, percent, container } = elements;
  const progressPercent = calculateJobPercent(job.progress.current, job.progress.total);
  const statusInfo = formatVizJobStatus(job.status);

  // Update progress bar
  if (bar) {
    bar.style.width = `${progressPercent}%`;
    bar.style.backgroundColor = statusInfo.color;
  }

  // Update text
  if (text) {
    if (job.progress.current && job.progress.total) {
      text.textContent = `${job.protein}: Step ${job.progress.current}/${job.progress.total}`;
    } else {
      text.textContent = `${job.protein}: ${job.progress.text}`;
    }
  }

  // Update percent
  if (percent) {
    percent.textContent = `${progressPercent}%`;
  }

  // Update container state
  if (container) {
    container.setAttribute('data-status', job.status);
  }
}

/**
 * Remove mini job card from DOM with fade animation
 */
function removeMiniJobCard(container, callback) {
  if (!container) {
    if (callback) callback();
    return;
  }

  container.style.opacity = '0';
  container.style.transform = 'translateY(-10px)';

  setTimeout(() => {
    if (container.parentNode) {
      container.parentNode.removeChild(container);
    }
    if (callback) callback();
  }, 300);
}

// ============================================================================
// VIZ JOB TRACKER - Multi-Job Orchestration for Visualization Page
// ============================================================================

class VizJobTracker {
  constructor(containerId) {
    this.jobs = new Map();           // protein -> job state
    this.intervals = new Map();      // protein -> intervalId
    this.uiElements = new Map();     // protein -> DOM elements
    this.container = document.getElementById(containerId);
    this._isRestoring = false;       // FIXED: Guard against parallel restores

    if (!this.container) {
      console.warn(`[VizJobTracker] Container #${containerId} not found. Creating fallback.`);
      this._createFallbackContainer();
    }
  }

  /**
   * Create fallback container if none exists
   */
  _createFallbackContainer() {
    const notification = document.getElementById('job-notification');
    if (notification) {
      const container = document.createElement('div');
      container.id = 'mini-job-container';
      container.className = 'mini-job-container';
      notification.insertBefore(container, notification.firstChild);
      this.container = container;
    }
  }

  /**
   * Add a new job to tracker and start polling
   */
  addJob(protein, config = {}) {
    // Guard: prevent duplicate jobs
    if (this.jobs.has(protein)) {
      const existingJob = this.jobs.get(protein);
      if (existingJob.status === 'processing') {
        console.warn(`[VizJobTracker] Job for ${protein} already running`);

        // Show user-friendly warning
        const confirmed = confirm(
          `A query for ${protein} is already running.\n\nCancel the existing job and start a new one?`
        );

        if (confirmed) {
          this.cancelJob(protein);
          // Wait a moment for cleanup
          setTimeout(() => this._addJobInternal(protein, config), 500);
        }
        return;
      }
    }

    this._addJobInternal(protein, config);
  }

  /**
   * Internal method to add job (separated for recursion after cancel)
   */
  _addJobInternal(protein, config) {
    // Create job state
    const job = createVizJobState(protein, config);
    this.jobs.set(protein, job);

    // Show header when first job starts
    showHeader();

    // Render UI
    this._renderJob(protein);

    // Start polling
    this._startPolling(protein);

    console.log(`[VizJobTracker] Added job for ${protein}`);
  }

  /**
   * Remove job from tracker (UI only, job continues in background)
   */
  removeFromTracker(protein) {
    console.log(`[VizJobTracker] Removing ${protein} from tracker (job continues in background)`);

    // Stop polling
    this._stopPolling(protein);

    // Remove UI
    const elements = this.uiElements.get(protein);
    if (elements) {
      removeMiniJobCard(elements.container, () => {
        this.uiElements.delete(protein);
      });
    }

    // Remove from state
    this.jobs.delete(protein);

    // Hide header if no more jobs
    if (this.jobs.size === 0) {
      setTimeout(hideHeader, 500);
    }
  }

  /**
   * Cancel job (stops backend job + removes from tracker)
   * FIXED: Stop polling BEFORE cancel request to prevent race condition
   */
  async cancelJob(protein) {
    console.log(`[VizJobTracker] Cancelling job for ${protein}`);

    const job = this.jobs.get(protein);
    if (!job) {
      console.warn(`[VizJobTracker] No job found for ${protein}`);
      return;
    }

    // FIXED: Stop polling FIRST to prevent race with completion
    this._stopPolling(protein);

    // Disable cancel button to prevent double-clicks
    const elements = this.uiElements.get(protein);
    if (elements && elements.cancelBtn) {
      elements.cancelBtn.disabled = true;
    }

    try {
      // Send cancel request to backend
      const response = await fetch(`/api/cancel/${encodeURIComponent(protein)}`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error('Cancel request failed');
      }

      // Update state
      const cancelledJob = markVizJobCancelled(job);
      this.jobs.set(protein, cancelledJob);

      // Update UI
      this._updateJobUI(protein);

      // Remove after delay
      setTimeout(() => {
        this.removeFromTracker(protein);
      }, 2000);

    } catch (error) {
      console.error(`[VizJobTracker] Failed to cancel ${protein}:`, error);

      // Re-enable cancel button on error
      if (elements && elements.cancelBtn) {
        elements.cancelBtn.disabled = false;
      }

      // Show error in UI
      const errorJob = markVizJobError(job, 'Failed to cancel job');
      this.jobs.set(protein, errorJob);
      this._updateJobUI(protein);

      // Restart polling on error (cancel failed, job still running)
      this._startPolling(protein);
    }
  }

  /**
   * Update job progress
   */
  updateJob(protein, progressData) {
    const job = this.jobs.get(protein);
    if (!job) return;

    const updatedJob = updateVizJobProgress(job, progressData);
    this.jobs.set(protein, updatedJob);
    this._updateJobUI(protein);
  }

  /**
   * Mark job as complete (with custom callback)
   */
  completeJob(protein, onComplete) {
    const job = this.jobs.get(protein);
    if (!job) return;

    const completedJob = markVizJobComplete(job);
    this.jobs.set(protein, completedJob);
    this._updateJobUI(protein);
    this._stopPolling(protein);

    // Call custom completion callback
    if (onComplete) {
      setTimeout(() => {
        onComplete();
        this.removeFromTracker(protein);
      }, 1000);
    } else {
      // Default: auto-remove after delay
      setTimeout(() => {
        this.removeFromTracker(protein);
      }, 3000);
    }
  }

  /**
   * Mark job as error
   */
  errorJob(protein, errorText) {
    const job = this.jobs.get(protein);
    if (!job) return;

    const errorJob = markVizJobError(job, errorText);
    this.jobs.set(protein, errorJob);
    this._updateJobUI(protein);
    this._stopPolling(protein);

    // Auto-remove after delay
    setTimeout(() => {
      this.removeFromTracker(protein);
    }, 5000);
  }

  /**
   * Render mini job card in UI
   */
  _renderJob(protein) {
    if (!this.container) return;

    const job = this.jobs.get(protein);
    if (!job) return;

    // Create job card
    const elements = createMiniJobCard(protein);
    this.uiElements.set(protein, elements);

    // Wire up event listeners
    elements.removeBtn.onclick = () => this.removeFromTracker(protein);
    elements.cancelBtn.onclick = () => this.cancelJob(protein);

    // Add to DOM
    this.container.appendChild(elements.container);

    // Initial render
    this._updateJobUI(protein);

    // Trigger animation
    setTimeout(() => {
      elements.container.style.opacity = '1';
    }, 10);
  }

  /**
   * Update job UI from state
   */
  _updateJobUI(protein) {
    const job = this.jobs.get(protein);
    const elements = this.uiElements.get(protein);

    if (!job || !elements) return;

    updateMiniJobCard(elements, job);
  }

  /**
   * Start polling for job status
   * FIXED: Uses fetchWithRetry for resilience
   */
  _startPolling(protein) {
    const intervalId = setInterval(async () => {
      try {
        const response = await fetchWithRetry(`/api/status/${encodeURIComponent(protein)}`);

        if (!response.ok) {
          console.warn(`[VizJobTracker] Status check failed for ${protein}`);
          return;
        }

        const data = await response.json();
        const job = this.jobs.get(protein);

        if (!job) {
          // Job was removed, stop polling
          this._stopPolling(protein);
          return;
        }

        // Handle different statuses
        if (data.status === 'complete') {
          // Get custom completion callback from job config
          this.completeJob(protein, job.config.onComplete);
        } else if (data.status === 'cancelled' || data.status === 'cancelling') {
          const cancelledJob = markVizJobCancelled(job);
          this.jobs.set(protein, cancelledJob);
          this._updateJobUI(protein);
          this._stopPolling(protein);
          setTimeout(() => this.removeFromTracker(protein), 2000);
        } else if (data.status === 'error') {
          const errorText = typeof data.progress === 'object' ? data.progress.text : data.progress;
          this.errorJob(protein, errorText || 'Unknown error');
        } else if (data.progress) {
          // Processing - update progress
          this.updateJob(protein, data.progress);
        }

      } catch (error) {
        console.error(`[VizJobTracker] Polling error for ${protein}:`, error);
      }
    }, 5000); // FIXED: Standardized to 5s (was 4s)

    this.intervals.set(protein, intervalId);
  }

  /**
   * Stop polling for job
   */
  _stopPolling(protein) {
    const intervalId = this.intervals.get(protein);
    if (intervalId) {
      clearInterval(intervalId);
      this.intervals.delete(protein);
    }
  }

  /**
   * Get count of active jobs
   */
  getActiveJobCount() {
    return Array.from(this.jobs.values()).filter(
      job => job.status === 'processing'
    ).length;
  }

  /**
   * Save active jobs to sessionStorage for persistence across page navigations
   * FIXED: Merges with existing jobs to prevent multi-tab corruption
   */
  saveToSessionStorage() {
    // Read existing saved jobs from sessionStorage
    const existing = sessionStorage.getItem('vizActiveJobs');
    const existingJobs = existing ? JSON.parse(existing) : [];

    // Get current processing jobs
    const currentProteins = new Set();
    this.jobs.forEach((job, protein) => {
      if (job.status === 'processing') {
        currentProteins.add(protein);
      }
    });

    // Merge: Keep existing jobs not in current tab, add current tab's jobs
    const merged = existingJobs.filter(j => !currentProteins.has(j.protein));

    this.jobs.forEach((job, protein) => {
      if (job.status === 'processing') {
        merged.push({
          protein: protein,
          startTime: job.startTime,
          config: job.config || {}
        });
      }
    });

    sessionStorage.setItem('vizActiveJobs', JSON.stringify(merged));
    console.log(`[SessionStorage] Saved ${merged.length} active job(s) (merged from ${existingJobs.length} existing)`);
  }

  /**
   * Restore jobs from sessionStorage on page load
   * Only restores jobs that are still actually running on backend
   * FIXED: Guard against parallel restores
   */
  async restoreFromSessionStorage() {
    // Guard against parallel restores
    if (this._isRestoring) {
      console.log('[SessionStorage] Restore already in progress, skipping');
      return;
    }

    this._isRestoring = true;

    try {
      const saved = sessionStorage.getItem('vizActiveJobs');
      if (!saved) {
        console.log('[SessionStorage] No saved jobs found');
        return;
      }

      const savedJobs = JSON.parse(saved);
      const oneHourAgo = Date.now() - (60 * 60 * 1000);

      console.log(`[SessionStorage] Found ${savedJobs.length} saved job(s), checking status...`);

      let restoredCount = 0;

      for (const savedJob of savedJobs) {
        // Skip stale jobs (>1 hour old)
        if (savedJob.startTime < oneHourAgo) {
          console.log(`[SessionStorage] Skipping stale job: ${savedJob.protein} (${Math.round((Date.now() - savedJob.startTime) / 60000)}min old)`);
          continue;
        }

        // Check if job is still running
        try {
          const response = await fetchWithRetry(`/api/status/${encodeURIComponent(savedJob.protein)}`);
          if (!response.ok) {
            console.log(`[SessionStorage] Job ${savedJob.protein} not found on backend`);
            continue;
          }

          const data = await response.json();

          if (data.status === 'processing') {
            // FIXED: Check if already tracked (from auto-resume) to prevent duplicate dialog
            if (!this.jobs.has(savedJob.protein)) {
              console.log(`[SessionStorage] Restoring job: ${savedJob.protein}`);
              this.addJob(savedJob.protein, savedJob.config || {});
              restoredCount++;
            } else {
              console.log(`[SessionStorage] Skipping ${savedJob.protein} (already tracked)`);
            }
          } else {
            console.log(`[SessionStorage] Job ${savedJob.protein} no longer processing (status: ${data.status})`);
          }
        } catch (error) {
          console.log(`[SessionStorage] Failed to check job ${savedJob.protein}:`, error.message);
        }
      }

      console.log(`[SessionStorage] Restored ${restoredCount} active job(s)`);

      // FIXED: Clean up sessionStorage to only keep currently active jobs
      const activeJobs = [];
      this.jobs.forEach((job, protein) => {
        if (job.status === 'processing') {
          activeJobs.push({
            protein: protein,
            startTime: job.startTime,
            config: job.config || {}
          });
        }
      });
      sessionStorage.setItem('vizActiveJobs', JSON.stringify(activeJobs));
      console.log(`[SessionStorage] Cleaned up, ${activeJobs.length} active jobs remain`);

    } catch (error) {
      console.error('[SessionStorage] Restore failed:', error);
    } finally {
      this._isRestoring = false;
    }
  }
}

// Initialize global job tracker for viz page
const vizJobTracker = new VizJobTracker('mini-job-container');

function showHeader(){
  const header = document.querySelector('.header');
  if (header) header.classList.add('header-visible');
}
function hideHeader(){
  const header = document.querySelector('.header');
  if (header) header.classList.remove('header-visible');
}

/**
 * Show notification message in header (for non-job messages)
 */
function showNotificationMessage(html) {
  const msg = document.getElementById('notification-message');
  if (msg) {
    msg.innerHTML = html;
    msg.style.display = 'block';
    showHeader();
    // Auto-hide after 5 seconds
    setTimeout(() => {
      msg.style.display = 'none';
      if (vizJobTracker.getActiveJobCount() === 0) {
        hideHeader();
      }
    }, 5000);
  }
}

/**
 * Show query prompt for protein not found in database
 * Matches index page behavior - gives user option to start query
 */
function showQueryPromptViz(proteinName) {
  const message = `
    <div style="text-align: center; padding: 12px;">
      <p style="font-size: 14px; color: #6b7280; margin-bottom: 12px;">
        Protein <strong>${proteinName}</strong> not found in database.
      </p>
      <button onclick="startQueryFromVisualizer('${proteinName}')"
              style="padding: 8px 16px; background: #3b82f6; color: white;
                     border: none; border-radius: 6px; font-weight: 500;
                     cursor: pointer; font-size: 13px;">
        Start Research Query
      </button>
    </div>
  `;
  showNotificationMessage(message);
}

function miniProgress(text, current, total, proteinName){
  const wrap = document.getElementById('mini-progress-wrapper');
  const bar  = document.getElementById('mini-progress-bar-inner');
  const txt  = document.getElementById('mini-progress-text');
  const msg  = document.getElementById('notification-message');
  const cancelBtn = document.getElementById('mini-cancel-btn');

  if (msg) msg.innerHTML = '';

  // FALLBACK: If old elements don't exist, use new tracker system
  if (!wrap || !bar || !txt) {
    if (proteinName) {
      // Use new job tracker if tracking a specific protein
      const existingJob = vizJobTracker.jobs.get(proteinName);
      if (!existingJob) {
        // Auto-create job in tracker if it doesn't exist
        vizJobTracker.addJob(proteinName, {});
      }
      // Update progress
      if (typeof current === 'number' && typeof total === 'number') {
        vizJobTracker.updateJob(proteinName, { current, total, text: text || 'Processing' });
      }
    } else {
      // Show as notification for non-protein-specific messages
      showNotificationMessage(`<span>${text || 'Processing...'}</span>`);
    }
    return;
  }

  // OLD CODE PATH: Use old elements if they exist
  showHeader();
  wrap.style.display = 'grid';

  // Track current job
  if (proteinName) {
    currentJobProtein = proteinName;
    currentRunningJob = proteinName;  // Keep both variables in sync
    // Show cancel button for all jobs
    if (cancelBtn) {
      cancelBtn.style.display = 'inline-block';
      cancelBtn.disabled = false;  // Re-enable in case it was disabled
    }
  }

  if (typeof current==='number' && typeof total==='number' && total>0){
    const pct = Math.max(0, Math.min(100, Math.round((current/total)*100)));
    bar.style.width = pct+'%';
    // Simplified format for visualization page: just protein name and percentage
    if (proteinName) {
      txt.textContent = `${proteinName}: ${pct}%`;
    } else {
      txt.textContent = `${text||'Processing…'} (${pct}%)`;
    }
  } else {
    bar.style.width = '25%';
    // When no progress numbers available, show protein name with status
    if (proteinName) {
      txt.textContent = `${proteinName}: ${text || 'Processing…'}`;
    } else {
      txt.textContent = text || 'Processing…';
    }
  }
}

function miniDone(html){
  const wrap = document.getElementById('mini-progress-wrapper');
  const bar  = document.getElementById('mini-progress-bar-inner');
  const msg  = document.getElementById('notification-message');
  const cancelBtn = document.getElementById('mini-cancel-btn');

  // FALLBACK: If old elements don't exist, use new notification system
  if (!wrap || !bar) {
    if (html) {
      showNotificationMessage(html);
    }
    currentJobProtein = null;
    currentRunningJob = null;
    return;
  }

  // OLD CODE PATH: Use old elements if they exist
  if (wrap) wrap.style.display='none';
  if (bar) bar.style.width='0%';
  if (cancelBtn) cancelBtn.style.display='none';
  if (msg && html) msg.innerHTML = html;

  // Hide header after a delay
  setTimeout(hideHeader, 3000);
  currentJobProtein = null;
  currentRunningJob = null;  // Clear both variables
}

async function cancelCurrentJob(){
  if (!currentJobProtein) {
    console.warn('No current job to cancel');
    return;
  }

  const cancelBtn = document.getElementById('mini-cancel-btn');
  if (cancelBtn) cancelBtn.disabled = true;

  try {
    const response = await fetch(`/api/cancel/${encodeURIComponent(currentJobProtein)}`, {
      method: 'POST'
    });

    if (response.ok) {
      miniDone('<span style="color:#dc2626;">Job cancelled.</span>');
    } else {
      const data = await response.json();
      miniDone(`<span style="color:#dc2626;">Failed to cancel: ${data.error || 'Unknown error'}</span>`);
    }
  } catch (error) {
    console.error('Cancel request failed:', error);
    miniDone('<span style="color:#dc2626;">Failed to cancel job.</span>');
  } finally {
    if (cancelBtn) cancelBtn.disabled = false;
  }
}
async function pollUntilComplete(p, onUpdate){
  for(;;){
    await new Promise(r=>setTimeout(r, 4000));
    try{
      const r = await fetch(`/api/status/${encodeURIComponent(p)}`);
      if (!r.ok){ onUpdate && onUpdate({text:`Waiting on ${p}…`}); continue; }
      const s = await r.json();
      if (s.status==='complete'){ onUpdate && onUpdate({text:`Complete: ${p}`,current:1,total:1}); break; }
      if (s.status==='cancelled' || s.status==='cancelling'){
        miniDone('<span style="color:#dc2626;">Job cancelled.</span>');
        throw new CancellationError('Job was cancelled by user');
      }
      const prog = s.progress || s;
      onUpdate && onUpdate({current:prog.current, total:prog.total, text:prog.text || s.status || 'Processing'});
    }catch(e){
      if (e instanceof CancellationError || e.name === 'CancellationError') throw e;
      onUpdate && onUpdate({text:`Rechecking ${p}…`});
    }
  }
}

// === Pruned expansion (client prefers prune, falls back to full) ===
const PRUNE_KEEP = 20;  // (#2) client cap; backend will enforce its own hard cap

function getCurrentProteinNodes() {
  // Only main + interactors (omit function boxes) (#3)
  return nodes.filter(n => n.type === 'main' || n.type === 'interactor').map(n => n.id);
}

function findMainEdgePayload(targetId) {
  // Enrich pruning relevance when main ↔ target exists; otherwise omit (#3)
  const hit = links.find(l => l.type === 'interaction' && (
    ((l.source.id || l.source) === SNAP.main && (l.target.id || l.target) === targetId) ||
    ((l.source.id || l.source) === targetId && (l.target.id || l.target) === SNAP.main)
  ));
  if (!hit) return null;
  const L = hit.data || {};
  return {
    arrow: hit.arrow || L.arrow || '',
    intent: L.intent || hit.intent || '',
    direction: L.direction || hit.direction || '',
    support_summary: L.support_summary || ''
  };
}

async function pollPruned(jobId, onUpdate) {
  for (;;) {
    await new Promise(r => setTimeout(r, 3000));
    try {
      const r = await fetch(`/api/expand/status/${encodeURIComponent(jobId)}`);
      if (!r.ok) throw new Error(`status ${r.status}`);
      const s = await r.json();
      if (s.status === 'complete') { onUpdate && onUpdate({ text: s.text || 'complete' }); break; }
      if (s.status === 'error') throw new Error(s.text || 'prune error');
      onUpdate && onUpdate({ text: s.text || s.status || 'processing' });
    } catch {
      onUpdate && onUpdate({ text: 'checking…' });
    }
  }
}

async function queueAndWaitFull(protein) {
  // (#6) Only label text changes, bar stays the same
  miniProgress('Initializing…', null, null, protein);
  const q = await fetch('/api/query', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ protein })
  });
  if (!q.ok) throw new Error('failed to queue full job');

  try {
    await pollUntilComplete(protein, ({ current, total, text }) =>
      miniProgress(text || 'Processing', current, total, protein)
    );
  } catch (e) {
    // Re-throw with proper error type
    if (e instanceof CancellationError || e.name === 'CancellationError') {
      throw new CancellationError(e.message);
    }
    throw e;
  }
}

async function tryPrunedExpand(interNode) {
  const payload = {
    parent: SNAP.main,                    // (#1) always the current root as parent
    protein: interNode.id,
    current_nodes: getCurrentProteinNodes(),
    parent_edge: findMainEdgePayload(interNode.id) || undefined,
    max_keep: PRUNE_KEEP                  // (#2) client-side cap
  };

  const resp = await fetch('/api/expand/pruned', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!resp.ok) throw new Error(`pruned request failed: ${resp.status}`);
  const j = await resp.json();
  const jobId = j.job_id;

  if (j.status === 'needs_full') {
    await queueAndWaitFull(interNode.id);
    return await tryPrunedExpand(interNode); // re-enter prune after full is built
  }

  if (j.status === 'queued' || j.status === 'processing') {
    // (#6) progress label: show "Pruning (relevance…)" and switch to "LLM" if backend reports it
    miniProgress('Pruning (relevance)…', null, null, interNode.id);
    await pollPruned(jobId, p => {
      const t = (p.text || '').toLowerCase();
      const label = t.includes('llm') ? 'Pruning (LLM)' : 'Pruning (relevance)';
      miniProgress(`${label}…`, null, null, interNode.id);
    });
  } else if (j.status !== 'complete') {
    throw new Error(`unexpected pruned status: ${j.status || 'unknown'}`);
  }

  const rr = await fetch(`/api/expand/results/${encodeURIComponent(jobId)}`);
  if (!rr.ok) throw new Error(`failed to load pruned results`);
  const pruned = await rr.json();
  await mergeSubgraph(pruned, interNode);
  miniDone(`<span>Added pruned subgraph for <b>${interNode.id}</b> (≤${PRUNE_KEEP}).</span>`);
}

// Current full-flow used as fallback
async function expandViaFullFlow(interNode) {
  const id = interNode.id;
  let res = await fetch(`/api/results/${encodeURIComponent(id)}`);
  if (res.ok) {
    const raw = await res.json();
    await mergeSubgraph(raw, interNode);
    miniDone(`<span>Added subgraph for <b>${id}</b>.</span>`);
    return;
  }
  if (res.status === 404) {
    try {
      await queueAndWaitFull(id);
    } catch (e) {
      // Re-throw cancellation errors
      if (e instanceof CancellationError || e.name === 'CancellationError') {
        throw e;
      }
      throw e;
    }
    const r2 = await fetch(`/api/results/${encodeURIComponent(id)}`);
    if (!r2.ok) { miniDone(`<span>No results for ${id} after job.</span>`); return; }
    const raw2 = await r2.json();
    await mergeSubgraph(raw2, interNode);
    miniDone(`<span>Added subgraph for <b>${id}</b>.</span>`);
    return;
  }
  miniDone(`<span>Error loading ${id}: ${res.status}</span>`);
}

/* ===== Expand-on-click with depth limit ===== */
const MAX_DEPTH = 3;
const depthMap = new Map();
const expanded = new Set();
// NOTE: depthMap is now populated in buildInitialGraph() for NEW format compatibility
// REMOVED: Legacy seedDepths() IIFE that only worked with SNAP.interactors (OLD format)

async function expandInteractor(interNode){
  const id = interNode.id;
  const depth = depthMap.get(id) ?? 1;
  const msg = document.getElementById('notification-message');

  // Toggle collapse
  if (expanded.has(id)){
    await collapseInteractor(id);
    if (msg) msg.innerHTML = `<span>Collapsed subgraph for <b>${id}</b>.</span>`;
    return;
  }
  if (depth >= MAX_DEPTH){
    if (msg) msg.innerHTML = `<span>Depth limit (${MAX_DEPTH}) reached for ${id}.</span>`;
    return;
  }

  try {
    // Prefer pruned; clean fallback to full flow
    await tryPrunedExpand(interNode).catch(async (e) => {
      // Don't fallback if user cancelled
      if (e instanceof CancellationError || e.name === 'CancellationError') {
        throw e;  // Re-throw cancellation errors
      }
      console.warn('Pruned expand failed, falling back:', e);
      await expandViaFullFlow(interNode);
    });
  } catch (err) {
    // Don't show error message for cancellations
    if (err instanceof CancellationError || err.name === 'CancellationError') {
      return;  // Silent exit on cancellation
    }
    miniDone(`<span>Error expanding ${id}: ${err?.message || err}</span>`);
  }
}

async function mergeSubgraph(raw, clickedNode){
  // NEW: Extract from new data structure
  const sub = (raw && raw.snapshot_json) ? raw.snapshot_json : raw;

  // NEW: Check for new data structure (proteins and interactions arrays)
  if (!sub || !Array.isArray(sub.proteins) || !Array.isArray(sub.interactions)) {
    console.error('❌ mergeSubgraph: Invalid data structure!');
    console.error('  Expected: { proteins: [...], interactions: [...] }');
    console.error('  Got:', sub);
    return;
  }

  // Determine cluster position for the expansion
  // Calculate ONCE and store for later cluster creation
  let newClusterPos = null;
  let centerX, centerY;

  if (clusters.has(clickedNode.id)) {
    // Cluster already exists, use its position
    const cluster = clusters.get(clickedNode.id);
    centerX = cluster.centerPos.x;
    centerY = cluster.centerPos.y;
  } else {
    // New cluster - calculate position now, create cluster later
    // Pass interactor count for dynamic spacing
    const interactorCount = sub.proteins.length - 1; // Exclude the clicked protein itself
    newClusterPos = getNextClusterPosition(interactorCount);
    centerX = newClusterPos.x;
    centerY = newClusterPos.y;
  }

  const nodeIds = new Set(nodes.map(n=>n.id));
  const linkIds = new Set(links.map(l=>l.id));
  const parentDepth = depthMap.get(clickedNode.id) ?? 1;
  const childDepth = Math.min(MAX_DEPTH, parentDepth+1);

  const regNodes = new Set();
  const regLinks = new Set();

  // NEW: Add protein nodes (exclude clicked node if already exists)
  const newProteins = sub.proteins.filter(p => p !== clickedNode.id && !nodeIds.has(p));

  // Calculate cluster radius for positioning (use existing cluster if available, or calculate new one)
  let clusterRadius;
  if (clusters.has(clickedNode.id)) {
    clusterRadius = clusters.get(clickedNode.id).radius;
  } else {
    // Calculate radius for new cluster based on protein count
    clusterRadius = calculateClusterRadius(newProteins.length);
  }

  newProteins.forEach((protein, idx) => {
    // Position nodes in a small circle within the cluster
    const angle = (2*Math.PI*idx)/Math.max(1, newProteins.length) - Math.PI/2;
    const radius = clusterRadius * 0.6; // Position within cluster bounds (60% of calculated radius)
    const x = centerX + Math.cos(angle)*radius;
    const y = centerY + Math.sin(angle)*radius;

    // Create new protein node
    nodes.push({
      id: protein,
      label: protein,
      type: 'interactor',
      radius: interactorNodeRadius,
      x: x,
      y: y
    });

    nodeIds.add(protein);
    depthMap.set(protein, childDepth);

    // Track for expansion registry (for collapse)
    if (!baseNodes || !baseNodes.has(protein)){
      if (!regNodes.has(protein)){
        refCounts.set(protein, (refCounts.get(protein) || 0) + 1);
        regNodes.add(protein);
      }
    }
  });

  // NEW: Add interaction links (all types: direct, shared, cross_link)
  sub.interactions.forEach(interaction => {
    const source = interaction.source;
    const target = interaction.target;

    if (!source || !target) {
      console.warn('mergeSubgraph: Interaction missing source/target', interaction);
      return;
    }

    // Determine arrow type
    const arrow = arrowKind(
      interaction.arrow || 'binds',
      interaction.intent || 'binding',
      interaction.direction || 'main_to_primary'
    );

    // Create link ID with arrow type (to allow parallel links with different arrows)
    const linkId = `${source}-${target}-${arrow}`;
    const reverseLinkId = `${target}-${source}-${arrow}`;

    // Skip if link already exists in base graph
    const inBase = (baseLinks && (baseLinks.has(linkId) || baseLinks.has(reverseLinkId)));
    if (inBase) {
      return;
    }

    // Skip if link already added in this merge
    if (linkIds.has(linkId)) {
      return;
    }

    // Check if reverse exists
    const reverseExists = linkIds.has(reverseLinkId);

    // Determine if bidirectional
    const isBidirectional = isBiDir(interaction.direction) || reverseExists;

    // Create link
    const link = {
      id: linkId,
      source: source,
      target: target,
      type: 'interaction',
      interactionType: interaction.type || 'direct',
      arrow: arrow,
      intent: interaction.intent || 'binding',
      direction: interaction.direction || 'main_to_primary',
      data: interaction,
      isBidirectional: isBidirectional,
      linkOffset: reverseExists ? 1 : 0,
      showBidirectionalMarkers: isBidirectional,
      confidence: interaction.confidence || 0.5,

      // PERFORMANCE: Cache constant values to avoid recalculation in every tick
      _sourceRadius: null,  // Will be set after D3 binds node objects
      _targetRadius: null,  // Will be set after D3 binds node objects
      _isShared: (interaction.type === 'shared' || interaction.interactionType === 'shared'),
      _needsCurve: isBidirectional || (interaction.type === 'shared') || (interaction.interactionType === 'shared')
    };

    links.push(link);
    linkIds.add(linkId);

    // Track for expansion registry (for collapse)
    if (!baseLinks || !baseLinks.has(linkId)){
      if (!regLinks.has(linkId)){
        refCounts.set(linkId, (refCounts.get(linkId) || 0) + 1);
        regLinks.add(linkId);
      }
    }
  });

  // Create new cluster for the expanded protein if needed
  if (!clusters.has(clickedNode.id) && newClusterPos) {
    // Remove clicked node from its old cluster
    const oldClusterId = getNodeCluster(clickedNode.id);
    if (oldClusterId) {
      const oldCluster = clusters.get(oldClusterId);
      if (oldCluster) {
        oldCluster.members.delete(clickedNode.id);
        // PERFORMANCE: Update reverse cluster lookup map
        nodeToClusterMap.delete(clickedNode.id);
      }
    }

    // Create new cluster and move the clicked node to it
    createCluster(clickedNode.id, newClusterPos, newProteins.length);
  } else if (!newClusterPos && !clusters.has(clickedNode.id)) {
    console.error(`ERROR: newClusterPos is null/undefined for ${clickedNode.id}`);
  }

  // ALWAYS add new proteins to the cluster (whether newly created or pre-existing)
  // CRITICAL FIX: This was inside the conditional above, causing drag issues on re-expansion
  const targetCluster = clusters.get(clickedNode.id);

  if (targetCluster && newProteins.length > 0) {
    // Add all new proteins to the expanded cluster
    newProteins.forEach(protein => {
      addNodeToCluster(clickedNode.id, protein);
    });

    // Mark intra-cluster links
    sub.interactions.forEach(interaction => {
      const source = interaction.source;
      const target = interaction.target;
      const arrow = arrowKind(interaction.arrow || 'binds', interaction.intent || 'binding', interaction.direction || 'main_to_primary');
      const linkId = `${source}-${target}-${arrow}`;

      // If both nodes are in the cluster, it's an intra-cluster link
      if (targetCluster.members.has(source) && targetCluster.members.has(target)) {
        targetCluster.localLinks.add(linkId);
      }
    });

    // CRITICAL FIX: Ensure all cluster member positions are valid and synced
    // This prevents drag issues where member positions might not be initialized yet
    let validPosCount = 0;
    let invalidPosCount = 0;
    const clusterCenterX = centerX; // Use the centerX/centerY calculated earlier
    const clusterCenterY = centerY;

    targetCluster.members.forEach(memberId => {
      const member = nodeMap.get(memberId); // PERFORMANCE: O(1) lookup
      if (member) {
        if (Number.isFinite(member.x) && Number.isFinite(member.y) &&
            member.x !== 0 && member.y !== 0) {
          validPosCount++;
        } else {
          invalidPosCount++;
          // If position is invalid, set it to cluster center + small offset
          const offset = Math.random() * 50 - 25;
          member.x = clusterCenterX + offset;
          member.y = clusterCenterY + offset;
          console.warn(`Fixed invalid position for ${memberId}: set to (${member.x}, ${member.y})`);
        }
      }
    });

    // PERFORMANCE: Console logs commented out to improve rendering speed
    // console.log(`\n✅ CLUSTER UPDATE COMPLETE for ${clickedNode.id}:`);
    // console.log(`  - Position: (${clusterCenterX}, ${clusterCenterY})`);
    // console.log(`  - Members (${targetCluster.members.size}):`, Array.from(targetCluster.members));
    // console.log(`  - New proteins added: ${newProteins.join(', ')}`);
    // console.log(`  - Center node position: (${clickedNode.x}, ${clickedNode.y}), fixed: (${clickedNode.fx}, ${clickedNode.fy})`);
    // console.log(`  - Member positions: ${validPosCount} valid, ${invalidPosCount} fixed`);
    // console.log(`  - Cluster in map:`, clusters.has(clickedNode.id));
    // console.log(`  - Total clusters:`, clusters.size);
    // console.log(`  - All cluster keys:`, Array.from(clusters.keys()));
  } else if (!targetCluster) {
    console.error(`❌ CLUSTER ERROR: No cluster found for ${clickedNode.id} after creation attempt!`);
  } else if (newProteins.length === 0) {
    console.warn(`⚠️ WARNING: No new proteins to add to cluster ${clickedNode.id}`);
  }

  // Mark expansion as complete
  expanded.add(clickedNode.id);
  expansionRegistry.set(clickedNode.id, { nodes: regNodes, links: regLinks });

  // Reposition indirect interactors near their upstream interactors (hybrid layout)
  // Group newly added indirect nodes by upstream
  const newIndirectGroups = new Map();

  // PERFORMANCE: Build link lookup map to avoid O(N×M) nested loop
  const linksByTarget = new Map();
  links.forEach(link => {
    const target = (link.target && link.target.id) ? link.target.id : link.target;
    if (!linksByTarget.has(target)) {
      linksByTarget.set(target, []);
    }
    linksByTarget.get(target).push(link);
  });

  // Now iterate nodes once and look up links in O(1)
  nodes.forEach(node => {
    if (regNodes.has(node.id) && node.type === 'interactor') {
      // Check if this newly added node is an indirect interactor - PERFORMANCE: O(1) lookup
      const nodeLinks = linksByTarget.get(node.id) || [];
      const link = nodeLinks.find(l =>
        l?.data?.interaction_type === 'indirect' && l?.data?.upstream_interactor
      );

      if (link) {
        const upstream = link.data.upstream_interactor;
        if (!newIndirectGroups.has(upstream)) {
          newIndirectGroups.set(upstream, []);
        }
        newIndirectGroups.get(upstream).push(node);

        // Copy upstream info to node for force simulation
        node.upstream_interactor = upstream;
        node.interaction_type = 'indirect';
      }
    }
  });

  // Position each group around its upstream node
  newIndirectGroups.forEach((indirectNodes, upstreamId) => {
    const upstreamNode = nodeMap.get(upstreamId); // PERFORMANCE: O(1) lookup

    if (!upstreamNode) {
      console.warn(`mergeSubgraph: Upstream node ${upstreamId} not found`);
      return;
    }

    // Position in small orbital ring around upstream
    const orbitalRadius = 200;
    indirectNodes.forEach((node, idx) => {
      const angle = (2 * Math.PI * idx) / Math.max(indirectNodes.length, 1);
      node.x = upstreamNode.x + Math.cos(angle) * orbitalRadius;
      node.y = upstreamNode.y + Math.sin(angle) * orbitalRadius;
      delete node.fx;
      delete node.fy;
    });
  });

  // PERFORMANCE: Rebuild node lookup map after adding new nodes
  rebuildNodeMap();

  // Update graph with smooth transitions
  updateGraphWithTransitions();
}

// --- collapse helper: remove one expansion safely ---
async function collapseInteractor(ownerId){
  const reg = expansionRegistry.get(ownerId);
  if (!reg){ expanded.delete(ownerId); return; }

  // Remove links first
  const toRemoveLinks = [];
  reg.links.forEach(lid => {
    if (baseLinks && baseLinks.has(lid)) return; // never remove base
    const c = (refCounts.get(lid) || 0) - 1;
    if (c <= 0){ refCounts.delete(lid); toRemoveLinks.push(lid); }
    else { refCounts.set(lid, c); }
  });
  if (toRemoveLinks.length){
    links = links.filter(l => !toRemoveLinks.includes(l.id));
  }

  // Remove nodes (only if no remaining incident links)
  const toRemoveNodes = [];
  reg.nodes.forEach(nid => {
    if (baseNodes && baseNodes.has(nid)) return;
    const c = (refCounts.get(nid) || 0) - 1;
    if (c <= 0){
      const stillUsed = links.some(l => ((l.source.id||l.source)===nid) || ((l.target.id||l.target)===nid));
      if (!stillUsed){ refCounts.delete(nid); toRemoveNodes.push(nid); }
      else { refCounts.set(nid, 0); }
    } else {
      refCounts.set(nid, c);
    }
  });
  if (toRemoveNodes.length){
    nodes = nodes.filter(n => !toRemoveNodes.includes(n.id));
  }

  // Remove cluster if it was created for this expansion
  if (clusters.has(ownerId)) {
    // Before deleting, move the owner node back to root cluster
    const ownerNode = nodeMap.get(ownerId); // PERFORMANCE: O(1) lookup
    if (ownerNode) {
      // Release fixed position so it can move
      ownerNode.fx = null;
      ownerNode.fy = null;

      // Find the main cluster (root cluster) - PERFORMANCE: Use cached reference
      const mainNode = cachedMainNode;
      if (mainNode && clusters.has(mainNode.id)) {
        const rootCluster = clusters.get(mainNode.id);
        rootCluster.members.add(ownerId);
        // PERFORMANCE: Update reverse cluster lookup map
        nodeToClusterMap.set(ownerId, mainNode.id);

        // Position it near the root cluster center for smooth transition
        const rootPos = rootCluster.centerPos;
        const angle = Math.random() * Math.PI * 2;
        const radius = rootCluster.radius * 0.7; // Use cluster's calculated radius
        ownerNode.x = rootPos.x + Math.cos(angle) * radius;
        ownerNode.y = rootPos.y + Math.sin(angle) * radius;
      }
    }

    // PERFORMANCE: Clean up reverse cluster lookup map for all members of deleted cluster
    const deletedCluster = clusters.get(ownerId);
    if (deletedCluster) {
      deletedCluster.members.forEach(memberId => {
        if (nodeToClusterMap.get(memberId) === ownerId) {
          nodeToClusterMap.delete(memberId);
        }
      });
    }
    clusters.delete(ownerId);
  }

  expansionRegistry.delete(ownerId);
  expanded.delete(ownerId);
  // PERFORMANCE: Rebuild node lookup map after removing nodes
  rebuildNodeMap();
  updateGraphWithTransitions();
}

/**
 * Updates graph with smooth D3 transitions (works with force simulation)
 */
function updateGraphWithTransitions(){
  // Initialize new nodes with orbital positions
  nodes.forEach(node => {
    if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) {
      const pos = calculateOrbitalPosition(node);
      node.x = pos.x;
      node.y = pos.y;
    }
  });

  // Update links with transitions
  if (!linkGroup) {
    // First render - no transitions
    rebuild();
    return;
  }

  // LINK UPDATE PATTERN
  const linkData = linkGroup.data(links, d => d.id);

  // EXIT: Remove old links
  linkData.exit()
    .transition().duration(300)
    .style('opacity', 0)
    .remove();

  // UPDATE: Update existing links
  linkData
    .transition().duration(400)
    .attr('d', calculateLinkPath);

  // ENTER: Add new links
  const linkEnter = linkData.enter().append('path')
    .attr('class', d=>{
      const arrow = d.arrow||'binds';
      let classes = 'link';
      if (arrow==='binds') classes += ' link-binding';
      else if (arrow==='activates') classes += ' link-activate';
      else if (arrow==='inhibits') classes += ' link-inhibit';
      else classes += ' link-binding';
      if (d.interaction_type === 'indirect') {
        classes += ' link-indirect';
      }
      if (d.interactionType === 'shared' || d.interactionType === 'cross_link') {
        classes += ' link-shared';
      }
      if (d._incomplete_pathway) {
        classes += ' link-incomplete';
      }
      return classes;
    })
    .attr('marker-start', d=>{
      const dir = (d.direction || '').toLowerCase();
      // marker-start shows arrow at source end
      // Use for bidirectional (both ends) only
      if (dir === 'bidirectional') {
        const a=d.arrow||'binds';
        if (a==='activates') return 'url(#arrow-activate)';
        if (a==='inhibits') return 'url(#arrow-inhibit)';
        return 'url(#arrow-binding)';
      }
      return null;
    })
    .attr('marker-end', d=>{
      const dir = (d.direction || '').toLowerCase();
      // marker-end shows arrow at target end (default for all directed arrows)
      // Support both query-relative (main_to_primary) AND absolute (a_to_b) directions
      // Query-relative: main_to_primary, primary_to_main, bidirectional
      // Absolute: a_to_b, b_to_a (used for shared links and database storage)
      if (dir === 'main_to_primary' || dir === 'primary_to_main' || dir === 'bidirectional' ||
          dir === 'a_to_b' || dir === 'b_to_a') {
        const a=d.arrow||'binds';
        if (a==='activates') return 'url(#arrow-activate)';
        if (a==='inhibits') return 'url(#arrow-inhibit)';
        return 'url(#arrow-binding)';
      }
      return null;
    })
    .attr('fill','none')
    .attr('d', calculateLinkPath)
    .style('opacity', 0)
    .on('mouseover', function(){ d3.select(this).style('stroke-width','3.5'); svg.style('cursor','pointer'); })
    .on('mouseout',  function(){ d3.select(this).style('stroke-width',null);  svg.style('cursor',null); })
    .on('click', handleLinkClick);

  linkEnter.transition().duration(400).style('opacity', 1);

  // Merge enter + update
  linkGroup = linkEnter.merge(linkData);

  // PERFORMANCE: Initialize cached radii for new expansion links (D3 has now bound node objects)
  links.forEach(link => {
    if (!link._sourceRadius || !link._targetRadius) {
      const src = link.source;
      const tgt = link.target;
      if (typeof src === 'object' && typeof tgt === 'object') {
        link._sourceRadius = src.type === 'main' ? mainNodeRadius :
                            (src.type === 'interactor' ? interactorNodeRadius : 0);
        link._targetRadius = tgt.type === 'main' ? mainNodeRadius :
                            (tgt.type === 'interactor' ? interactorNodeRadius : 0);
      }
    }
  });

  // NODE UPDATE PATTERN
  const nodeData = nodeGroup.data(nodes, d => d.id);

  // EXIT: Remove old nodes
  nodeData.exit()
    .transition().duration(300)
    .style('opacity', 0)
    .remove();

  // UPDATE: Move existing nodes and update expanded state
  nodeData.each(function(d) {
    if (d.type === 'interactor') {
      // Update class and radius based on whether this node is now a cluster center
      const isExpanded = clusters.has(d.id);
      const nodeClass = isExpanded ? 'node expanded-node' : 'node interactor-node';
      const nodeRadius = isExpanded ? expandedNodeRadius : interactorNodeRadius;
      d3.select(this).select('circle')
        .attr('class', nodeClass)
        .attr('r', nodeRadius);
    }
  });
  nodeData
    .transition().duration(500)
    .attr('transform', d => `translate(${d.x},${d.y})`);

  // ENTER: Add new nodes
  const nodeEnter = nodeData.enter().append('g')
    .attr('class','node-group')
    .attr('transform', d => {
      // Start from parent position for smooth animation - PERFORMANCE: Search registry map instead of nodes array
      let parent = null;
      for (const [parentId, registry] of expansionRegistry.entries()) {
        if (registry && registry.nodes && registry.nodes.has(d.id)) {
          parent = nodeMap.get(parentId);
          break;
        }
      }
      if (parent && parent.x && parent.y) {
        return `translate(${parent.x},${parent.y})`;
      }
      return `translate(${d.x},${d.y})`;
    })
    .style('opacity', 0);

  nodeEnter.each(function(d){
    const group = d3.select(this);
    if (d.type==='main'){
      group.append('circle')
        .attr('class','node main-node')
        .attr('r', mainNodeRadius)
        .style('cursor','pointer')
        .on('click', (ev)=>{ ev.stopPropagation(); handleNodeClick(d); });
      group.append('text').attr('class','node-label main-label').attr('dy',5).text(d.label);
    } else if (d.type==='interactor'){
      // Check if this interactor has been expanded (is a cluster center)
      const isExpanded = clusters.has(d.id);
      const nodeClass = isExpanded ? 'node expanded-node' : 'node interactor-node';
      group.append('circle')
        .attr('class', nodeClass)
        .attr('r', isExpanded ? expandedNodeRadius : interactorNodeRadius)
        .style('cursor','pointer')
        .on('click', (ev)=>{ ev.stopPropagation(); handleNodeClick(d); });
      group.append('text').attr('class','node-label').attr('dy',5).text(d.label);
    }
  });

  // Animate new nodes to final position
  nodeEnter.transition().duration(500)
    .attr('transform', d => `translate(${d.x},${d.y})`)
    .style('opacity', 1);

  // Merge enter + update
  nodeGroup = nodeEnter.merge(nodeData);

  // Add drag handlers to new nodes
  nodeEnter.call(d3.drag()
    .on('start', dragstarted)
    .on('drag', dragged)
    .on('end', dragended));

  // Update simulation with new data
  if (simulation) {
    simulation.nodes(nodes);

    // Filter to only intra-cluster links for force
    const intraClusterLinks = links.filter(link => {
      const type = classifyLink(link);
      return type === 'intra-cluster';
    });

    simulation.force('link').links(intraClusterLinks);

    // Reheat simulation to settle new nodes
    if (nodeEnter.size() > 0) {
      reheatSimulation(0.4);
    }
  }

  // Update table view
  buildTableView();

  // After transitions complete, zoom to new nodes
  if (nodeEnter.size() > 0) {
    setTimeout(() => {
      focusOnNewNodes(nodeEnter.data());
    }, 600); // Wait for node animations to complete
  }
}

/**
 * Smoothly zooms camera to focus on newly added nodes
 * @param {array} newNodes - Array of newly added node data objects
 */
function focusOnNewNodes(newNodes) {
  if (!newNodes || newNodes.length === 0) return;

  // Calculate bounding box of new nodes
  const padding = 150;
  const xs = newNodes.map(n => n.x).filter(x => Number.isFinite(x));
  const ys = newNodes.map(n => n.y).filter(y => Number.isFinite(y));

  if (xs.length === 0 || ys.length === 0) return;

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  // Calculate cluster dimensions
  const clusterWidth = Math.max(maxX - minX, 100); // Min 100px
  const clusterHeight = Math.max(maxY - minY, 100);
  const clusterCenterX = (minX + maxX) / 2;
  const clusterCenterY = (minY + maxY) / 2;

  // Calculate zoom scale to fit cluster with padding
  const viewWidth = width || 1000;
  const viewHeight = height || 800;
  const scaleX = (viewWidth - padding * 2) / clusterWidth;
  const scaleY = (viewHeight - padding * 2) / clusterHeight;
  const scale = Math.min(Math.max(scaleX, scaleY, 0.5), 2.0); // Clamp between 0.5x and 2x

  // Calculate translate to center the cluster
  const translateX = viewWidth / 2 - scale * clusterCenterX;
  const translateY = viewHeight / 2 - scale * clusterCenterY;

  // Apply smooth zoom transition
  const transform = d3.zoomIdentity
    .translate(translateX, translateY)
    .scale(scale);

  svg.transition()
    .duration(750)
    .ease(d3.easeCubicOut)
    .call(zoomBehavior.transform, transform);
}

/**
 * Full rebuild (used for initial render only)
 */
function rebuild(){
  // Clear existing visualization
  g.selectAll('*').remove();

  // Create force simulation with orbital constraints
  createSimulation();

  // Rebind interactor click handlers
  try{
    g.selectAll('.node-group').filter(d=>d.type==='interactor')
      .on('click', (ev,d)=>{ ev.stopPropagation(); handleNodeClick(d); });
  }catch(e){}

  // Update table view when graph changes
  buildTableView();
}

/* Zoom controls */
function scheduleFitToView(delay = 450, animate = true) {
  if (fitToViewTimer) {
    clearTimeout(fitToViewTimer);
  }
  fitToViewTimer = setTimeout(() => {
    fitGraphToView(120, animate);
  }, Math.max(0, delay));
}

function fitGraphToView(padding = 120, animate = true) {
  if (!svg || !zoomBehavior) return;
  const container = document.getElementById('network');
  if (!container) return;

  const viewWidth = container.clientWidth || width || 0;
  const viewHeight = container.clientHeight || height || 0;
  if (viewWidth < 10 || viewHeight < 10) return;

  width = viewWidth;
  height = viewHeight;
  svg.attr('width', width).attr('height', height);

  const positioned = nodes.filter(n => Number.isFinite(n.x) && Number.isFinite(n.y));
  if (!positioned.length) return;

  const [minX, maxX] = d3.extent(positioned, d => d.x);
  const [minY, maxY] = d3.extent(positioned, d => d.y);
  if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) return;

  const graphWidth = Math.max(maxX - minX, 1);
  const graphHeight = Math.max(maxY - minY, 1);
  const safePadding = Math.min(padding, Math.min(viewWidth, viewHeight) / 3);

  const scaleX = (viewWidth - safePadding * 2) / graphWidth;
  const scaleY = (viewHeight - safePadding * 2) / graphHeight;
  const targetScale = Math.max(0.35, Math.min(2.4, Math.min(scaleX, scaleY)));

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  const translateX = (viewWidth / 2) - targetScale * centerX;
  const translateY = (viewHeight / 2) - targetScale * centerY;
  const transform = d3.zoomIdentity.translate(translateX, translateY).scale(targetScale);

  if (animate) {
    svg.transition().duration(500).ease(d3.easeCubicOut).call(zoomBehavior.transform, transform);
  } else {
    svg.call(zoomBehavior.transform, transform);
  }

  graphInitialFitDone = true;
}

function reheatSimulation(alpha = 0.65) {
  if (!simulation) return;
  const targetAlpha = Math.max(alpha, simulation.alpha());
  simulation.alpha(targetAlpha).alphaTarget(0);
  simulation.restart();
}

function zoomIn(){
  if (!svg || !zoomBehavior) return;
  svg.transition().duration(250).ease(d3.easeCubicOut).call(zoomBehavior.scaleBy, 1.2);
}
function zoomOut(){
  if (!svg || !zoomBehavior) return;
  svg.transition().duration(250).ease(d3.easeCubicOut).call(zoomBehavior.scaleBy, 0.8);
}
function resetView(){
  if (!svg || !zoomBehavior) return;
  nodes.forEach(node => {
    if (node.type === 'main') {
      node.fx = width / 2;
      node.fy = height / 2;
    } else {
      node.fx = null;
      node.fy = null;
    }
  });
  reheatSimulation(0.7);
  scheduleFitToView(360, true);
}

function toggleTheme(){
  document.body.classList.toggle('dark-mode');
  const isDark = document.body.classList.contains('dark-mode');
  const icon = document.getElementById('theme-icon');
  if (icon) {
    icon.textContent = isDark ? '☀️' : '🌙';
  }
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

/* ===== Graph Filters ===== */
let graphActiveFilters = new Set(['activates', 'inhibits', 'binds', 'regulates']);
let graphActiveDepths = new Set([0, 1, 2, 3]); // All depths visible by default (0=main, 1=direct, 2=indirect, 3=tertiary)

function toggleGraphFilter(filterType) {
  if (graphActiveFilters.has(filterType)) {
    graphActiveFilters.delete(filterType);
  } else {
    graphActiveFilters.add(filterType);
  }

  // Update button visual state
  const btn = document.querySelector(`.graph-filter-btn.${filterType}`);
  if (btn) {
    btn.classList.toggle('active');
  }

  // Update graph visibility
  applyGraphFilters();
}

function toggleDepthFilter(depth) {
  // Never allow hiding depth 0 (main protein)
  if (depth === 0) return;

  if (graphActiveDepths.has(depth)) {
    graphActiveDepths.delete(depth);
  } else {
    graphActiveDepths.add(depth);
  }

  // Update button visual state
  const btn = document.querySelector(`.depth-filter[data-depth="${depth}"]`);
  if (btn) {
    btn.classList.toggle('active');
  }

  // Update graph visibility
  applyGraphFilters();
}

function refreshVisualization() {
  // Clear existing SVG elements to prevent duplicate graphs
  if (g) g.selectAll('*').remove();

  // Clear clusters - PERFORMANCE: Also clear reverse lookup map
  clusters.clear();
  nodeToClusterMap.clear();
  nextClusterAngle = 0;

  // Rebuild the graph from current data (buildInitialGraph already clears nodes/links)
  if (typeof buildInitialGraph === 'function') {
    buildInitialGraph();

    // Reset base graph tracking
    baseNodes = new Set(nodes.map(n => n.id));
    baseLinks = new Set(links.map(l => l.id));
    // PERFORMANCE: Cache main node reference for O(1) lookup in calculateLinkPath
    cachedMainNode = nodes.find(n => n.type === 'main');
    // PERFORMANCE: Build node lookup map for O(1) access
    rebuildNodeMap();

    // Recreate force simulation
    createSimulation();

    // Reset expansion tracking
    expansionRegistry.clear();
    expanded.clear();
    refCounts.clear();
  }
}

function applyGraphFilters() {
  if (!g) return;

  // Update link visibility and opacity
  g.selectAll('path.link').each(function(d) {
    const link = d3.select(this);
    const arrow = d.arrow || 'binds';

    if (d.type === 'interaction') {
      // Check both arrow type and depth filters - PERFORMANCE: O(1) lookup
      const targetId = d.target?.id || d.target;
      const sourceId = d.source?.id || d.source;
      const targetNode = typeof targetId === 'string' ? nodeMap.get(targetId) : d.target;
      const sourceNode = typeof sourceId === 'string' ? nodeMap.get(sourceId) : d.source;
      const maxDepth = Math.max(
        depthMap.get(targetNode?.id || '') || 0,
        depthMap.get(sourceNode?.id || '') || 0
      );

      const arrowMatch = graphActiveFilters.has(arrow);
      const depthMatch = graphActiveDepths.has(maxDepth);
      const shouldShow = arrowMatch && depthMatch;

      link.style('display', shouldShow ? null : 'none');
      link.style('opacity', shouldShow ? 0.7 : 0);
    }
  });

  // Update node visibility - hide interactors if all their interactions are filtered out OR depth filtered
  g.selectAll('g.node-group').each(function(d) {
    const nodeGroup = d3.select(this);

    // Main protein is always visible
    if (d.type === 'main') {
      nodeGroup.style('opacity', 1);
      nodeGroup.style('pointer-events', 'all');
      return;
    }

    if (d.type === 'interactor') {
      const nodeDepth = depthMap.get(d.id) || 0;
      const depthVisible = graphActiveDepths.has(nodeDepth);

      // Check if any links to this interactor are visible
      const hasVisibleLink = depthVisible && links.some(l => {
        if (l.type !== 'interaction') return false;
        const targetId = (l.target && l.target.id) ? l.target.id : l.target;
        const sourceId = (l.source && l.source.id) ? l.source.id : l.source;
        const isConnected = targetId === d.id || sourceId === d.id;
        const arrow = l.arrow || 'binds';

        // Check if the link itself passes depth filter - PERFORMANCE: O(1) lookup
        const linkTargetNode = typeof targetId === 'string' ? nodeMap.get(targetId) : l.target;
        const linkSourceNode = typeof sourceId === 'string' ? nodeMap.get(sourceId) : l.source;
        const linkMaxDepth = Math.max(
          depthMap.get(linkTargetNode?.id || '') || 0,
          depthMap.get(linkSourceNode?.id || '') || 0
        );

        return isConnected && graphActiveFilters.has(arrow) && graphActiveDepths.has(linkMaxDepth);
      });

      nodeGroup.style('opacity', hasVisibleLink ? 1 : 0.2);
      nodeGroup.style('pointer-events', hasVisibleLink ? 'all' : 'none');
    }
  });
}

/* ===== Table View ===== */
// Search and filter state
let searchQuery = '';
let activeFilters = new Set(['activates', 'inhibits', 'binds', 'regulates']);
let searchDebounceTimer = null;

function switchView(viewName) {
  const graphView = document.getElementById('network');
  const tableView = document.getElementById('table-view');
  const chatView = document.getElementById('chat-view');
  const tabs = document.querySelectorAll('.tab-btn');
  const header = document.querySelector('.header');
  const container = document.querySelector('.container');

  // Hide all views first
  graphView.style.display = 'none';
  tableView.style.display = 'none';
  chatView.style.display = 'none';

  // Remove active from all tabs
  tabs.forEach(tab => tab.classList.remove('active'));

  if (viewName === 'graph') {
    graphView.style.display = 'block';
    tabs[0].classList.add('active');
    // Remove static class to restore auto-hide behavior
    if (header) header.classList.remove('header-static');
    // Enable graph view scroll behavior
    document.body.classList.remove('table-view-active');
    document.body.classList.add('graph-view-active');
    if (container) container.classList.add('graph-active');
    scheduleFitToView(180, true);
  } else if (viewName === 'table') {
    tableView.style.display = 'flex';
    tabs[1].classList.add('active');
    buildTableView(); // Rebuild on switch to ensure current state
    // Make header static (always visible) for table view
    if (header) header.classList.add('header-static');
    // Enable page scroll for table view
    document.body.classList.remove('graph-view-active');
    document.body.classList.add('table-view-active');
    if (container) container.classList.remove('graph-active');
    // Reset search and filters when switching to table view
    searchQuery = '';
    activeFilters = new Set(['activates', 'inhibits', 'binds', 'regulates']);
    const searchInput = document.getElementById('table-search');
    if (searchInput) searchInput.value = '';
    document.querySelectorAll('.filter-chip').forEach(chip => chip.classList.add('filter-active'));
    applyFilters();
  } else if (viewName === 'chat') {
    chatView.style.display = 'block';
    tabs[2].classList.add('active');
    // Use auto-hide header for chat view (same as graph view)
    if (header) header.classList.remove('header-static');
    // Enable page scroll for chat view
    document.body.classList.remove('graph-view-active');
    document.body.classList.add('table-view-active');
    if (container) container.classList.remove('graph-active');
    // Focus chat input when switching to chat view
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
      setTimeout(() => chatInput.focus(), 100);
    }
  }
}

function handleSearchInput(event) {
  const query = event.target.value;
  const clearBtn = document.getElementById('search-clear-btn');

  // Show/hide clear button
  if (clearBtn) {
    clearBtn.style.display = query ? 'flex' : 'none';
  }

  // Debounce search
  clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(() => {
    searchQuery = query.toLowerCase().trim();
    applyFilters();
  }, 300);
}

function clearSearch() {
  const searchInput = document.getElementById('table-search');
  if (searchInput) {
    searchInput.value = '';
    searchQuery = '';
    const clearBtn = document.getElementById('search-clear-btn');
    if (clearBtn) clearBtn.style.display = 'none';
    applyFilters();
  }
}

function toggleFilter(filterType) {
  if (activeFilters.has(filterType)) {
    activeFilters.delete(filterType);
  } else {
    activeFilters.add(filterType);
  }

  // Update visual state
  const chip = document.querySelector(`.filter-chip.${filterType}`);
  if (chip) {
    chip.classList.toggle('filter-active');
  }

  applyFilters();
}

function applyFilters() {
  const tbody = document.getElementById('table-body');
  if (!tbody) return;

  const functionRows = tbody.querySelectorAll('tr.function-row');
  let visibleCount = 0;

  functionRows.forEach(row => {
    const arrow = row.dataset.arrow || 'binds';
    const searchText = row.dataset.search || '';

    const typeMatch = activeFilters.has(arrow);
    const searchMatch = !searchQuery || searchText.includes(searchQuery);

    const shouldShow = typeMatch && searchMatch;
    row.style.display = shouldShow ? '' : 'none';

    if (shouldShow) visibleCount++;
  });

  updateFilterResults(visibleCount, functionRows.length);
}

function updateFilterResults(visible, total) {
  const resultsDiv = document.getElementById('filter-results');
  if (!resultsDiv) return;

  if (visible === undefined) {
    resultsDiv.textContent = '';
    return;
  }

  if (total === 0) {
    resultsDiv.textContent = '';
    resultsDiv.style.color = '#6b7280';
    return;
  }

  if (visible === 0) {
    resultsDiv.textContent = 'No interactions match current filters';
    resultsDiv.style.color = '#dc2626';
  } else if (visible === total) {
    resultsDiv.textContent = '';
  } else {
    resultsDiv.textContent = `Showing ${visible} of ${total} interactions`;
    resultsDiv.style.color = '#6b7280';
  }
}

/* ===== View Mode Switching (NET vs DIRECT) ===== */
let currentInteractionMode = 'direct';  // 'direct', 'net', 'both'

/**
 * Switch between interaction view modes (DIRECT / NET / BOTH)
 */
function switchInteractionMode(mode) {
  if (!['direct', 'net', 'both'].includes(mode)) {
    console.error('Invalid interaction mode:', mode);
    return;
  }

  console.log(`Switching interaction mode: ${currentInteractionMode} → ${mode}`);
  currentInteractionMode = mode;

  // Update button active states
  document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
  const activeButton = document.getElementById(`mode-${mode}`);
  if (activeButton) {
    activeButton.classList.add('active');
  }

  // Save preference to localStorage
  try {
    localStorage.setItem('interaction_view_mode', mode);
  } catch (e) {
    console.warn('Could not save view mode to localStorage:', e);
  }

  // Rebuild graph with filtered interactions
  buildInitialGraph();

  // Restart simulation gently
  if (simulation) {
    simulation.alpha(0.3).restart();
  }

  // Rebuild table view if visible
  const tableView = document.getElementById('table-view');
  if (tableView && tableView.style.display !== 'none') {
    buildTableView();
  }

  // Update counter
  updateViewModeCounter();

  console.log(`View mode switched to: ${mode}`);
}

/**
 * Get current interaction view mode
 */
function getCurrentViewMode() {
  return currentInteractionMode;
}

/**
 * Update the view mode counter display
 */
function updateViewModeCounter() {
  const counterEl = document.getElementById('view-mode-counter');
  if (!counterEl) return;

  const mode = getCurrentViewMode();
  const totalInteractions = SNAP.interactions ? SNAP.interactions.length : 0;
  const visibleInteractions = links.length;

  if (visibleInteractions === totalInteractions) {
    counterEl.textContent = '';
  } else {
    const modeLabel = mode === 'direct' ? 'DIRECT' : mode === 'net' ? 'NET' : 'ALL';
    counterEl.textContent = `${modeLabel}: ${visibleInteractions} of ${totalInteractions}`;
    counterEl.style.fontSize = '12px';
    counterEl.style.color = '#6b7280';
    counterEl.style.marginLeft = '8px';
    counterEl.style.fontWeight = '500';
  }
}

/**
 * Initialize view mode from localStorage on page load
 */
function initializeViewMode() {
  try {
    const savedMode = localStorage.getItem('interaction_view_mode') || 'direct';
    if (['direct', 'net', 'both'].includes(savedMode)) {
      currentInteractionMode = savedMode;

      // Update button states
      document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
      const activeButton = document.getElementById(`mode-${savedMode}`);
      if (activeButton) {
        activeButton.classList.add('active');
      }

      console.log(`Initialized view mode: ${savedMode}`);
    }
  } catch (e) {
    console.warn('Could not load view mode from localStorage:', e);
  }
}

/* ===== Table Sorting ===== */
let currentSortColumn = null;
let currentSortDirection = null;

function sortTable(column) {
  const tbody = document.getElementById('table-body');
  if (!tbody) return;

  const rows = Array.from(tbody.querySelectorAll('tr.function-row'));

  // Toggle sort direction
  if (currentSortColumn === column) {
    if (currentSortDirection === 'asc') {
      currentSortDirection = 'desc';
    } else if (currentSortDirection === 'desc') {
      // Third click: reset to unsorted
      currentSortColumn = null;
      currentSortDirection = null;
    } else {
      currentSortDirection = 'asc';
    }
  } else {
    currentSortColumn = column;
    currentSortDirection = 'asc';
  }

  // Update header indicators
  document.querySelectorAll('.data-table th.sortable').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
  });

  if (currentSortColumn && currentSortDirection) {
    const header = document.querySelector(`.data-table th[data-sort="${column}"]`);
    if (header) {
      header.classList.add(`sort-${currentSortDirection}`);
    }

    // Sort rows
    rows.sort((a, b) => {
      let aVal, bVal;

      switch (column) {
        case 'interaction':
          aVal = (a.querySelector('.interaction-name')?.textContent || '').trim();
          bVal = (b.querySelector('.interaction-name')?.textContent || '').trim();
          break;
        case 'function':
          aVal = (a.querySelector('.col-function .function-name')?.textContent || '').trim();
          bVal = (b.querySelector('.col-function .function-name')?.textContent || '').trim();
          break;
        case 'effect':
          aVal = (a.querySelector('.col-effect .effect-badge')?.textContent || '').trim();
          bVal = (b.querySelector('.col-effect .effect-badge')?.textContent || '').trim();
          break;
        case 'effectType':
          aVal = (a.querySelector('.col-effect-type')?.textContent || '').trim();
          bVal = (b.querySelector('.col-effect-type')?.textContent || '').trim();
          break;
        case 'mechanism':
          aVal = (a.querySelector('.col-mechanism')?.textContent || '').trim();
          bVal = (b.querySelector('.col-mechanism')?.textContent || '').trim();
          break;
        default:
          return 0;
      }

      const comparison = aVal.localeCompare(bVal, undefined, { numeric: true, sensitivity: 'base' });
      return currentSortDirection === 'asc' ? comparison : -comparison;
    });
  }

  // Re-append rows in sorted order
  rows.forEach(row => {
    // Also move the corresponding expanded row if it exists
    const expandedRow = row.nextElementSibling;
    tbody.appendChild(row);
    if (expandedRow && expandedRow.classList.contains('expanded-row')) {
      tbody.appendChild(expandedRow);
    }
  });
}

/* ===== Column Resizing ===== */
let resizingColumn = null;
let startX = 0;
let startWidth = 0;

function initColumnResizing() {
  const table = document.getElementById('interactions-table');
  if (!table) return;

  const resizeHandles = table.querySelectorAll('.resize-handle');
  resizeHandles.forEach(handle => {
    handle.addEventListener('mousedown', startResize);
  });

  document.addEventListener('mousemove', doResize);
  document.addEventListener('mouseup', stopResize);
}

function startResize(e) {
  e.preventDefault();
  e.stopPropagation();

  resizingColumn = e.target.closest('th');
  if (!resizingColumn) return;

  startX = e.pageX;
  startWidth = resizingColumn.offsetWidth;

  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';
}

function doResize(e) {
  if (!resizingColumn) return;

  const diff = e.pageX - startX;
  const newWidth = Math.max(40, startWidth + diff);

  resizingColumn.style.width = newWidth + 'px';
  resizingColumn.style.minWidth = newWidth + 'px';
}

function stopResize() {
  if (resizingColumn) {
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    resizingColumn = null;
  }
}

/* ===== Row Expansion ===== */
function toggleRowExpansion(clickedRow) {
  const isExpanded = clickedRow.dataset.expanded === 'true';

  // Find any existing expanded row
  const nextRow = clickedRow.nextElementSibling;
  const isExpandedRow = nextRow && nextRow.classList.contains('expanded-row');

  if (isExpanded) {
    // Collapse
    clickedRow.dataset.expanded = 'false';
    if (isExpandedRow) {
      nextRow.classList.remove('show');
      setTimeout(() => nextRow.remove(), 300);
    }
  } else {
    // Expand
    clickedRow.dataset.expanded = 'true';

    // Get entry data from row
    const entry = getEntryDataFromRow(clickedRow);
    if (!entry) return;

    // Create expanded row
    const expandedRow = createExpandedRow(entry);
    clickedRow.insertAdjacentElement('afterend', expandedRow);

    // Trigger animation
    setTimeout(() => expandedRow.classList.add('show'), 10);
  }
}

function getEntryDataFromRow(row) {
  const cells = row.querySelectorAll('td');
  if (cells.length < 6) return null; // Changed from 7 to 6 (we now have 6 columns)

  // We need to reconstruct the entry data from the row
  // We'll find it from the original entries using the stored data attributes
  const entries = collectFunctionEntries();
  const arrow = row.dataset.arrow;
  const searchKey = row.dataset.search;

  // Find matching entry
  const entry = entries.find(e => e.arrow === arrow && e.searchKey === searchKey);
  return entry;
}

function createExpandedRow(entry) {
  const expandedRow = document.createElement('tr');
  expandedRow.className = 'expanded-row';

  const td = document.createElement('td');
  td.colSpan = 6; // Match number of columns (reduced from 7 to 6)

  const content = document.createElement('div');
  content.className = 'expanded-content';

  // Build the expanded content - CLEAN TWO-COLUMN LAYOUT
  let html = '';

  // SECTION 1: INTERACTION DETAILS
  html += '<div class="detail-section">';
  html += '<h3 class="detail-section-header">INTERACTION DETAILS</h3>';
  html += '<div class="detail-divider"></div>';
  html += '<dl class="detail-grid">';

  // Interaction
  html += '<dt class="detail-label">Interaction:</dt>';
  html += `<dd class="detail-value">
    <span class="detail-interaction">
      ${escapeHtml(entry.source || 'Unknown')}
      <span class="detail-arrow">→</span>
      ${escapeHtml(entry.target || 'Unknown')}
    </span>
  </dd>`;

  // Function
  html += '<dt class="detail-label">Function:</dt>';
  html += `<dd class="detail-value">${escapeHtml(entry.functionLabel || 'Not specified')}</dd>`;

  // Interaction Effect (on the downstream protein)
  const interactionArrowClass = entry.interactionArrow || entry.arrow || 'binds';
  html += '<dt class="detail-label">Interaction Effect:</dt>';
  html += `<dd class="detail-value">
    <span class="detail-effect detail-effect-${interactionArrowClass}">${escapeHtml(entry.interactionEffectBadgeText || entry.effectBadgeText || 'Not specified')}</span>
    <span style="margin-left: 8px; font-size: 0.875em; color: var(--color-text-secondary);">(on ${escapeHtml(entry.interactorLabel)})</span>
  </dd>`;

  // Function Effect (on this specific function)
  const functionArrowClass = entry.functionArrow || entry.arrow || 'binds';
  html += '<dt class="detail-label">Function Effect:</dt>';
  html += `<dd class="detail-value">
    <span class="function-effect function-effect-${functionArrowClass}">${escapeHtml(entry.functionEffectBadgeText || entry.effectBadgeText || 'Not specified')}</span>
    <span style="margin-left: 8px; font-size: 0.875em; color: var(--color-text-secondary);">(on ${escapeHtml(entry.functionLabel)})</span>
  </dd>`;

  // Effect Type
  html += '<dt class="detail-label">Effect Type:</dt>';
  if (entry.effectTypeDetails && entry.effectTypeDetails.text) {
    html += `<dd class="detail-value">${escapeHtml(entry.effectTypeDetails.text)}</dd>`;
  } else {
    html += '<dd class="detail-value detail-empty">Not specified</dd>';
  }

  // Mechanism
  html += '<dt class="detail-label">Mechanism:</dt>';
  if (entry.mechanismText) {
    html += `<dd class="detail-value">${escapeHtml(entry.mechanismText)}</dd>`;
  } else {
    html += '<dd class="detail-value detail-empty">Not specified</dd>';
  }

  html += '</dl>';
  html += '</div>'; // end section

  // SECTION 2: CELLULAR CONTEXT
  html += '<div class="detail-section">';
  html += '<h3 class="detail-section-header">CELLULAR CONTEXT</h3>';
  html += '<div class="detail-divider"></div>';
  html += '<dl class="detail-grid">';

  // Cellular Process
  html += '<dt class="detail-label">Process:</dt>';
  if (entry.cellularProcess) {
    html += `<dd class="detail-value">${escapeHtml(entry.cellularProcess)}</dd>`;
  } else {
    html += '<dd class="detail-value detail-empty">Not specified</dd>';
  }

  // Specific Effects
  html += '<dt class="detail-label">Specific Effects:</dt>';
  if (entry.specificEffects && entry.specificEffects.length > 0) {
    html += '<dd class="detail-value"><ul class="detail-list">';
    entry.specificEffects.forEach(effect => {
      html += `<li>${escapeHtml(effect)}</li>`;
    });
    html += '</ul></dd>';
  } else {
    html += '<dd class="detail-value detail-empty">Not specified</dd>';
  }

  // Biological Cascade
  html += '<dt class="detail-label">Biological Cascade:</dt>';
  if (entry.biologicalCascade && entry.biologicalCascade.length > 0) {
    // Normalize: flatten all segments and split by arrow (→)
    const allSteps = [];
    entry.biologicalCascade.forEach(segment => {
      const text = (segment == null ? '' : segment).toString().trim();
      if (!text) return;
      const steps = text.split('→').map(s => s.trim()).filter(s => s.length > 0);
      allSteps.push(...steps);
    });

    if (allSteps.length > 0) {
      html += '<dd class="detail-value"><ol class="detail-list detail-list-ordered">';
      allSteps.forEach(step => {
        html += `<li>${escapeHtml(step)}</li>`;
      });
      html += '</ol></dd>';
    } else {
      html += '<dd class="detail-value detail-empty">Not specified</dd>';
    }
  } else {
    html += '<dd class="detail-value detail-empty">Not specified</dd>';
  }

  html += '</dl>';
  html += '</div>'; // end section

  // SECTION 3: EVIDENCE
  html += '<div class="detail-section">';
  html += '<h3 class="detail-section-header">EVIDENCE & PUBLICATIONS</h3>';
  html += '<div class="detail-divider"></div>';
  if (entry.evidence && entry.evidence.length > 0) {
    html += '<div class="expanded-evidence-list">';
    entry.evidence.forEach((ev, evIndex) => {
      // Determine primary link (PMID preferred, then DOI)
      const primaryLink = ev.pmid
        ? `https://pubmed.ncbi.nlm.nih.gov/${escapeHtml(ev.pmid)}`
        : (ev.doi ? `https://doi.org/${escapeHtml(ev.doi)}` : null);

      // Simplified: Remove wrapper, keep card only
      html += `<div class="expanded-evidence-card" data-evidence-link="${primaryLink || ''}" data-has-link="${primaryLink ? 'true' : 'false'}">`;

      // Title
      const title = ev.paper_title || 'Untitled Publication';
      html += `<div class="expanded-evidence-title">${escapeHtml(title)}</div>`;

      // Meta information
      html += '<div class="expanded-evidence-meta">';
      if (ev.authors) {
        html += `<div class="expanded-evidence-meta-item"><strong>Authors:</strong> ${escapeHtml(ev.authors)}</div>`;
      }
      if (ev.journal) {
        html += `<div class="expanded-evidence-meta-item"><strong>Journal:</strong> ${escapeHtml(ev.journal)}</div>`;
      }
      if (ev.year) {
        html += `<div class="expanded-evidence-meta-item"><strong>Year:</strong> ${escapeHtml(ev.year)}</div>`;
      }
      if (ev.assay) {
        html += `<div class="expanded-evidence-meta-item"><strong>Assay:</strong> ${escapeHtml(ev.assay)}</div>`;
      }
      if (ev.species) {
        html += `<div class="expanded-evidence-meta-item"><strong>Species:</strong> ${escapeHtml(ev.species)}</div>`;
      }
      html += '</div>';

      // Quote
      if (ev.relevant_quote) {
        html += `<div class="expanded-evidence-quote">${escapeHtml(ev.relevant_quote)}</div>`;
      }

      // PMIDs and DOI
      html += '<div class="expanded-evidence-pmids">';
      if (ev.pmid) {
        html += `<a href="https://pubmed.ncbi.nlm.nih.gov/${escapeHtml(ev.pmid)}" target="_blank" class="expanded-pmid-badge" onclick="event.stopPropagation();">PMID: ${escapeHtml(ev.pmid)}</a>`;
      }
      if (ev.doi) {
        html += `<a href="https://doi.org/${escapeHtml(ev.doi)}" target="_blank" class="expanded-pmid-badge" onclick="event.stopPropagation();">DOI: ${escapeHtml(ev.doi)}</a>`;
      }
      html += '</div>';

      html += '</div>'; // end evidence-card
    });
    html += '</div>';
  } else if (entry.fnData && entry.fnData.pmids && entry.fnData.pmids.length > 0) {
    // Show PMIDs even if no full evidence
    html += '<div class="expanded-evidence-pmids">';
    entry.fnData.pmids.forEach(pmid => {
      html += `<a href="https://pubmed.ncbi.nlm.nih.gov/${escapeHtml(pmid)}" target="_blank" class="expanded-pmid-badge">PMID: ${escapeHtml(pmid)}</a>`;
    });
    html += '</div>';
  } else {
    html += '<p class="detail-empty" style="margin-top: 0;">No evidence provided</p>';
  }
  html += '</div>'; // end section

  content.innerHTML = html;
  td.appendChild(content);
  expandedRow.appendChild(td);

  // Add click handlers to evidence cards after DOM insertion
  setTimeout(() => {
    const evidenceCards = content.querySelectorAll('.expanded-evidence-card[data-has-link="true"]');
    evidenceCards.forEach(card => {
      card.addEventListener('click', (e) => {
        // Don't trigger if clicking on the badge links (they have stopPropagation)
        const link = card.dataset.evidenceLink;
        if (link) {
          window.open(link, '_blank');
        }
      });
    });
  }, 50);

  return expandedRow;
}

function buildTableView() {
  const tbody = document.getElementById('table-body');
  if (!tbody) return;

  tbody.innerHTML = '';

  const entries = collectFunctionEntries();

  entries.forEach(entry => {
    const row = document.createElement('tr');
    row.className = 'function-row';
    row.dataset.arrow = entry.arrow;
    row.dataset.search = entry.searchKey;
    row.dataset.expanded = 'false';

    const displaySource = entry.source || '—';
    const displayTarget = entry.target || '—';

    // Determine direction arrow symbol and color class
    // Support both query-relative AND absolute directions
    const direction = entry.direction || 'main_to_primary';
    let arrowSymbol = '↔';
    if (direction === 'main_to_primary' || direction === 'a_to_b' || direction.includes('to_primary')) arrowSymbol = '→';
    else if (direction === 'primary_to_main' || direction === 'b_to_a' || direction.includes('to_main')) arrowSymbol = '←';

    const arrowColorClass = `interaction-arrow-${entry.arrow}`;

    // Clean mechanism text (no wrapper)
    const mechanismHtml = entry.mechanismText
      ? `<span class="mechanism-text">${escapeHtml(entry.mechanismText.toUpperCase())}</span>`
      : '<span class="muted-text">Not specified</span>';

    // Clean effect type text (no wrapper)
    const effectTypeHtml = entry.effectTypeDetails && entry.effectTypeDetails.text
      ? `<span class="effect-type-text">${escapeHtml(entry.effectTypeDetails.text)}</span>`
      : '<span class="muted-text">Not specified</span>';

    row.innerHTML = `
      <td class="col-expand"><span class="expand-icon">▼</span></td>
      <td class="col-interaction">
        <div class="interaction-cell">
          <span class="interaction-text">
            ${escapeHtml(displaySource)}
            <span class="interaction-arrow ${arrowColorClass}">${arrowSymbol}</span>
            ${escapeHtml(displayTarget)}
          </span>
          <div class="interaction-subtitle">${escapeHtml(entry.interactorLabel)}</div>
        </div>
      </td>
      <td class="col-effect">
        <div style="display: flex; flex-direction: column; gap: 4px;">
          <span class="effect-text effect-text-${entry.interactionArrow}" style="font-size: 10px;" title="Interaction effect (on protein)">${escapeHtml(entry.interactionEffectBadgeText)}</span>
          <span class="function-effect-text function-effect-text-${entry.functionArrow}" style="font-size: 10px;" title="Function effect">${escapeHtml(entry.functionEffectBadgeText)}</span>
        </div>
      </td>
      <td class="col-function">
        <span class="function-text">${escapeHtml(entry.functionLabel)}</span>
      </td>
      <td class="col-effect-type">${effectTypeHtml}</td>
      <td class="col-mechanism">${mechanismHtml}</td>
    `;

    // Add click handler for row expansion
    row.addEventListener('click', (e) => {
      // Don't toggle if clicking on a link
      if (e.target.tagName === 'A' || e.target.closest('a')) {
        return;
      }
      // Toggle expansion for any other click on the row
      toggleRowExpansion(row);
    });

    tbody.appendChild(row);
  });

  applyFilters();
}

function collectFunctionEntries() {
  const entries = [];
  // NEW: Read from live links array instead of static SNAP.interactions
  // This ensures expanded subgraph data is included in the table view
  const interactionLinks = links.filter(l => l.type === 'interaction');

  if (!SNAP.main) {
    console.warn('collectFunctionEntries: No main protein');
    return entries;
  }

  // NEW: Loop through interaction links, then their functions
  interactionLinks.forEach(link => {
    // Safe property accessor: expanded links store data in link.data, initial links store directly
    const L = link.data || link;

    // Skip interactions without functions (e.g., shared links without context-specific functions)
    const functions = L.functions || [];
    if (functions.length === 0) {
      return;
    }

    // Extract source/target IDs (handle D3 node object references)
    // D3 replaces source/target strings with node objects after simulation starts
    const source = L.semanticSource || ((link.source && link.source.id) ? link.source.id : link.source) || '';
    const target = L.semanticTarget || ((link.target && link.target.id) ? link.target.id : link.target) || '';

    // Extract interaction metadata
    const interactionArrow = L.arrow || 'binds';
    const intent = L.intent || 'binding';
    const supportSummary = L.support_summary || '';
    const direction = L.direction || 'main_to_primary';

    // Determine which protein is the "interactor" for display purposes
    // If interaction involves main protein, the other one is the interactor
    let interactorLabel = '';
    if (source === SNAP.main) {
      interactorLabel = target;
    } else if (target === SNAP.main) {
      interactorLabel = source;
    } else {
      // Shared link between two interactors - use source as display
      interactorLabel = source;
    }

    // Process each function
    functions.forEach((fn, fnIndex) => {
      if (!fn || typeof fn !== 'object') {
        console.warn('collectFunctionEntries: Invalid function data', fn);
        return;
      }

      const functionLabel = fn.function || 'Function';

      // IMPORTANT: Separate interaction effect from function effect
      // 1. Interaction Effect: Effect on the downstream PROTEIN (e.g., "ATXN3 inhibits VCP")
      // 2. Function Effect: Effect on this specific FUNCTION (e.g., "This interaction activates Autophagy")

      // Normalize interaction arrow (effect on the protein)
      const normalizedInteractionArrow = arrowKind(interactionArrow, intent, direction);

      // Normalize function arrow (effect on this specific function)
      const fnArrow = fn.arrow || interactionArrow;  // Fallback to interaction if function has no arrow
      const normalizedFunctionArrow = arrowKind(fnArrow, fn.intent || intent, direction);

      // Extract function details
      const cellularProcess = fn.cellular_process || '';
      const specificEffects = Array.isArray(fn.specific_effects) ? fn.specific_effects : [];
      const biologicalCascade = Array.isArray(fn.biological_consequence) ? fn.biological_consequence : [];
      const evidence = Array.isArray(fn.evidence) ? fn.evidence : [];
      const pmids = Array.isArray(fn.pmids) ? fn.pmids : [];

      // Get effect type details (use function arrow for function-specific details)
      const effectTypeDetails = getEffectTypeDetails(fn, normalizedFunctionArrow);

      // Get mechanism text
      const mechanismText = getMechanismText(intent);

      // Build searchable text
      const evidenceText = evidence.map(ev => [
        ev.paper_title,
        ev.authors,
        ev.journal,
        ev.year,
        ev.relevant_quote,
        ev.pmid
      ].filter(Boolean).join(' ')).join(' ');

      const searchParts = [
        source,
        target,
        interactorLabel,
        functionLabel,
        cellularProcess,
        specificEffects.join(' '),
        effectTypeDetails.text,
        mechanismText || '',
        supportSummary,
        biologicalCascade.join(' '),
        evidenceText,
        pmids.join(' ')
      ];

      // Create entry with BOTH interaction and function effects
      entries.push({
        interactorId: interactorLabel,
        interactorLabel: interactorLabel,
        source: String(source),
        target: String(target),
        direction: direction,

        // Interaction effect (on the downstream protein)
        interactionArrow: normalizedInteractionArrow,
        interactionEffectBadgeText: formatArrow(normalizedInteractionArrow),

        // Function effect (on this specific function)
        functionArrow: normalizedFunctionArrow,
        functionEffectBadgeText: formatArrow(normalizedFunctionArrow),

        // Legacy field for backward compatibility (use interactionArrow for most displays)
        arrow: normalizedInteractionArrow,
        effectBadgeText: formatArrow(normalizedInteractionArrow),

        functionLabel: functionLabel,
        cellularProcess: cellularProcess,
        specificEffects: specificEffects,
        effectTypeDetails: effectTypeDetails,
        mechanismText: mechanismText,
        biologicalCascade: biologicalCascade,
        evidence: evidence,
        fnData: fn,
        supportSummary: supportSummary,
        searchKey: searchParts.filter(Boolean).join(' ').toLowerCase()
      });
    });
  });

  return entries;
}

function renderSpecificEffects(effects) {
  if (!Array.isArray(effects) || effects.length === 0) {
    return '<span class="muted-text">Not specified</span>';
  }

  return `<div class="specific-effects-list">
    ${effects.map(effect => `<div class="specific-effect-chip">${escapeHtml(effect)}</div>`).join('')}
  </div>`;
}

function renderBiologicalCascade(steps) {
  if (!Array.isArray(steps) || steps.length === 0) {
    return '<span class="muted-text">Not specified</span>';
  }

  // Normalize: flatten all segments and split by arrows
  const allSteps = [];
  steps.forEach(segment => {
    const text = (segment == null ? '' : segment).toString().trim();
    if (!text) return;

    // Split by both arrow types (→ and \u001a) and clean each step
    const normalized = text.replace(/\u001a/g, '→');
    const stepsList = normalized.split('→').map(s => s.trim()).filter(s => s.length > 0);
    allSteps.push(...stepsList);
  });

  if (allSteps.length === 0) {
    return '<span class="muted-text">Not specified</span>';
  }

  return `<div class="biological-cascade-list">
    ${allSteps.map(step => `<div class="biological-cascade-item">${escapeHtml(step)}</div>`).join('')}
  </div>`;
}

function renderEvidenceSummary(evidence, fnData) {
  const items = Array.isArray(evidence) ? evidence.filter(Boolean) : [];
  const fnPmids = Array.isArray(fnData && fnData.pmids) ? fnData.pmids.filter(Boolean) : [];

  if (!items.length && !fnPmids.length) {
    return '<span class="muted-text">No evidence provided</span>';
  }

  if (!items.length) {
    return `<div class="table-evidence-pmids">
      ${fnPmids.map(p => `<a href="https://pubmed.ncbi.nlm.nih.gov/${escapeHtml(p)}" target="_blank" class="pmid-link">PMID: ${escapeHtml(p)}</a>`).join('')}
    </div>`;
  }

  const limited = items.slice(0, 3);
  const displayedPmids = new Set();
  const listHtml = limited.map(ev => {
    const title = escapeHtml(ev.paper_title || 'Untitled');
    const authors = ev.authors ? escapeHtml(ev.authors) : '';
    const journal = ev.journal ? escapeHtml(ev.journal) : '';
    const year = ev.year ? escapeHtml(ev.year) : '';
    const metaParts = [];
    if (authors) metaParts.push(authors);
    if (journal) metaParts.push(journal);
    if (year) metaParts.push(`(${year})`);
    const metaHtml = metaParts.length ? `<div class="table-evidence-meta">${metaParts.join(' · ')}</div>` : '';
    let pmidHtml = '';
    if (ev.pmid) {
      const safePmid = escapeHtml(ev.pmid);
      displayedPmids.add(ev.pmid);
      pmidHtml = `<div class="table-evidence-pmids"><a href="https://pubmed.ncbi.nlm.nih.gov/${safePmid}" target="_blank" class="pmid-link">PMID: ${safePmid}</a></div>`;
    }
    return `<div class="table-evidence-item">
      <div class="table-evidence-title">${title}</div>
      ${metaHtml}
      ${pmidHtml}
    </div>`;
  }).join('');

  const moreCount = items.length - limited.length;
  const extraPmids = fnPmids.filter(p => p && !displayedPmids.has(p));
  const extraPmidHtml = extraPmids.length ? `<div class="table-evidence-pmids">
    ${extraPmids.map(p => `<a href="https://pubmed.ncbi.nlm.nih.gov/${escapeHtml(p)}" target="_blank" class="pmid-link">PMID: ${escapeHtml(p)}</a>`).join('')}
  </div>` : '';
  const moreHtml = moreCount > 0 ? `<div class="table-evidence-more">+${moreCount} more sources</div>` : '';

  return `<div class="table-evidence-list">${listHtml}${extraPmidHtml}${moreHtml}</div>`;
}

function renderEffectType(details) {
  if (!details || !details.text) {
    return '<span class="muted-text">Not specified</span>';
  }

  const arrowClass = details.arrow === 'activates' || details.arrow === 'inhibits' ? details.arrow : 'binds';
  return `<div class="expanded-effect-type ${arrowClass}">
    <span class="effect-type-badge ${arrowClass}">${escapeHtml(details.text)}</span>
  </div>`;
}

function getEffectTypeDetails(fn, arrow) {
  const normalized = (arrow || '').toLowerCase();
  const arrowKey = normalized === 'activates' || normalized === 'inhibits' ? normalized : 'binds';

  let text = '';
  if (fn && fn.effect_description) {
    text = fn.effect_description;
  }

  if (!text) {
    if (arrowKey === 'activates') text = 'Function is enhanced or activated';
    else if (arrowKey === 'inhibits') text = 'Function is inhibited or disrupted';
    else text = 'Binds / interacts';
  }

  return { text, arrow: arrowKey };
}

function getMechanismText(intent) {
  if (!intent) return null;
  const value = Array.isArray(intent) ? intent.find(Boolean) : intent;
  if (!value) return null;
  const str = String(value).trim();
  if (!str) return null;
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function formatArrow(arrow) {
  if (arrow === 'activates') return 'Activates';
  if (arrow === 'inhibits') return 'Inhibits';
  return 'Binds';
}

function toPastTense(verb) {
  // Convert infinitive verb form to past tense/past participle
  // Handles common verbs used in interaction/function effects
  const v = verb.toLowerCase();

  // Direct word mappings for all common forms
  const pastTenseMap = {
    'activate': 'activated',
    'activates': 'activated',
    'inhibit': 'inhibited',
    'inhibits': 'inhibited',
    'bind': 'bound',
    'binds': 'bound',  // Irregular verb - FIXED
    'regulate': 'regulated',
    'regulates': 'regulated',
    'modulate': 'modulated',
    'modulates': 'modulated',
    'complex': 'complexed',
    'suppress': 'suppressed',
    'suppresses': 'suppressed',
    'enhance': 'enhanced',
    'enhances': 'enhanced',
    'promote': 'promoted',
    'promotes': 'promoted',
    'repress': 'repressed',
    'represses': 'repressed'
  };

  if (pastTenseMap[v]) return pastTenseMap[v];

  // Default fallback for regular verbs
  if (v.endsWith('e')) return v + 'd';
  return v + 'ed';
}

function extractSourceProteinFromChain(fn, interactorProtein) {
  // Extract the immediate upstream protein that acts on the target (interactor)
  // For chain context: [Query, A, B, Target] → returns B (acts on Target)
  // Returns the protein that directly causes the effect on interactorProtein

  if (!fn._context || fn._context.type !== 'chain') {
    // No chain context - fallback to interactor itself
    return interactorProtein;
  }

  const chainArray = fn._context.chain;
  const queryProtein = fn._context.query_protein || '';

  if (!Array.isArray(chainArray) || chainArray.length === 0) {
    return interactorProtein;
  }

  // Full chain: [Query, ...intermediates, Target]
  const fullChain = [queryProtein, ...chainArray];

  // Find the target protein in the chain
  const targetIndex = fullChain.findIndex(p => p === interactorProtein);

  if (targetIndex > 0) {
    // Return the protein immediately before target (the one acting on it)
    return fullChain[targetIndex - 1];
  }

  // Fallback: return last protein in chain before target
  return chainArray[chainArray.length - 1] || interactorProtein;
}

function buildFullChainPath(queryProtein, chainArray, linkData) {
  // Build full chain display for INDIRECT labels
  // Input: query protein + chain array from link/function metadata
  // Output: "ATF6 → SREBP2 → HMGCR"

  if (!Array.isArray(chainArray) || chainArray.length === 0) {
    // No chain - check if linkData has upstream_interactor
    if (linkData && linkData.upstream_interactor) {
      return `${escapeHtml(queryProtein)} → ${escapeHtml(linkData.upstream_interactor)} → ${escapeHtml(linkData.primary)}`;
    }
    return '';
  }

  const fullChain = [queryProtein, ...chainArray];
  return fullChain.map(p => escapeHtml(p)).join(' → ');
}

function formatDirection(dir) {
  const v = (dir || '').toLowerCase();
  // Handle both query-relative AND absolute directions
  if (v === 'bidirectional' || v === 'undirected' || v === 'both') return 'Bidirectional';
  if (v === 'primary_to_main' || v === 'b_to_a') return 'Protein → Main';
  if (v === 'main_to_primary' || v === 'a_to_b') return 'Main → Protein';
  return 'Bidirectional';
}

function renderPMIDs(pmids) {
  if (!Array.isArray(pmids) || pmids.length === 0) return '—';

  return `<div class="pmid-list">
    ${pmids.slice(0, 5).map(p =>
      `<a href="https://pubmed.ncbi.nlm.nih.gov/${escapeHtml(p)}" target="_blank" class="pmid-link">${escapeHtml(p)}</a>`
    ).join('')}
    ${pmids.length > 5 ? `<span style="color:#6b7280;font-size:12px;">+${pmids.length - 5} more</span>` : ''}
  </div>`;
}

function escapeHtml(text) {
  if (text == null) return '';
  const div = document.createElement('div');
  div.textContent = String(text);
  return div.innerHTML;
}

function escapeCsv(text) {
  if (text == null) return '';
  const str = String(text);
  // Escape quotes and wrap in quotes if contains comma, quote, or newline
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}

function toggleExportDropdown() {
  const menu = document.getElementById('export-dropdown-menu');
  if (menu) {
    menu.classList.toggle('show');
  }
}

function closeExportDropdown() {
  const menu = document.getElementById('export-dropdown-menu');
  if (menu) {
    menu.classList.remove('show');
  }
}

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
  const dropdown = document.querySelector('.export-dropdown');
  if (dropdown && !dropdown.contains(e.target)) {
    closeExportDropdown();
  }
});

function buildFunctionExportRows() {
  const header = [
    'Source',
    'Target',
    'Interaction',
    'Effect',
    'Function',
    'Cellular Process',
    'Specific Effects',
    'Effect Type',
    'Mechanism',
    'Biological Cascade',
    'Support Summary',
    'Evidence Title',
    'Authors',
    'Journal',
    'Year',
    'PMID',
    'Quote'
  ];

  const rows = [header];
  const entries = collectFunctionEntries();

  if (entries.length === 0) {
    rows.push(new Array(header.length).fill(''));
    return rows;
  }

  entries.forEach(entry => {
    const fnData = entry.fnData || {};
    const interaction = `${entry.source} -> ${entry.target}`;
    const effectLabel = entry.arrow === 'activates' ? 'Activates' : (entry.arrow === 'inhibits' ? 'Inhibits' : 'Binds');
    const cellularProcessText = entry.cellularProcess || 'Not specified';
    const specificEffectsText = entry.specificEffects.length ? entry.specificEffects.join(' | ') : 'Not specified';
    const effectTypeText = entry.effectTypeDetails.text || '';
    const mechanismText = entry.mechanismText || 'Not specified';
    const bioCascadeText = entry.biologicalCascade.length ? entry.biologicalCascade.join(' -> ') : '';
    const supportSummary = entry.supportSummary || '';
    const evidenceItems = entry.evidence.length ? entry.evidence : [null];
    const pmidFallback = Array.isArray(fnData.pmids) ? fnData.pmids.join(' | ') : '';

    evidenceItems.forEach((ev, evIndex) => {
      const pmidValue = ev && ev.pmid ? ev.pmid : pmidFallback;

      rows.push([
        entry.source,
        entry.target,
        interaction,
        effectLabel,
        entry.functionLabel,
        cellularProcessText,
        specificEffectsText,
        effectTypeText,
        mechanismText,
        evIndex === 0 ? bioCascadeText : '',  // Only show biological cascade in first evidence row
        evIndex === 0 ? supportSummary : '',  // Only show support summary in first evidence row
        ev ? (ev.paper_title || '') : '',
        ev ? (ev.authors || '') : '',
        ev ? (ev.journal || '') : '',
        ev ? (ev.year || '') : '',
        pmidValue,
        ev ? (ev.relevant_quote || '') : ''
      ]);
    });
  });

  return rows;
}

function exportToCSV() {
  const rows = buildFunctionExportRows();
  const csvContent = rows
    .map(row => row.map(escapeCsv).join(','))
    .join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${SNAP.main}_interaction_network.csv`;
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function exportToExcel() {
  if (typeof XLSX === 'undefined') {
    alert('Excel export library not loaded. Please refresh the page.');
    return;
  }

  const wb = XLSX.utils.book_new();
  const data = buildFunctionExportRows();
  const ws = XLSX.utils.aoa_to_sheet(data);
  XLSX.utils.book_append_sheet(wb, ws, 'Functions');
  XLSX.writeFile(wb, `${SNAP.main}_interaction_network.xlsx`);
}

/* ===== Re-query and Cancellation ===== */
let currentRunningJob = null;

async function requeryMainProtein() {
  if (!SNAP || !SNAP.main) {
    alert('No main protein found');
    return;
  }

  // Check if there's a running job
  if (currentRunningJob) {
    const confirmed = confirm(`A query is already running for ${currentRunningJob}. Cancel it and start a new re-query?`);
    if (confirmed) {
      await cancelCurrentJob();
      // Wait a moment for cancellation to process
      await new Promise(resolve => setTimeout(resolve, 500));
    } else {
      return;
    }
  }

  // Prompt for number of rounds
  const interactorInput = prompt('Number of interactor discovery rounds (1-8, default 1):', '1');
  if (interactorInput === null) return; // User cancelled

  const functionInput = prompt('Number of function mapping rounds (1-8, default 1):', '1');
  if (functionInput === null) return; // User cancelled

  const interactorRounds = Math.max(1, Math.min(8, parseInt(interactorInput) || 1));
  const functionRounds = Math.max(1, Math.min(8, parseInt(functionInput) || 1));

  currentRunningJob = SNAP.main;

  try {
    // Get list of current nodes to send as context
    const currentNodes = nodes
      .filter(n => n.type === 'main' || n.type === 'interactor')
      .map(n => n.id);

    // Start re-query
    const response = await fetch('/api/requery', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        protein: SNAP.main,
        current_nodes: currentNodes,
        interactor_rounds: interactorRounds,
        function_rounds: functionRounds
      })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Re-query failed');
    }

    // Start polling for status
    pollForComplete(SNAP.main, () => {
      // On complete, reload the page to show new data
      location.reload();
    });

  } catch (err) {
    console.error('Error starting re-query:', err);
    alert(`Failed to start re-query: ${err.message}`);
    currentRunningJob = null;
  }
}

async function pollForComplete(proteinName, onComplete) {
  const maxAttempts = 600; // 10 minutes max (1 check per second)
  let attempts = 0;

  const checkStatus = async () => {
    try {
      const response = await fetch(`/api/status/${proteinName}`);
      const data = await response.json();

      if (data.status === 'complete') {
        miniDone('Re-query complete! Refreshing...');
        currentRunningJob = null;
        currentJobProtein = null;
        // Reload immediately to show new data
        if (onComplete) {
          onComplete();
        } else {
          // Fallback: reload anyway
          setTimeout(() => location.reload(), 500);
        }
        return;
      } else if (data.status === 'error') {
        const errorText = typeof data.progress === 'object' ? data.progress.text : data.progress;
        miniDone(`Error: ${errorText}`);
        currentRunningJob = null;
        return;
      } else if (data.status === 'cancelled') {
        miniDone('Cancelled');
        currentRunningJob = null;
        return;
      } else if (data.status === 'processing') {
        const prog = data.progress || {};
        const text = prog.text || 'Processing...';
        const current = prog.current || 0;
        const total = prog.total || 100;
        miniProgress(text, current, total, proteinName);
      }

      // Keep polling
      attempts++;
      if (attempts < maxAttempts) {
        setTimeout(checkStatus, 1000);
      } else {
        miniDone('Timeout waiting for re-query');
        currentRunningJob = null;
      }
    } catch (err) {
      console.error('Error polling status:', err);
      miniDone('Error checking status');
      currentRunningJob = null;
    }
  };

  checkStatus();
}

/* ===== Chat Functions ===== */
// Chat state
let chatHistory = [];
let chatPending = false;
const MAX_CHAT_HISTORY = 10; // Configurable max history to send to LLM

/**
 * Build compact state snapshot for LLM context.
 * Sends only visible protein list - backend reads full data from cache JSON.
 */
function buildChatCompactState() {
  // Collect all visible proteins (main + interactors only, not function nodes)
  const visibleProteins = new Set();

  // Always include root protein (with safety check)
  const mainProtein = SNAP && SNAP.main ? SNAP.main : 'Unknown';
  if (mainProtein !== 'Unknown') {
    visibleProteins.add(mainProtein);
  }

  // Add all visible interactor proteins from nodes array (with safety check)
  if (Array.isArray(nodes)) {
    nodes.forEach(n => {
      if (n && n.id && (n.type === 'main' || n.type === 'interactor')) {
        visibleProteins.add(n.id);
      }
    });
  }

  return {
    parent: mainProtein,
    visible_proteins: Array.from(visibleProteins)
  };
}

/**
 * Render a chat message in the UI.
 */
function renderChatMessage(role, content, isError = false) {
  const messagesContainer = document.getElementById('chat-messages');
  if (!messagesContainer) return;

  const messageDiv = document.createElement('div');

  if (isError) {
    messageDiv.className = 'chat-message error-message';
  } else if (role === 'user') {
    messageDiv.className = 'chat-message user-message';
  } else if (role === 'assistant') {
    messageDiv.className = 'chat-message assistant-message';
  } else if (role === 'system') {
    messageDiv.className = 'chat-message system-message';
  }

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.textContent = content;

  messageDiv.appendChild(contentDiv);
  messagesContainer.appendChild(messageDiv);

  // Auto-scroll to bottom
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Send chat message to backend.
 */
async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send-btn');
  const sendText = document.getElementById('chat-send-text');
  const sendLoading = document.getElementById('chat-send-loading');

  if (!input || !sendBtn) return;

  const userMessage = input.value.trim();
  if (!userMessage || chatPending) return;

  // Early validation: ensure SNAP exists before starting
  if (!SNAP || !SNAP.main) {
    renderChatMessage('error', 'Error: No protein data loaded', true);
    return;
  }

  // Update UI state
  chatPending = true;
  input.value = '';
  input.disabled = true;
  sendBtn.disabled = true;
  sendText.style.display = 'none';
  sendLoading.style.display = 'inline';

  // Add user message to history and UI
  chatHistory.push({ role: 'user', content: userMessage });
  renderChatMessage('user', userMessage);

  try {
    // Build compact state for context
    const compactState = buildChatCompactState();

    // Prepare request payload
    const payload = {
      parent: SNAP.main,
      messages: chatHistory,
      state: compactState,
      max_history: MAX_CHAT_HISTORY,
    };

    // Call chat API
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      // Handle API error
      const errorMsg = data.error || `Server error (${response.status})`;
      throw new Error(errorMsg);
    }

    // Extract reply
    const reply = data.reply;
    if (!reply) {
      throw new Error('Empty response from server');
    }

    // Add assistant response to history and UI
    chatHistory.push({ role: 'assistant', content: reply });
    renderChatMessage('assistant', reply);

    // Trim chat history to prevent unbounded growth
    // Keep only the most recent MAX_CHAT_HISTORY * 2 messages (generous buffer)
    const maxClientHistory = MAX_CHAT_HISTORY * 2;
    if (chatHistory.length > maxClientHistory) {
      chatHistory = chatHistory.slice(-maxClientHistory);
    }

  } catch (error) {
    console.error('Chat error:', error);

    // Render error message
    const errorText = error.message || 'Failed to get response. Please try again.';
    renderChatMessage('error', `Error: ${errorText}`, true);

    // Remove the user message from history if request failed
    chatHistory.pop();

  } finally {
    // Reset UI state
    chatPending = false;
    input.disabled = false;
    sendBtn.disabled = false;
    sendText.style.display = 'inline';
    sendLoading.style.display = 'none';
    input.focus();
  }
}

/**
 * Handle Enter key in chat input (Shift+Enter for new line, Enter to send).
 */
function handleChatKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendChatMessage();
  }
}

// Wire up chat input keyboard handler
document.addEventListener('DOMContentLoaded', () => {
  const chatInput = document.getElementById('chat-input');
  if (chatInput) {
    chatInput.addEventListener('keydown', handleChatKeydown);
  }
});

/* Wire up */
document.addEventListener('DOMContentLoaded', () => {
  // Restore theme preference (dark mode is default)
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'light') {
    document.body.classList.remove('dark-mode');
  } else if (!savedTheme) {
    // First visit: ensure dark mode is set
    localStorage.setItem('theme', 'dark');
  }

  // Initialize view mode from localStorage
  if (typeof initializeViewMode === 'function') {
    initializeViewMode();
  }

  // Update theme toggle icon
  const isDark = document.body.classList.contains('dark-mode');
  const icon = document.getElementById('theme-icon');
  if (icon) {
    icon.textContent = isDark ? '☀️' : '🌙';
  }

  // Wire up search bar - matches index page behavior (search first, then prompt)
  const queryBtn = document.getElementById('query-button');
  const proteinInp = document.getElementById('protein-input');
  if (queryBtn && proteinInp) {
    const handleQuery = async () => {
      const p = proteinInp.value.trim();
      if (!p) {
        showNotificationMessage('<span style="color:#dc2626;">Please enter a protein name.</span>');
        return;
      }
      if (!/^[a-zA-Z0-9_-]+$/.test(p)) {
        showNotificationMessage('<span style="color:#dc2626;">Invalid format. Use only letters, numbers, hyphens, and underscores.</span>');
        return;
      }

      // Search database first (like index page)
      showNotificationMessage(`<span>Searching for ${p}...</span>`);

      try {
        const response = await fetch(`/api/search/${encodeURIComponent(p)}`);

        if (!response.ok) {
          const errorData = await response.json();
          showNotificationMessage(`<span style="color:#dc2626;">${errorData.error || 'Search failed'}</span>`);
          return;
        }

        const data = await response.json();

        if (data.status === 'found') {
          // Protein exists - navigate to it
          showNotificationMessage(`<span>Found! Loading ${p}...</span>`);
          vizJobTracker.saveToSessionStorage(); // Persist jobs before navigation
          setTimeout(() => {
            window.location.href = `/api/visualize/${encodeURIComponent(p)}?t=${Date.now()}`;
          }, 500);
        } else {
          // Not found in DB - check if query is currently running
          try {
            const statusResponse = await fetch(`/api/status/${encodeURIComponent(p)}`);

            if (statusResponse.ok) {
              const statusData = await statusResponse.json();

              if (statusData.status === 'processing') {
                // Job is running! Add to tracker (don't navigate)
                vizJobTracker.addJob(p, {});
                showNotificationMessage(`<span>Query running for ${p} (not in database yet)</span>`);
                return;
              }
            }
          } catch (e) {
            console.log('[handleQuery] No running job found for', p);
          }

          // Not found AND not running - show query prompt
          showQueryPromptViz(p);
        }
      } catch(error) {
        console.error('[handleQuery] Search failed:', error);
        showNotificationMessage('<span style="color:#dc2626;">Search failed</span>');
      }
    };
    queryBtn.addEventListener('click', handleQuery);
    proteinInp.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); handleQuery(); } });
  }

  // === CLEANUP ON PAGE UNLOAD ===
  // FIXED: Stop all polling intervals to prevent wasted requests
  window.addEventListener('beforeunload', () => {
    vizJobTracker.intervals.forEach((intervalId) => {
      clearInterval(intervalId);
    });
    vizJobTracker.intervals.clear();
    console.log('[VizJobTracker] Cleaned up all polling intervals on unload');
  });

  // === AUTO-RESUME JOB TRACKING ===
  // Check if current protein has a running job and resume tracking
  (async function checkAndResumeJob() {
    if (!SNAP || !SNAP.main) return;

    const currentProtein = SNAP.main;

    try {
      const response = await fetch(`/api/status/${encodeURIComponent(currentProtein)}`);
      if (!response.ok) return; // Protein not being queried

      const data = await response.json();

      // If job is still processing, add to tracker
      if (data.status === 'processing') {
        console.log(`[Auto-Resume] Found running job for ${currentProtein}, resuming tracking...`);

        vizJobTracker.addJob(currentProtein, {
          onComplete: () => {
            // Reload page to show updated data
            showNotificationMessage(`<span>Query complete! Reloading...</span>`);
            setTimeout(() => {
              window.location.reload();
            }, 1000);
          }
        });
      }
    } catch (error) {
      // Silently fail - protein just doesn't have a running job
      console.log(`[Auto-Resume] No running job for ${currentProtein}`);
    }

    // After checking current protein, restore other jobs from sessionStorage
    await vizJobTracker.restoreFromSessionStorage();
  })();

  // === SMART HEADER AUTO-HIDE ===
  // Solves the "hover chase" bug where panels shift as header shows/hides
  // Strategy: Delay hiding + extend hover zone to include panels
  (function initHeaderAutoHide() {
    const header = document.querySelector('.header');
    const headerTrigger = document.querySelector('.header-trigger');
    const controlsPanel = document.querySelector('.controls');
    const infoPanel = document.querySelector('.info-panel');

    if (!header || !headerTrigger) return;

    let hideTimer = null;
    let isHeaderVisible = false;

    // Check if header is in static mode (table view)
    function isStaticMode() {
      return header.classList.contains('header-static');
    }

    // Show header immediately (unless in static mode)
    function show() {
      if (isStaticMode()) return; // Don't toggle in static mode

      if (hideTimer) {
        clearTimeout(hideTimer);
        hideTimer = null;
      }
      if (!isHeaderVisible) {
        header.classList.add('header-visible');
        isHeaderVisible = true;
      }
    }

    // Hide header after delay (allows smooth mouse movement)
    function scheduleHide() {
      if (isStaticMode()) return; // Don't toggle in static mode

      if (hideTimer) clearTimeout(hideTimer);
      hideTimer = setTimeout(() => {
        header.classList.remove('header-visible');
        isHeaderVisible = false;
        hideTimer = null;
      }, 400); // 400ms grace period
    }

    // Attach hover listeners to all relevant elements
    [headerTrigger, header, controlsPanel, infoPanel].forEach(el => {
      if (!el) return;

      el.addEventListener('mouseenter', () => {
        show();
      });

      el.addEventListener('mouseleave', () => {
        scheduleHide();
      });
    });

    // Also respond to focus within header (keyboard accessibility)
    header.addEventListener('focusin', () => {
      show();
    });

    header.addEventListener('focusout', () => {
      scheduleHide();
    });
  })();

  initNetwork();
  buildTableView(); // Build initial table
  initColumnResizing(); // Initialize column resizing
  // Initialize with graph view active
  document.body.classList.add('graph-view-active');
  const container = document.querySelector('.container');
  if (container) container.classList.add('graph-active');
});
window.addEventListener('resize', ()=>{
  const el = document.getElementById('network');
  if (!el || !svg) return;
  const newWidth = el.clientWidth || width;
  const newHeight = el.clientHeight || height;
  if (newWidth) width = newWidth;
  if (newHeight) height = newHeight;
  svg.attr('width', width).attr('height', height);
  if (simulation) {
    simulation.force('center', d3.forceCenter(width / 2, height / 2));
    reheatSimulation(0.4);
  }
  scheduleFitToView(200, false);
});

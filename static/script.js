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
function calculatePercent(current, total) {
  if (typeof current !== 'number' || typeof total !== 'number') return 0;
  if (total <= 0) return 0;
  if (current >= total) return 100;
  return Math.round((current / total) * 100);
}

/**
 * Format job status into display metadata
 * @pure
 */
function formatJobStatus(status) {
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
function createJobState(protein, config) {
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
function updateJobProgress(job, progressData) {
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
function markJobComplete(job) {
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
function markJobError(job, errorText) {
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
function markJobCancelled(job) {
  return {
    ...job,
    status: 'cancelled',
    progress: {
      ...job.progress,
      text: 'Cancelled by user'
    }
  };
}

/**
 * Extract config from form inputs
 * @pure (reads DOM but doesn't mutate)
 */
function readConfigFromInputs() {
  const interactorRoundsInput = document.getElementById('interactor-rounds');
  const functionRoundsInput = document.getElementById('function-rounds');
  const maxDepthSelect = document.getElementById('max-depth');
  const skipValidationCheckbox = document.getElementById('skip-validation');
  const skipDeduplicatorCheckbox = document.getElementById('skip-deduplicator');
  const skipArrowCheckbox = document.getElementById('skip-arrow-determination');
  const skipFactCheckingCheckbox = document.getElementById('skip-fact-checking');

  if (interactorRoundsInput && functionRoundsInput) {
    return {
      interactor_rounds: parseInt(interactorRoundsInput.value) || 3,
      function_rounds: parseInt(functionRoundsInput.value) || 3,
      max_depth: maxDepthSelect ? parseInt(maxDepthSelect.value) || 3 : 3,
      skip_validation: skipValidationCheckbox ? skipValidationCheckbox.checked : false,
      skip_deduplicator: skipDeduplicatorCheckbox ? skipDeduplicatorCheckbox.checked : false,
      skip_arrow_determination: skipArrowCheckbox ? skipArrowCheckbox.checked : false,
      skip_fact_checking: skipFactCheckingCheckbox ? skipFactCheckingCheckbox.checked : false
    };
  }

  // Fallback to localStorage
  return {
    interactor_rounds: parseInt(localStorage.getItem('interactor_rounds')) || 3,
    function_rounds: parseInt(localStorage.getItem('function_rounds')) || 3,
    max_depth: parseInt(localStorage.getItem('max_depth')) || 3,
    skip_validation: localStorage.getItem('skip_validation') === 'true',
    skip_deduplicator: localStorage.getItem('skip_deduplicator') === 'true',
    skip_arrow_determination: localStorage.getItem('skip_arrow_determination') === 'true',
    skip_fact_checking: localStorage.getItem('skip_fact_checking') === 'true'
  };
}

/**
 * Save config to localStorage
 * @impure (writes to localStorage)
 */
function saveConfigToLocalStorage(config) {
  localStorage.setItem('interactor_rounds', config.interactor_rounds);
  localStorage.setItem('function_rounds', config.function_rounds);
  localStorage.setItem('max_depth', config.max_depth);
  localStorage.setItem('skip_validation', config.skip_validation);
  localStorage.setItem('skip_deduplicator', config.skip_deduplicator);
  localStorage.setItem('skip_arrow_determination', config.skip_arrow_determination);
  localStorage.setItem('skip_fact_checking', config.skip_fact_checking);
}

// ============================================================================
// IMPERATIVE SHELL - DOM Manipulation (Thin I/O Layer)
// ============================================================================

/**
 * Create a job card DOM element
 * @returns {Object} { container, bar, text, percent, removeBtn, cancelBtn }
 */
function createJobCard(protein) {
  const container = document.createElement('div');
  container.className = 'job-card';
  container.id = `job-${protein}`;

  container.innerHTML = `
    <div class="job-header">
      <span class="job-protein">${protein}</span>
      <div class="job-actions">
        <button class="job-btn job-remove" title="Remove from tracker (job continues in background)" aria-label="Remove from tracker">
          <span class="job-btn-icon">−</span>
        </button>
        <button class="job-btn job-cancel" title="Cancel job" aria-label="Cancel job">
          <span class="job-btn-icon">✕</span>
        </button>
      </div>
    </div>
    <div class="job-progress-container">
      <div class="job-progress-text">Initializing...</div>
      <div class="job-progress-percent">0%</div>
    </div>
    <div class="job-progress-bar-outer">
      <div class="job-progress-bar-inner"></div>
    </div>
  `;

  return {
    container,
    bar: container.querySelector('.job-progress-bar-inner'),
    text: container.querySelector('.job-progress-text'),
    percent: container.querySelector('.job-progress-percent'),
    removeBtn: container.querySelector('.job-remove'),
    cancelBtn: container.querySelector('.job-cancel')
  };
}

/**
 * Update job card UI with current job state
 */
function updateJobCard(elements, job) {
  if (!elements || !job) return;

  const { bar, text, percent, container } = elements;
  const progressPercent = calculatePercent(job.progress.current, job.progress.total);
  const statusInfo = formatJobStatus(job.status);

  // Update progress bar
  if (bar) {
    bar.style.width = `${progressPercent}%`;
    bar.style.backgroundColor = statusInfo.color;
  }

  // Update text
  if (text) {
    if (job.progress.current && job.progress.total) {
      text.textContent = `Step ${job.progress.current}/${job.progress.total}: ${job.progress.text}`;
    } else {
      text.textContent = job.progress.text;
    }
  }

  // Update percent
  if (percent) {
    percent.textContent = `${progressPercent}%`;
  }

  // Update container color/state
  if (container) {
    container.setAttribute('data-status', job.status);
  }
}

/**
 * Remove job card from DOM with fade animation
 */
function removeJobCard(container, callback) {
  if (!container) {
    if (callback) callback();
    return;
  }

  container.style.opacity = '0';
  container.style.transform = 'translateX(-10px)';

  setTimeout(() => {
    if (container.parentNode) {
      container.parentNode.removeChild(container);
    }
    if (callback) callback();
  }, 300);
}

/**
 * Show status message (for non-job updates)
 */
function showStatusMessage(text) {
  const statusMessage = document.getElementById('status-message');
  if (statusMessage) {
    statusMessage.style.display = 'block';
    statusMessage.innerHTML = `<p>${text}</p>`;
  }
}

/**
 * Hide status message
 */
function hideStatusMessage() {
  const statusMessage = document.getElementById('status-message');
  if (statusMessage) {
    statusMessage.style.display = 'none';
  }
}

// ============================================================================
// JOB TRACKER - Multi-Job Orchestration (Composition Layer)
// ============================================================================

class JobTracker {
  constructor(containerId) {
    this.jobs = new Map();           // protein -> job state
    this.intervals = new Map();      // protein -> intervalId
    this.uiElements = new Map();     // protein -> DOM elements
    this.container = document.getElementById(containerId);

    if (!this.container) {
      console.warn(`[JobTracker] Container #${containerId} not found. Creating fallback.`);
      this._createFallbackContainer();
    }
  }

  /**
   * Create fallback container if none exists
   */
  _createFallbackContainer() {
    const statusDisplay = document.getElementById('status-display');
    if (statusDisplay) {
      const container = document.createElement('div');
      container.id = 'job-container';
      container.className = 'job-container';
      statusDisplay.insertBefore(container, statusDisplay.firstChild);
      this.container = container;
    }
  }

  /**
   * Add a new job to tracker and start polling
   */
  addJob(protein, config) {
    // Guard: prevent duplicate jobs
    if (this.jobs.has(protein)) {
      const existingJob = this.jobs.get(protein);
      if (existingJob.status === 'processing') {
        console.warn(`[JobTracker] Job for ${protein} already running`);

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
    const job = createJobState(protein, config);
    this.jobs.set(protein, job);

    // Render UI
    this._renderJob(protein);

    // Start polling
    this._startPolling(protein);

    console.log(`[JobTracker] Added job for ${protein}`);
  }

  /**
   * Remove job from tracker (UI only, job continues in background)
   */
  removeFromTracker(protein) {
    console.log(`[JobTracker] Removing ${protein} from tracker (job continues in background)`);

    // Stop polling
    this._stopPolling(protein);

    // Remove UI
    const elements = this.uiElements.get(protein);
    if (elements) {
      removeJobCard(elements.container, () => {
        this.uiElements.delete(protein);
      });
    }

    // Remove from state
    this.jobs.delete(protein);
  }

  /**
   * Cancel job (stops backend job + removes from tracker)
   * FIXED: Stop polling BEFORE cancel request to prevent race condition
   */
  async cancelJob(protein) {
    console.log(`[JobTracker] Cancelling job for ${protein}`);

    const job = this.jobs.get(protein);
    if (!job) {
      console.warn(`[JobTracker] No job found for ${protein}`);
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
      const cancelledJob = markJobCancelled(job);
      this.jobs.set(protein, cancelledJob);

      // Update UI
      this._updateJobUI(protein);

      // Remove after delay
      setTimeout(() => {
        this.removeFromTracker(protein);
      }, 2000);

    } catch (error) {
      console.error(`[JobTracker] Failed to cancel ${protein}:`, error);

      // Re-enable cancel button on error
      if (elements && elements.cancelBtn) {
        elements.cancelBtn.disabled = false;
      }

      // Show error in UI
      const errorJob = markJobError(job, 'Failed to cancel job');
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

    const updatedJob = updateJobProgress(job, progressData);
    this.jobs.set(protein, updatedJob);
    this._updateJobUI(protein);
  }

  /**
   * Mark job as complete
   */
  completeJob(protein) {
    const job = this.jobs.get(protein);
    if (!job) return;

    const completedJob = markJobComplete(job);
    this.jobs.set(protein, completedJob);
    this._updateJobUI(protein);
    this._stopPolling(protein);

    // Navigate to visualization after brief delay
    setTimeout(() => {
      window.location.href = `/api/visualize/${encodeURIComponent(protein)}?t=${Date.now()}`;
    }, 1000);
  }

  /**
   * Mark job as error
   */
  errorJob(protein, errorText) {
    const job = this.jobs.get(protein);
    if (!job) return;

    const errorJob = markJobError(job, errorText);
    this.jobs.set(protein, errorJob);
    this._updateJobUI(protein);
    this._stopPolling(protein);

    // Auto-remove after delay
    setTimeout(() => {
      this.removeFromTracker(protein);
    }, 5000);
  }

  /**
   * Render job card in UI
   */
  _renderJob(protein) {
    if (!this.container) return;

    const job = this.jobs.get(protein);
    if (!job) return;

    // Create job card
    const elements = createJobCard(protein);
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

    updateJobCard(elements, job);
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
          console.warn(`[JobTracker] Status check failed for ${protein}`);
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
          this.completeJob(protein);
        } else if (data.status === 'cancelled' || data.status === 'cancelling') {
          const cancelledJob = markJobCancelled(job);
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
        console.error(`[JobTracker] Polling error for ${protein}:`, error);
      }
    }, 5000);

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
}

// ============================================================================
// MAIN APPLICATION LOGIC
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // --- Determine Page Context ---
  const isIndexPage = !!document.getElementById('status-display');

  // --- Early exit on viz page ---
  if (!isIndexPage) {
    return;
  }

  // --- Initialize Job Tracker ---
  const jobTracker = new JobTracker('job-container');

  // --- Get DOM elements ---
  const queryButton = document.getElementById('query-button');
  const proteinInput = document.getElementById('protein-input');
  const statusMessage = document.getElementById('status-message');

  // --- Search protein in database ---
  const searchProtein = async (proteinName) => {
    showStatusMessage(`Searching for ${proteinName}...`);

    try {
      const response = await fetch(`/api/search/${encodeURIComponent(proteinName)}`);

      if (!response.ok) {
        const errorData = await response.json();
        showStatusMessage(errorData.error || 'Search failed');
        return;
      }

      const data = await response.json();
      console.log('[DEBUG] Search result:', data);

      if (data.status === 'found') {
        // Protein exists - navigate immediately
        showStatusMessage(`Found! Loading visualization for ${proteinName}...`);
        window.location.href = `/api/visualize/${encodeURIComponent(proteinName)}?t=${Date.now()}`;
      } else {
        // Not found - show query prompt
        showQueryPrompt(proteinName);
      }
    } catch (error) {
      console.error('[ERROR] Search failed:', error);
      showStatusMessage('Failed to search database.');
    }
  };

  // --- Show query prompt ---
  const showQueryPrompt = (proteinName) => {
    const message = `
      <div style="text-align: center; padding: 20px;">
        <p style="font-size: 16px; color: #6b7280; margin-bottom: 16px;">
          Protein <strong>${proteinName}</strong> not found in database.
        </p>
        <button onclick="window.startQueryFromPrompt('${proteinName}')"
                style="padding: 10px 20px; background: #3b82f6; color: white; border: none; border-radius: 6px; font-weight: 500; cursor: pointer; font-size: 14px;">
          Start Research Query
        </button>
      </div>
    `;
    if (statusMessage) {
      statusMessage.style.display = 'block';
      statusMessage.innerHTML = message;
    }
  };

  // --- Start query ---
  const startQuery = async (proteinName) => {
    // Hide status message when starting job
    hideStatusMessage();

    // Get config from inputs
    const config = readConfigFromInputs();

    // Save to localStorage
    saveConfigToLocalStorage(config);

    console.log(`[DEBUG] Starting query for: ${proteinName}`);
    console.log(`[DEBUG] Config:`, config);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          protein: proteinName,
          ...config
        })
      });

      console.log(`[DEBUG] Response status: ${response.status}`);

      if (!response.ok) throw new Error('Server error');

      const data = await response.json();
      console.log(`[DEBUG] Response data:`, data);

      if (data.status === 'complete') {
        // Cached result - navigate immediately
        showStatusMessage(`Cached result found! Loading visualization for ${proteinName}...`);
        window.location.href = `/api/visualize/${encodeURIComponent(proteinName)}?t=${Date.now()}`;
      } else if (data.status === 'processing') {
        // Add to job tracker
        jobTracker.addJob(proteinName, config);

        // Clear input
        proteinInput.value = '';
      } else {
        showStatusMessage(`Error: ${data.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to start query:', error);
      showStatusMessage('Failed to connect to the server.');
    }
  };

  // --- Make startQuery available globally ---
  window.startQueryFromPrompt = (proteinName) => {
    startQuery(proteinName);
  };

  // --- Event Listeners ---
  if (proteinInput) {
    proteinInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        if (queryButton) queryButton.click();
      }
    });
  }

  if (queryButton) {
    queryButton.addEventListener('click', () => {
      const proteinName = proteinInput.value.trim();
      const validProteinRegex = /^[a-zA-Z0-9_-]+$/;

      if (!proteinName) {
        showStatusMessage('Please enter a protein name.');
        return;
      }

      if (!validProteinRegex.test(proteinName)) {
        showStatusMessage('Invalid format. Please use only letters, numbers, hyphens, and underscores.');
        return;
      }

      searchProtein(proteinName);
    });
  }

  // --- CLEANUP ON PAGE UNLOAD ---
  // FIXED: Stop all polling intervals to prevent wasted requests
  window.addEventListener('beforeunload', () => {
    jobTracker.intervals.forEach((intervalId) => {
      clearInterval(intervalId);
    });
    jobTracker.intervals.clear();
    console.log('[JobTracker] Cleaned up all polling intervals on unload');
  });

  // --- Restore saved config on page load ---
  const interactorRoundsInput = document.getElementById('interactor-rounds');
  const functionRoundsInput = document.getElementById('function-rounds');
  const maxDepthSelect = document.getElementById('max-depth');
  const skipValidationCheckbox = document.getElementById('skip-validation');
  const skipDeduplicatorCheckbox = document.getElementById('skip-deduplicator');
  const skipArrowCheckbox = document.getElementById('skip-arrow-determination');
  const skipFactCheckingCheckbox = document.getElementById('skip-fact-checking');

  if (interactorRoundsInput && functionRoundsInput) {
    const savedInteractor = localStorage.getItem('interactor_rounds');
    const savedFunction = localStorage.getItem('function_rounds');
    const savedMaxDepth = localStorage.getItem('max_depth');
    const savedSkipValidation = localStorage.getItem('skip_validation');
    const savedSkipDeduplicator = localStorage.getItem('skip_deduplicator');
    const savedSkipArrow = localStorage.getItem('skip_arrow_determination');
    const savedSkipFactChecking = localStorage.getItem('skip_fact_checking');

    if (savedInteractor) interactorRoundsInput.value = savedInteractor;
    if (savedFunction) functionRoundsInput.value = savedFunction;
    if (savedMaxDepth && maxDepthSelect) maxDepthSelect.value = savedMaxDepth;
    if (skipValidationCheckbox) skipValidationCheckbox.checked = (savedSkipValidation === 'true');
    if (skipDeduplicatorCheckbox) skipDeduplicatorCheckbox.checked = (savedSkipDeduplicator === 'true');
    if (skipArrowCheckbox) skipArrowCheckbox.checked = (savedSkipArrow === 'true');
    if (skipFactCheckingCheckbox) skipFactCheckingCheckbox.checked = (savedSkipFactChecking === 'true');
  }
});

// --- Global Helper Functions (for inline onclick handlers) ---
function setPreset(interactorRounds, functionRounds, maxDepth) {
  const interactorInput = document.getElementById('interactor-rounds');
  const functionInput = document.getElementById('function-rounds');
  const depthSelect = document.getElementById('max-depth');

  if (interactorInput) interactorInput.value = interactorRounds;
  if (functionInput) functionInput.value = functionRounds;
  if (depthSelect) depthSelect.value = maxDepth;
}

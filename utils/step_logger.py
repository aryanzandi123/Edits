"""
Comprehensive step-by-step logging for the ProPaths pipeline.

This module provides detailed logging of each pipeline step including:
- Input/output JSON data
- AI responses and metadata
- Terminal output capture
- Automatic diff generation
- AI-powered summaries

Only active when ENABLE_STEP_LOGGING=true in environment.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import difflib
import io
from contextlib import redirect_stdout, redirect_stderr


class StepLogger:
    """
    Comprehensive logger for pipeline execution steps.

    Creates organized log directories with full context for each step:
    - input.json, output.json, processed.json
    - metadata.json (tokens, costs, timing)
    - terminal_output.txt (captured console output)
    - ai_response.txt (raw AI response)
    - summary.md (AI-generated summary)
    - diff_from_previous.json (changes from previous step)
    """

    def __init__(self, protein: str, enabled: bool = None):
        """
        Initialize the step logger for a protein query.

        Args:
            protein: Protein symbol being queried
            enabled: Override enable check (default: read from env ENABLE_STEP_LOGGING)
        """
        if enabled is None:
            enabled = os.getenv('ENABLE_STEP_LOGGING', 'false').lower() == 'true'

        self.enabled = enabled
        self.protein = protein
        self.base_dir = None
        self.current_step_dir = None
        self.current_step_name = None
        self.step_counter = 0
        self.previous_output = None
        self.step_start_time = None
        self.terminal_buffer = io.StringIO()

        if self.enabled:
            self._setup_directories()

    def _setup_directories(self):
        """Create the base logging directory structure."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = Path('Logs') / self.protein / timestamp
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Step logging enabled: {self.base_dir}")

    def log_step_start(self, step_name: str, input_data: Any = None, step_type: str = "pipeline"):
        """
        Log the start of a pipeline step.

        Args:
            step_name: Name of the step (e.g., "step1a_discover", "evidence_validation")
            input_data: Input data/JSON for this step
            step_type: Type of step ("pipeline", "post_processing", "utility")
        """
        if not self.enabled:
            return

        self.step_counter += 1

        # Create step directory with proper prefix
        prefix = "step" if step_type == "pipeline" else "post"
        dir_name = f"{prefix}_{self.step_counter:02d}_{step_name}"
        self.current_step_dir = self.base_dir / dir_name
        self.current_step_dir.mkdir(parents=True, exist_ok=True)
        self.current_step_name = step_name
        self.step_start_time = time.time()

        # Reset terminal buffer for this step
        self.terminal_buffer = io.StringIO()

        # Log input data
        if input_data is not None:
            self._write_json(self.current_step_dir / 'input.json', input_data)

        print(f"  📝 Logging: {dir_name}")

    def log_terminal_output(self, message: str):
        """
        Capture terminal output for current step.

        Args:
            message: Terminal message to log
        """
        if not self.enabled or not self.current_step_dir:
            return

        self.terminal_buffer.write(message + '\n')

    def log_ai_response(self, response_text: str, metadata: Dict[str, Any] = None):
        """
        Log raw AI response and metadata.

        Args:
            response_text: Raw response text from AI
            metadata: Optional metadata (tokens, model, etc.)
        """
        if not self.enabled or not self.current_step_dir:
            return

        # Save raw AI response
        response_file = self.current_step_dir / 'ai_response.txt'
        response_file.write_text(response_text, encoding='utf-8')

        # Save metadata if provided
        if metadata:
            self._write_json(self.current_step_dir / 'ai_metadata.json', metadata)

    def log_step_complete(
        self,
        output_data: Any,
        processed_data: Any = None,
        metadata: Dict[str, Any] = None,
        generate_summary: bool = True
    ):
        """
        Log the completion of a pipeline step with all outputs.

        Args:
            output_data: Raw output from this step
            processed_data: Processed/transformed output (if different from raw)
            metadata: Step metadata (tokens, costs, timing, counts)
            generate_summary: Whether to generate AI summary (default True)
        """
        if not self.enabled or not self.current_step_dir:
            return

        # Calculate elapsed time
        elapsed = time.time() - self.step_start_time if self.step_start_time else 0

        # Save output data
        self._write_json(self.current_step_dir / 'output.json', output_data)

        # Save processed data if different
        if processed_data is not None:
            self._write_json(self.current_step_dir / 'processed.json', processed_data)

        # Save terminal output
        terminal_content = self.terminal_buffer.getvalue()
        if terminal_content:
            (self.current_step_dir / 'terminal_output.txt').write_text(
                terminal_content, encoding='utf-8'
            )

        # Enhance metadata with timing
        if metadata is None:
            metadata = {}
        metadata['step_name'] = self.current_step_name
        metadata['elapsed_seconds'] = round(elapsed, 2)
        metadata['timestamp'] = datetime.now().isoformat()

        self._write_json(self.current_step_dir / 'metadata.json', metadata)

        # Compute diff from previous step
        if self.previous_output is not None:
            self._compute_and_save_diff(self.previous_output, output_data)

        # Store for next diff
        self.previous_output = output_data

        # Generate summary
        if generate_summary:
            self._generate_summary(output_data, processed_data, metadata)

        print(f"  ✅ Logged: {self.current_step_dir.name}")

    def log_final_output(self, final_data: Any):
        """
        Log the final pipeline output.

        Args:
            final_data: Final output JSON from complete pipeline
        """
        if not self.enabled:
            return

        final_dir = self.base_dir / 'final_output'
        final_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(final_dir / 'final.json', final_data)

        # Generate final summary
        summary_content = self._create_final_summary(final_data)
        (final_dir / 'summary.md').write_text(summary_content, encoding='utf-8')

        print(f"\n✅ Complete pipeline log saved: {self.base_dir}")

    def _write_json(self, filepath: Path, data: Any):
        """Write data to JSON file with pretty formatting."""
        try:
            with filepath.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  ⚠️  Warning: Could not write {filepath.name}: {e}")

    def _compute_and_save_diff(self, previous: Any, current: Any):
        """Compute and save diff between previous and current step outputs."""
        try:
            # Convert to formatted JSON strings for diff
            prev_str = json.dumps(previous, indent=2, sort_keys=True)
            curr_str = json.dumps(current, indent=2, sort_keys=True)

            # Generate unified diff
            diff = list(difflib.unified_diff(
                prev_str.splitlines(keepends=True),
                curr_str.splitlines(keepends=True),
                fromfile='previous_step',
                tofile='current_step',
                lineterm=''
            ))

            # Save diff as text
            diff_file = self.current_step_dir / 'diff_from_previous.txt'
            diff_file.write_text(''.join(diff), encoding='utf-8')

            # Also compute high-level changes for JSON
            changes = self._extract_high_level_changes(previous, current)
            self._write_json(self.current_step_dir / 'diff_from_previous.json', changes)

        except Exception as e:
            print(f"  ⚠️  Warning: Could not compute diff: {e}")

    def _extract_high_level_changes(self, previous: Any, current: Any) -> Dict[str, Any]:
        """Extract high-level changes between two data structures."""
        changes = {
            "summary": "Changes from previous step",
            "changes": []
        }

        # Handle nested JSON structures
        if isinstance(current, dict) and isinstance(previous, dict):
            # Check for new/modified interactors
            curr_interactors = current.get('interactors', [])
            prev_interactors = previous.get('interactors', [])

            if len(curr_interactors) != len(prev_interactors):
                changes['changes'].append({
                    "type": "interactor_count_change",
                    "from": len(prev_interactors),
                    "to": len(curr_interactors),
                    "delta": len(curr_interactors) - len(prev_interactors)
                })

            # Check for new proteins
            curr_proteins = {i.get('primary') for i in curr_interactors if isinstance(i, dict)}
            prev_proteins = {i.get('primary') for i in prev_interactors if isinstance(i, dict)}
            new_proteins = curr_proteins - prev_proteins

            if new_proteins:
                changes['changes'].append({
                    "type": "new_proteins_added",
                    "count": len(new_proteins),
                    "proteins": sorted(list(new_proteins))
                })

            # Check for function changes
            curr_total_funcs = sum(
                len(i.get('functions', [])) for i in curr_interactors if isinstance(i, dict)
            )
            prev_total_funcs = sum(
                len(i.get('functions', [])) for i in prev_interactors if isinstance(i, dict)
            )

            if curr_total_funcs != prev_total_funcs:
                changes['changes'].append({
                    "type": "function_count_change",
                    "from": prev_total_funcs,
                    "to": curr_total_funcs,
                    "delta": curr_total_funcs - prev_total_funcs
                })

        return changes

    def _generate_summary(self, output_data: Any, processed_data: Any, metadata: Dict[str, Any]):
        """Generate AI-powered summary of the step."""
        try:
            # Create manual summary (AI-powered summary would require Gemini API call)
            summary_lines = [
                f"# {self.current_step_name.replace('_', ' ').title()}",
                "",
                "## Summary",
                f"Completed {self.current_step_name} in {metadata.get('elapsed_seconds', 0):.1f}s",
                ""
            ]

            # Add key metrics
            if isinstance(output_data, dict):
                interactors = output_data.get('interactors', [])
                if interactors:
                    summary_lines.extend([
                        "## Key Metrics",
                        f"- Interactors: {len(interactors)}",
                    ])

                    total_funcs = sum(
                        len(i.get('functions', [])) for i in interactors if isinstance(i, dict)
                    )
                    if total_funcs > 0:
                        summary_lines.append(f"- Functions: {total_funcs}")

            # Add token/cost info if available
            if 'input_tokens' in metadata or 'total_tokens' in metadata:
                summary_lines.extend([
                    "",
                    "## AI Metrics",
                ])
                if 'input_tokens' in metadata:
                    summary_lines.append(
                        f"- Tokens: {metadata.get('total_tokens', 0):,} "
                        f"(input: {metadata['input_tokens']:,}, "
                        f"output: {metadata.get('output_tokens', 0):,})"
                    )
                if 'total_cost' in metadata:
                    summary_lines.append(f"- Cost: ${metadata['total_cost']:.4f}")

            # Write summary
            summary_content = '\n'.join(summary_lines)
            (self.current_step_dir / 'summary.md').write_text(
                summary_content, encoding='utf-8'
            )

        except Exception as e:
            print(f"  ⚠️  Warning: Could not generate summary: {e}")

    def _create_final_summary(self, final_data: Any) -> str:
        """Create final pipeline summary."""
        lines = [
            f"# Final Pipeline Output: {self.protein}",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
        ]

        if isinstance(final_data, dict):
            # Check for snapshot_json
            snapshot = final_data.get('snapshot_json', final_data)

            if isinstance(snapshot, dict):
                main = snapshot.get('main', self.protein)
                interactors = snapshot.get('interactors', [])

                lines.extend([
                    f"- Query protein: {main}",
                    f"- Total interactors: {len(interactors)}",
                ])

                total_funcs = sum(
                    len(i.get('functions', [])) for i in interactors if isinstance(i, dict)
                )
                lines.append(f"- Total functions: {total_funcs}")

                # List interactors
                if interactors:
                    lines.extend([
                        "",
                        "## Discovered Interactors",
                        ""
                    ])
                    for i, interactor in enumerate(interactors[:20], 1):  # Limit to first 20
                        if isinstance(interactor, dict):
                            name = interactor.get('primary', 'Unknown')
                            func_count = len(interactor.get('functions', []))
                            lines.append(f"{i}. {name} ({func_count} functions)")

                    if len(interactors) > 20:
                        lines.append(f"... and {len(interactors) - 20} more")

        lines.extend([
            "",
            "## Log Directory",
            f"`{self.base_dir}`",
        ])

        return '\n'.join(lines)


# Convenience function for checking if logging is enabled
def is_logging_enabled() -> bool:
    """Check if step logging is enabled via environment variable."""
    return os.getenv('ENABLE_STEP_LOGGING', 'false').lower() == 'true'

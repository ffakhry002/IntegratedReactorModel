"""
Progress display functions for sampling execution.
"""

from .constants import RICH_AVAILABLE

if RICH_AVAILABLE:
    from rich.table import Table
    from rich.text import Text


def create_progress_table(methods, n_runs_per_method, progress_dict):
    """Create a Rich table showing progress."""
    if not RICH_AVAILABLE:
        return None

    table = Table(title="Sampling Progress", show_header=True, header_style="bold magenta")

    # Add columns
    table.add_column("Method", style="cyan", no_wrap=True)
    for i in range(max(n_runs_per_method.values())):
        table.add_column(f"Run {i+1}", justify="center", width=12)

    # Add rows
    for method in methods:
        row = [method.replace('_', ' ').title()]
        for run_idx in range(max(n_runs_per_method.values())):
            if run_idx < n_runs_per_method[method]:
                status, value = progress_dict.get((method, run_idx), ("", "WAITING"))

                # If it's a percentage progress (during execution)
                if value == "WAITING":
                    cell = "WAITING"
                elif value == "RUNNING":
                    cell = "RUNNING"
                elif "%" in str(value):
                    cell = value  # Show percentage
                elif value.startswith("i:"):
                    # K-medoids inertia value - keep as is
                    cell = value
                elif "." in str(value) and len(str(value)) > 3:
                    # Diversity score - keep as is
                    cell = value
                else:
                    # Show final result
                    cell = value

                row.append(cell)
            else:
                row.append("—")
        table.add_row(*row)

    return table


def create_method_progress_table(methods_status):
    """Create a Rich table for method-parallel execution progress."""
    if not RICH_AVAILABLE:
        return None

    table = Table(title="Method-Parallel Execution Progress", show_header=True, header_style="bold magenta")

    # Add columns
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center", width=15)
    table.add_column("Progress", justify="center", width=20)
    table.add_column("Time", justify="right", width=10)

    # Add rows
    total_methods = len(methods_status)
    completed_methods = 0

    for method, status_info in sorted(methods_status.items()):
        row = [method.replace('_', ' ').title()]

        status = status_info.get('status', 'WAITING')
        progress = status_info.get('progress', '')
        elapsed = status_info.get('elapsed', 0)

        # Status column with color coding
        if status == 'WAITING':
            status_cell = Text("⏳ Waiting", style="dim")
        elif status == 'RUNNING':
            status_cell = Text("🔄 Running", style="bold yellow")
        elif status == 'COMPLETED':
            status_cell = Text("✓ Completed", style="bold green")
            completed_methods += 1
        else:
            status_cell = Text(status, style="red")

        row.append(status_cell)

        # Progress details
        if progress:
            row.append(Text(progress, style="cyan"))
        else:
            row.append(Text("—", style="dim"))

        # Time elapsed
        if elapsed > 0:
            row.append(Text(f"{elapsed:.1f}s", style="white"))
        else:
            row.append(Text("—", style="dim"))

        table.add_row(*row)

    # Add footer with overall progress
    overall_percentage = (completed_methods / total_methods * 100) if total_methods > 0 else 0
    table.caption = f"Overall Progress: {completed_methods}/{total_methods} methods ({overall_percentage:.0f}%)"

    return table


def create_method_parallel_table(methods, n_runs, method_status):
    """Create a Rich table for method-parallel execution showing all runs."""
    if not RICH_AVAILABLE:
        return None

    table = Table(title="Method-Parallel Execution Progress", show_header=True, header_style="bold magenta")

    # Add columns
    table.add_column("Method", style="cyan", no_wrap=True)
    for i in range(n_runs):
        table.add_column(f"Run {i+1}", justify="center", width=12)

    # Add rows
    for method in sorted(methods):
        row = [method.replace('_', ' ').title()]

        status_info = method_status.get(method, {})
        method_state = status_info.get('state', 'WAITING')
        runs_progress = status_info.get('runs', {})

        for run_idx in range(n_runs):
            if method_state == 'WAITING':
                cell = "WAITING"
            elif method_state == 'ERROR':
                cell = "ERROR"
            elif method_state == 'COMPLETED':
                # Show the final results for each run
                run_result = runs_progress.get(run_idx, {})
                # For k-medoids show inertia, for others show diversity
                if 'inertia' in run_result:
                    # K-medoids: show inertia (optimization metric)
                    cell = f"i:{run_result['inertia']:.1f}"
                elif 'diversity' in run_result:
                    # Other methods: show diversity
                    cell = f"{run_result['diversity']:.4f}"
                else:
                    cell = "✓"
            else:
                # Running - show progress for current run
                run_info = runs_progress.get(run_idx, {})
                if 'progress' in run_info:
                    cell = run_info['progress']
                elif 'status' in run_info:
                    cell = run_info['status']
                else:
                    cell = "..."

            row.append(cell)

        table.add_row(*row)

    return table

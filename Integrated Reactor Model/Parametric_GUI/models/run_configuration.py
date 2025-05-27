"""
Run Configuration Model
Manages parametric run configurations
"""


class RunConfiguration:
    """Model for managing run configurations"""

    def __init__(self):
        # Simple parameter runs
        self.simple_runs = []

        # Complex multi-parameter loop sets
        self.loop_sets = []  # List of loop sets, each set contains multiple loops

    def add_simple_run(self, run_dict):
        """Add a simple parameter run"""
        self.simple_runs.append(run_dict)

    def remove_simple_run(self, index):
        """Remove a simple run by index"""
        if 0 <= index < len(self.simple_runs):
            del self.simple_runs[index]

    def clear_simple_runs(self):
        """Clear all simple runs"""
        self.simple_runs.clear()

    def add_loop_set(self, loop_set):
        """Add a new loop set"""
        self.loop_sets.append(loop_set)

    def remove_loop_set(self, index):
        """Remove a loop set by index"""
        if 0 <= index < len(self.loop_sets):
            del self.loop_sets[index]

    def clear_loop_sets(self):
        """Clear all loop sets"""
        self.loop_sets.clear()

    def get_all_runs(self):
        """Get all configured runs (simple + complex)"""
        all_runs = []

        # Add simple runs
        if self.simple_runs:
            all_runs.extend(self.simple_runs)

        # Add complex loop runs
        for loop_set in self.loop_sets:
            loop_runs = self._generate_loop_runs(loop_set)
            all_runs.extend(loop_runs)

        return all_runs

    def _generate_loop_runs(self, loop_set):
        """Generate runs from a loop set"""
        # This will be implemented to generate all combinations
        # For now, return empty list
        return []

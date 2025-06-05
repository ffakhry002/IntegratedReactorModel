"""
K-means clustering followed by nearest neighbor selection.
FIXED: Proper Jaccard distance calculation for finding nearest neighbors to centroids

This algorithm runs k-means clustering to find centroids, then selects
the nearest actual data points to those centroids as the samples.
"""
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Optional, Tuple, Callable
from .base_algorithm import BaseAlgorithm


class KMeansNearestAlgorithm(BaseAlgorithm):
    """K-means clustering followed by nearest neighbor selection."""

    def __init__(self, n_init: int = 10, max_iter: int = 300):
        super().__init__("kmeans_nearest")
        self.n_init = n_init
        self.max_iter = max_iter
        self.last_inertia = None

    def supports_quality_metric(self) -> bool:
        """K-means has inertia as quality metric."""
        return True

    def get_quality_metric(self) -> Tuple[Optional[float], Optional[str]]:
        """Return inertia (lower is better)."""
        return self.last_inertia, "inertia"

    def select_samples(self, distance_calculator, n_samples: int,
                    n_items: int, seed: Optional[int] = None,
                    progress_callback: Optional[Callable[[str], None]] = None) -> List[int]:
        """K-means selection using provided distance calculator."""

        # Store progress callback for use in methods
        self._progress_callback = progress_callback

        # Check if we can work in feature space (geometric case)
        if hasattr(distance_calculator, 'supports_feature_space') and \
        distance_calculator.supports_feature_space:
            selected_indices = self._kmeans_in_feature_space(
                distance_calculator, n_samples, seed
            )
            # ADDED: Return both indices and cluster assignments
            return selected_indices
        else:
            # For lattice methods, convert to feature representation
            return self._kmeans_with_distance_matrix(
                distance_calculator, n_samples, n_items, seed
            )

    def _kmeans_in_feature_space(self, distance_calculator,
                            n_samples: int, seed: Optional[int]) -> List[int]:
        """K-means directly on feature matrix for geometric space."""
        X = distance_calculator.feature_matrix

        print(f"  Running k-means clustering with k={n_samples}...")

        # Choose algorithm based on dataset size
        algorithm = 'elkan' if len(X) < 10000 else 'lloyd'

        if hasattr(distance_calculator, 'name') and distance_calculator.name == 'jaccard':
            print("  Note: K-means uses Euclidean distance internally.")
            print("  For Jaccard distance, we'll find nearest neighbors using Jaccard after clustering.")

        # Report initial progress
        if self._progress_callback:
            self._progress_callback("K:0%")

        kmeans = KMeans(
            n_clusters=n_samples,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=seed,
            algorithm=algorithm
        )

        # Fit k-means
        kmeans.fit(X)
        self.last_inertia = kmeans.inertia_

        # ADDED: Save cluster assignments
        self.cluster_assignments = kmeans.labels_
        self.n_clusters = n_samples

        # Report k-means complete
        if self._progress_callback:
            self._progress_callback("K:50%")

        print(f"  K-means complete. Inertia: {self.last_inertia:.2f}")
        print(f"  Finding nearest actual configurations to centroids...")

        # Find nearest actual points to centroids
        selected_indices = []
        centroids = kmeans.cluster_centers_

        for i, centroid in enumerate(centroids):
            # Calculate distances from centroid to all points using the appropriate metric
            if hasattr(distance_calculator, 'name') and distance_calculator.name == 'jaccard':
                # FIXED: Use proper discretization for Jaccard distance
                sampler = distance_calculator.sampler

                # Discretize the centroid using the same method as the data
                centroid_discretized = sampler._discretize_features_unified(centroid.reshape(1, -1))[0]

                # Get or compute discretized features for all data points
                if hasattr(sampler, 'discretized_features_cache') and sampler.discretized_features_cache is not None:
                    X_discretized = sampler.discretized_features_cache
                else:
                    print(f"    Computing discretized features for Jaccard distance...")
                    X_discretized = sampler._discretize_features_unified(X)
                    sampler.discretized_features_cache = X_discretized

                # Calculate Jaccard distances properly
                distances = []
                for j in range(len(X)):
                    x_discrete = X_discretized[j]
                    intersection = np.sum(np.logical_and(x_discrete, centroid_discretized))
                    union = np.sum(np.logical_or(x_discrete, centroid_discretized))

                    if union == 0:
                        jaccard_dist = 0.0
                    else:
                        jaccard_dist = 1.0 - (intersection / union)

                    distances.append(jaccard_dist)

                distances = np.array(distances)

            elif hasattr(distance_calculator, 'name') and distance_calculator.name == 'manhattan':
                # Manhattan distance (L1 norm)
                distances = np.sum(np.abs(X - centroid), axis=1)
            else:
                # Default to Euclidean distance (L2 norm)
                distances = np.sqrt(np.sum((X - centroid) ** 2, axis=1))

            # Exclude already selected points
            for idx in selected_indices:
                distances[idx] = np.inf

            # Find nearest point
            nearest_idx = np.argmin(distances)
            selected_indices.append(nearest_idx)

            # Report progress for each centroid
            if self._progress_callback:
                # Progress from 50% to 90% during nearest neighbor search
                progress = 50 + (40 * (i + 1) / n_samples)
                self._progress_callback(f"K:{progress:.0f}%")

            if (i + 1) % 10 == 0 or i == n_samples - 1:
                print(f"    Found {i + 1}/{n_samples} nearest configurations")

        # Calculate actual inertia with selected points (not centroids)
        actual_inertia = self._calculate_actual_inertia(X, selected_indices, distance_calculator)
        print(f"  Actual inertia with selected configurations: {actual_inertia:.2f}")

        # Report completion
        if self._progress_callback:
            self._progress_callback("K:100%")

        return selected_indices

    def _kmeans_with_distance_matrix(self, distance_calculator,
                                    n_samples: int, n_items: int,
                                    seed: Optional[int]) -> List[int]:
        """K-means for lattice methods using MDS embedding."""
        print(f"  Converting distance matrix to feature space using MDS...")

        # First, we need to build a distance matrix
        # For efficiency, we'll sample if dataset is too large
        # if n_items > 50000:
        #     print(f"  Large dataset ({n_items} items) - using sampled MDS")
        #     # Sample subset for MDS
        #     sample_size = min(50000, n_items)
        #     sample_indices = np.random.choice(n_items, sample_size, replace=False)

        #     # Build sampled distance matrix
        #     dist_matrix = np.zeros((sample_size, sample_size))
        #     if self._progress_callback:
        #         self._progress_callback("B:0%")
        #     for i in range(sample_size):
        #         for j in range(i+1, sample_size):
        #             dist = distance_calculator.get_distance(sample_indices[i], sample_indices[j])
        #             dist_matrix[i,j] = dist
        #             dist_matrix[j,i] = dist

        #         if (i + 1) % 100 == 0:
        #             print(f"    Building distance matrix: {i + 1}/{sample_size}")
        #             if self._progress_callback:
        #                 percent = round((i + 1) / sample_size * 100, 1)
        #                 self._progress_callback(f"B:{percent}%")

        #     # Apply MDS
        #     print(f"  Applying MDS embedding...")
        #     if self._progress_callback:
        #         self._progress_callback("M:RUN")
        #     from sklearn.manifold import MDS
        #     print(f"    MDS on {sample_size}x{sample_size} distance matrix...")
        #     print(f"    Using {min(10, sample_size-1)} components...")

        #     # Use fewer iterations for faster results
        #     import time
        #     mds_start = time.time()
        #     mds = MDS(n_components=min(10, sample_size-1), dissimilarity='precomputed',
        #              random_state=seed, n_jobs=1,  # Changed from -1 to avoid parallel overhead
        #              max_iter=100,  # Reduced from default 300 for speed
        #              n_init=1)      # Single initialization for speed

        #     try:
        #         X_embedded = mds.fit_transform(dist_matrix)
        #         mds_time = time.time() - mds_start
        #         print(f"    MDS completed in {mds_time:.1f} seconds")
        #     except Exception as e:
        #         print(f"    MDS failed: {e}")
        #         print(f"    Falling back to random selection...")
        #         # Fallback: just use random selection from the sample
        #         X_embedded = np.random.rand(sample_size, 2)

        #     if self._progress_callback:
        #         self._progress_callback("M:100%")

        #     # Run k-means on embedded space
        #     print(f"  Running k-means on MDS embedding...")
        #     if self._progress_callback:
        #         self._progress_callback("K:0%")
        #     kmeans = KMeans(n_clusters=min(n_samples, sample_size), n_init=self.n_init,
        #                   max_iter=self.max_iter, random_state=seed)
        #     kmeans.fit(X_embedded)

        #     # Store the inertia from MDS space (note: this is not directly comparable to original space)
        #     self.last_inertia = kmeans.inertia_

        #     # Map back to original indices
        #     if self._progress_callback:
        #         self._progress_callback("K:100%")
        #     selected_sample_indices = self._find_nearest_to_centroids(X_embedded, kmeans.cluster_centers_)
        #     selected_indices = [sample_indices[idx] for idx in selected_sample_indices]

        # else:
        # Small dataset - can do full MDS
        print(f"  Building full distance matrix for {n_items} items...")
        dist_matrix = np.zeros((n_items, n_items))

        if self._progress_callback:
            self._progress_callback("B:0%")

        total_pairs = n_items * (n_items - 1) // 2
        computed_pairs = 0

        for i in range(n_items):
            for j in range(i+1, n_items):
                dist = distance_calculator.get_distance(i, j)
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist
                computed_pairs += 1

                if computed_pairs % 1000 == 0:
                    percent = round((computed_pairs / total_pairs) * 100, 1)
                    print(f"    Progress: {percent}% ({computed_pairs}/{total_pairs} pairs)")
                    if self._progress_callback:
                        self._progress_callback(f"B:{percent}%")

        # Apply MDS
        print(f"  Applying MDS embedding...")
        if self._progress_callback:
            self._progress_callback("M:RUN")
        from sklearn.manifold import MDS
        mds = MDS(n_components=min(10, n_items-1), dissimilarity='precomputed',
                    random_state=seed, n_jobs=-1)
        X_embedded = mds.fit_transform(dist_matrix)
        if self._progress_callback:
            self._progress_callback("M:100%")

        # Run k-means
        print(f"  Running k-means on MDS embedding...")
        if self._progress_callback:
            self._progress_callback("K:0%")
        kmeans = KMeans(n_clusters=n_samples, n_init=self.n_init,
                        max_iter=self.max_iter, random_state=seed)
        kmeans.fit(X_embedded)
        self.last_inertia = kmeans.inertia_

        # Find nearest points to centroids
        if self._progress_callback:
            self._progress_callback("K:100%")
        selected_indices = self._find_nearest_to_centroids(X_embedded, kmeans.cluster_centers_)

        print(f"  K-means complete. Selected {len(selected_indices)} configurations")

        return selected_indices

    def _find_nearest_to_centroids(self, X: np.ndarray, centroids: np.ndarray) -> List[int]:
        """Find nearest actual points to centroids."""
        selected_indices = []

        for centroid in centroids:
            # Calculate Euclidean distances
            distances = np.sqrt(np.sum((X - centroid) ** 2, axis=1))

            # Exclude already selected
            for idx in selected_indices:
                distances[idx] = np.inf

            # Find nearest
            nearest_idx = np.argmin(distances)
            selected_indices.append(nearest_idx)

        return selected_indices

    def _calculate_actual_inertia(self, X: np.ndarray, selected_indices: List[int],
                                 distance_calculator) -> float:
        """Calculate inertia using actual selected points as centers."""
        total_inertia = 0.0
        centers = X[selected_indices]

        for i, point in enumerate(X):
            if i not in selected_indices:
                # Find nearest center
                if hasattr(distance_calculator, 'name') and distance_calculator.name == 'jaccard':
                    # FIXED: Use proper discretization for Jaccard distance
                    sampler = distance_calculator.sampler

                    # Get discretized features
                    if hasattr(sampler, 'discretized_features_cache') and sampler.discretized_features_cache is not None:
                        X_discretized = sampler.discretized_features_cache
                    else:
                        X_discretized = sampler._discretize_features_unified(X)

                    point_discrete = X_discretized[i]
                    centers_discrete = X_discretized[selected_indices]

                    # Calculate Jaccard distances
                    distances = []
                    for center_discrete in centers_discrete:
                        intersection = np.sum(np.logical_and(point_discrete, center_discrete))
                        union = np.sum(np.logical_or(point_discrete, center_discrete))
                        if union == 0:
                            dist = 0.0
                        else:
                            dist = 1.0 - (intersection / union)
                        distances.append(dist)
                    distances = np.array(distances)

                elif hasattr(distance_calculator, 'name') and distance_calculator.name == 'manhattan':
                    distances = np.sum(np.abs(point - centers), axis=1)
                else:
                    distances = np.sqrt(np.sum((point - centers) ** 2, axis=1))

                total_inertia += np.min(distances)

        return total_inertia

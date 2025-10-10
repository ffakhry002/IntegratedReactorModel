"""
Flexible Neural Network Architectures for Reactor Modeling

This module provides various architecture patterns and supports heterogeneous
activation functions across layers.
"""

import torch
import torch.nn as nn
from typing import List, Union


class FlexibleNeuralNet(nn.Module):
    """
    Flexible neural network supporting:
    1. Multiple architecture patterns (rectangular, pyramidal, funnel, hourglass, etc.)
    2. Heterogeneous activation functions per layer
    3. Layer-wise dropout and batch normalization
    """

    def __init__(self, input_dim, output_dim,
                 architecture_type='rectangular',
                 base_width=100,
                 depth=2,
                 activations='relu',  # Can be single string or list
                 dropout_rate=0.0,
                 use_batch_norm=False,
                 width_multipliers=None):  # For custom architectures
        """
        Initialize flexible neural network.

        Parameters
        ----------
        input_dim : int
            Number of input features
        output_dim : int
            Number of output targets
        architecture_type : str
            Architecture pattern: 'rectangular', 'pyramidal', 'funnel',
            'hourglass', 'expanding', 'custom'
        base_width : int
            Base width for calculating layer widths
        depth : int
            Number of hidden layers
        activations : str or List[str]
            Single activation for all layers, or list of activations per layer
            Options: 'relu', 'elu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu', 'selu'
        dropout_rate : float
            Dropout probability (0.0 = no dropout)
        use_batch_norm : bool
            Whether to use batch normalization
        width_multipliers : List[float], optional
            Custom width multipliers for 'custom' architecture type
            Example: [1.0, 0.75, 0.5, 0.25] for depth=4
        """
        super(FlexibleNeuralNet, self).__init__()

        self.architecture_type = architecture_type
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Calculate layer widths based on architecture type
        self.layer_widths = self._calculate_layer_widths(
            architecture_type, base_width, depth, width_multipliers
        )

        # Parse activation functions per layer
        self.activations = self._parse_activations(activations, depth)

        # Build network
        self.network = self._build_network(input_dim, output_dim)

    def _calculate_layer_widths(self, architecture_type, base_width, depth, width_multipliers):
        """Calculate width for each hidden layer based on architecture type."""

        if architecture_type == 'rectangular':
            # Uniform width across all layers
            return [base_width] * depth

        elif architecture_type == 'pyramidal':
            # Gradually narrow: start wide, end narrow
            # Example: 400 → 300 → 200 → 100 for depth=4
            widths = []
            for i in range(depth):
                # Linear decrease
                width = base_width - int(base_width * i / (depth + 1))
                widths.append(max(50, width))  # Minimum width of 50
            return widths

        elif architecture_type == 'funnel':
            # Aggressive narrowing (encoder-style)
            # Example: 400 → 200 → 100 → 50 for depth=4
            widths = []
            for i in range(depth):
                # Exponential-like decrease
                width = int(base_width * (0.5 ** (i * 0.8)))
                widths.append(max(32, width))
            return widths

        elif architecture_type == 'hourglass':
            # Narrow in middle, wide at ends
            # Example: 400 → 200 → 100 → 200 → 400 for depth=5
            widths = []
            mid_point = depth // 2
            for i in range(depth):
                if i <= mid_point:
                    # Narrowing phase
                    width = base_width - int(base_width * 0.6 * i / (mid_point + 1))
                else:
                    # Expanding phase
                    width = base_width - int(base_width * 0.6 * (depth - i - 1) / (mid_point + 1))
                widths.append(max(50, width))
            return widths

        elif architecture_type == 'expanding':
            # Gradually widen: start narrow, end wide
            # Example: 100 → 200 → 300 → 400 for depth=4
            widths = []
            start_width = max(50, base_width // 2)
            for i in range(depth):
                width = start_width + int((base_width - start_width) * i / (depth - 1)) if depth > 1 else base_width
                widths.append(width)
            return widths

        elif architecture_type == 'bottleneck':
            # Wide-narrow-wide (autencoder-style)
            # Example: 400 → 300 → 100 → 300 → 400 for depth=5
            widths = []
            mid_point = depth // 2
            min_width = max(50, base_width // 4)
            for i in range(depth):
                if i <= mid_point:
                    # Narrowing to bottleneck
                    width = base_width - int((base_width - min_width) * i / mid_point) if mid_point > 0 else base_width
                else:
                    # Expanding from bottleneck
                    width = min_width + int((base_width - min_width) * (i - mid_point) / (depth - mid_point)) if depth > mid_point else base_width
                widths.append(width)
            return widths

        elif architecture_type == 'custom':
            # User-specified multipliers
            if width_multipliers is None or len(width_multipliers) != depth:
                raise ValueError(f"For custom architecture, width_multipliers must have length {depth}")
            return [max(32, int(base_width * mult)) for mult in width_multipliers]

        else:
            raise ValueError(f"Unknown architecture type: {architecture_type}")

    def _parse_activations(self, activations, depth):
        """Parse activation specification into list of activation modules."""

        activation_map = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'swish': nn.SiLU(),  # SiLU is PyTorch's name for Swish
        }

        if isinstance(activations, str):
            # Single activation for all layers
            activation_fn = activation_map.get(activations.lower(), nn.ReLU())
            return [activation_fn] * depth

        elif isinstance(activations, list):
            # List of activations per layer
            if len(activations) != depth:
                raise ValueError(f"Activation list length ({len(activations)}) must match depth ({depth})")

            activation_fns = []
            for act in activations:
                if isinstance(act, str):
                    activation_fns.append(activation_map.get(act.lower(), nn.ReLU()))
                else:
                    # Already an nn.Module
                    activation_fns.append(act)
            return activation_fns

        else:
            raise ValueError("activations must be string or list of strings")

    def _build_network(self, input_dim, output_dim):
        """Build the network with specified architecture and activations."""

        layers = []

        # Input layer
        prev_width = input_dim

        # Hidden layers
        for i, (width, activation) in enumerate(zip(self.layer_widths, self.activations)):
            # Linear layer
            layers.append(nn.Linear(prev_width, width))

            # Batch normalization (before activation)
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(width))

            # Activation function (layer-specific!)
            layers.append(activation)

            # Dropout (after activation)
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

            prev_width = width

        # Output layer (no activation, no dropout, no batch norm)
        layers.append(nn.Linear(prev_width, output_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

    def get_architecture_info(self):
        """Return information about the network architecture."""
        return {
            'architecture_type': self.architecture_type,
            'depth': len(self.layer_widths),
            'layer_widths': self.layer_widths,
            'activations': [act.__class__.__name__ for act in self.activations],
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_heterogeneous_activations(depth, pattern='mixed'):
    """
    Create heterogeneous activation patterns for experimentation.

    Parameters
    ----------
    depth : int
        Number of hidden layers
    pattern : str
        Activation pattern:
        - 'mixed': Alternating ReLU and ELU
        - 'progressive': ReLU → ELU → Tanh
        - 'deep_relu': ReLU for early layers, ELU for deep layers
        - 'smooth_transition': ReLU → ELU → Tanh → Sigmoid

    Returns
    -------
    List[str]
        List of activation function names
    """

    if pattern == 'mixed':
        # Alternate between ReLU and ELU
        return ['relu' if i % 2 == 0 else 'elu' for i in range(depth)]

    elif pattern == 'progressive':
        # Gradually transition from ReLU to Tanh
        activations = []
        for i in range(depth):
            if i < depth // 3:
                activations.append('relu')
            elif i < 2 * depth // 3:
                activations.append('elu')
            else:
                activations.append('tanh')
        return activations

    elif pattern == 'deep_relu':
        # ReLU for early layers (feature extraction), ELU for deeper layers (refinement)
        threshold = max(1, depth // 2)
        return ['relu' if i < threshold else 'elu' for i in range(depth)]

    elif pattern == 'smooth_transition':
        # Smooth transition: ReLU → ELU → Tanh → Sigmoid
        activations = []
        quarter = max(1, depth // 4)
        for i in range(depth):
            if i < quarter:
                activations.append('relu')
            elif i < 2 * quarter:
                activations.append('elu')
            elif i < 3 * quarter:
                activations.append('tanh')
            else:
                activations.append('tanh')  # Keep tanh instead of sigmoid for stability
        return activations

    else:
        raise ValueError(f"Unknown activation pattern: {pattern}")


# Example usage and architecture suggestions
ARCHITECTURE_PRESETS = {
    'default_rectangular': {
        'architecture_type': 'rectangular',
        'base_width': 200,
        'depth': 3,
        'activations': 'relu'
    },
    'deep_pyramidal': {
        'architecture_type': 'pyramidal',
        'base_width': 400,
        'depth': 5,
        'activations': 'elu'
    },
    'funnel_encoder': {
        'architecture_type': 'funnel',
        'base_width': 400,
        'depth': 4,
        'activations': 'relu'
    },
    'hourglass_mixed': {
        'architecture_type': 'hourglass',
        'base_width': 300,
        'depth': 5,
        'activations': create_heterogeneous_activations(5, 'mixed')
    },
    'bottleneck_progressive': {
        'architecture_type': 'bottleneck',
        'base_width': 400,
        'depth': 5,
        'activations': create_heterogeneous_activations(5, 'progressive')
    }
}

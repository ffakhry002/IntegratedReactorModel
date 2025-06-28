##### THIS IS CODE DUOS

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, GroupKFold, KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scipy.stats import uniform, randint, loguniform
import joblib
import time
from datetime import datetime
import warnings
from functools import wraps
import signal
import platform
from typing import Dict, Tuple, List, Union, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class OptimizationConfig:
    """Configuration for optimization process"""
    stage_timeout: int = 3600 * 200  # 20 hours per stage
    total_timeout: int = 7200 * 300  # 30 hours total
    default_random_iter: int = 1000
    default_bayesian_iter: int = 100
    fast_random_iter: int = 100
    fast_bayesian_iter: int = 20

config = OptimizationConfig()

# ============================================================================
# TIMEOUT HANDLING
# ============================================================================
class TimeoutException(Exception):
    """Custom exception for timeout handling"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Operation timed out")

def with_timeout(timeout_seconds):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if platform.system() != 'Windows':
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result
            else:
                print(f"   ⚠️  Windows detected - using time-based timeout monitoring")
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        print(f"   ⚠️  Function completed but took {elapsed:.1f}s (> {timeout_seconds}s timeout)")
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        raise TimeoutException(f"Operation likely timed out on Windows after {elapsed:.1f}s")
                    raise e
        return wrapper
    return decorator

# ============================================================================
# PARAMETER BOUND UTILITIES
# ============================================================================
class BoundsCalculator:
    """Utility class for calculating parameter bounds"""

    @staticmethod
    def safe_bounds(lower: float, upper: float, min_val: Optional[float] = None,
                   max_val: Optional[float] = None, ensure_different: bool = True) -> Tuple[float, float]:
        """Ensure bounds are valid with lower < upper for continuous parameters"""
        # Apply absolute bounds
        if min_val is not None:
            lower = max(min_val, lower)
            upper = max(min_val, upper)
        if max_val is not None:
            lower = min(max_val, lower)
            upper = min(max_val, upper)

        # Ensure lower < upper
        if lower >= upper:
            if upper < lower:
                lower, upper = upper, lower
            else:
                center = (lower + upper) / 2
                if abs(center) > 1e-6:
                    if center > 0:
                        lower = center * 0.9
                        upper = center * 1.1
                    else:
                        lower = center * 1.1
                        upper = center * 0.9
                else:
                    if min_val is not None and min_val > 0:
                        offset = min_val * 0.1
                    elif max_val is not None and max_val > 0:
                        offset = max_val * 0.01
                    else:
                        offset = 0.1
                    lower = center - offset
                    upper = center + offset

        # Ensure minimum difference
        if ensure_different and (upper - lower) < 1e-6:
            center = (lower + upper) / 2
            if abs(center) > 1e-6:
                lower = center * (1 - 1e-6)
                upper = center * (1 + 1e-6)
            else:
                lower = center - 1e-6
                upper = center + 1e-6

        # Final check with absolute bounds
        if min_val is not None:
            lower = max(min_val, lower)
            upper = max(min_val + 1e-6, upper)
        if max_val is not None:
            upper = min(max_val, upper)
            lower = min(max_val - 1e-6, lower)

        # Last resort safety check
        if lower >= upper:
            if min_val is not None and max_val is not None:
                if max_val - min_val > 1e-6:
                    lower = min_val
                    upper = min_val + min(1e-6, (max_val - min_val) / 2)
                else:
                    lower = min_val
                    upper = max_val
            elif min_val is not None:
                lower = min_val
                upper = min_val + 1e-6
            elif max_val is not None:
                lower = max_val - 1e-6
                upper = max_val
            else:
                center = (lower + upper) / 2
                lower = center - 1e-6
                upper = center + 1e-6

        return lower, upper

    @staticmethod
    def safe_integer_bounds(lower: int, upper: int, min_val: Optional[int] = None,
                           max_val: Optional[int] = None) -> Tuple[int, int]:
        """Ensure integer bounds are valid with lower < upper"""
        lower = int(lower)
        upper = int(upper)

        if min_val is not None and max_val is not None and min_val == max_val:
            print(f"   ⚠️  WARNING: Only one integer value allowed ({min_val}), consider using Categorical")
            return min_val, min_val

        # Apply absolute bounds
        if min_val is not None:
            lower = max(min_val, lower)
            upper = max(min_val, upper)
        if max_val is not None:
            lower = min(max_val, lower)
            upper = min(max_val, upper)

        # Ensure lower < upper
        if lower >= upper:
            if upper < lower:
                lower, upper = upper, lower
            else:
                upper = lower + 1

        # Final check
        if min_val is not None:
            lower = max(min_val, lower)
            upper = max(min_val + 1, upper)
        if max_val is not None:
            upper = min(max_val, upper)
            lower = min(max_val - 1, lower)

        return lower, upper

    @staticmethod
    def get_safe_range_20_percent(value: float, param_type: str = 'continuous',
                                 min_val: Optional[float] = None,
                                 max_val: Optional[float] = None,
                                 param_name: Optional[str] = None) -> Tuple[float, float]:
        """Helper function to get ±20% range with bounds checking and ML parameter constraints"""

        # Use ML-aware constraints if parameter name is provided
        if param_name:
            ml_min = MLParameterConstraints.get_safe_min_value(param_name, min_val)
            if ml_min is not None:
                min_val = ml_min
            value = MLParameterConstraints.validate_value(param_name, value)

        # Handle edge case: value is essentially zero
        if abs(value) < 1e-9:
            if min_val is not None and min_val > 0:
                # Parameter cannot be zero, use minimum value as starting point
                lower_ideal = min_val
                upper_ideal = min_val * 2.0  # Conservative upper bound
            elif min_val is not None and min_val == 0:
                # Parameter can be zero
                lower_ideal = 0
                if max_val is not None:
                    upper_ideal = max_val * 0.01
                else:
                    upper_ideal = 1e-6
            else:
                # No constraints given, but check ML constraints
                if param_name and param_name in MLParameterConstraints.CANNOT_BE_ZERO:
                    safe_min = MLParameterConstraints.get_safe_min_value(param_name)
                    lower_ideal = safe_min
                    upper_ideal = safe_min * 2.0
                else:
                    lower_ideal = 0
                    if max_val is not None:
                        upper_ideal = max_val * 0.01
                    else:
                        upper_ideal = 1e-6
        else:
            # Standard ±20% calculation
            if value > 0:
                lower_ideal = value * 0.8
                upper_ideal = value * 1.2
            else:
                # Negative value - swap to maintain lower < upper
                lower_ideal = value * 1.2
                upper_ideal = value * 0.8

        # Convert to integer if needed
        if param_type == 'integer':
            lower_ideal = int(round(lower_ideal))
            upper_ideal = int(round(upper_ideal))

        # Ensure lower < upper BEFORE applying constraints
        if lower_ideal >= upper_ideal:
            lower_ideal, upper_ideal = upper_ideal, lower_ideal
            if lower_ideal >= upper_ideal:
                if param_type == 'integer':
                    upper_ideal = lower_ideal + 1
                else:
                    upper_ideal = lower_ideal + 1e-6

        # Apply constraint bounds CAREFULLY
        if min_val is not None:
            if lower_ideal < min_val:
                lower_ideal = min_val
                # If upper is also below min_val, adjust it
                if upper_ideal < min_val:
                    if param_type == 'integer':
                        upper_ideal = min_val + 1
                    else:
                        upper_ideal = min_val * 1.5 if min_val > 0 else min_val + 0.1

        if max_val is not None:
            if upper_ideal > max_val:
                upper_ideal = max_val
                # If lower is also above max_val, adjust it
                if lower_ideal > max_val:
                    if param_type == 'integer':
                        lower_ideal = max_val - 1
                    else:
                        lower_ideal = max_val * 0.7 if max_val > 0 else max_val - 0.1

        # CRITICAL: Final validation for ML parameter constraints
        # Apply parameter-specific validation
        if param_name:
            lower_ideal = MLParameterConstraints.validate_value(param_name, lower_ideal)
            upper_ideal = MLParameterConstraints.validate_value(param_name, upper_ideal)

        # Final safety check - ensure valid bounds
        if lower_ideal >= upper_ideal:
            print(f"   ⚠️  WARNING: Invalid bounds calculated for {param_type} parameter {param_name or 'unknown'}")
            print(f"      Original value: {value}, min_val: {min_val}, max_val: {max_val}")
            print(f"      Calculated bounds: [{lower_ideal}, {upper_ideal}]")

            # Emergency fallback to safe bounds
            if param_type == 'integer':
                return BoundsCalculator.safe_integer_bounds(lower_ideal, upper_ideal, min_val, max_val)
            else:
                return BoundsCalculator.safe_bounds(lower_ideal, upper_ideal, min_val, max_val)

        return lower_ideal, upper_ideal

# ============================================================================
# GRID CREATION
# ============================================================================
def create_focused_grid(best_value: float, param_type: str = 'continuous',
                       min_val: Optional[float] = None, max_val: Optional[float] = None,
                       n_values: int = 5) -> List[Union[int, float]]:
    """Create a focused grid around the best value with fixed ±20% range"""
    if best_value is None:
        return []

    if param_type == 'integer':
        if best_value <= 10:
            # For small integers, use fixed offsets but respect min_val
            offsets = [-2, -1, 0, 1, 2]
            if min_val is not None:
                # Filter offsets to ensure we don't go below min_val
                valid_offsets = [o for o in offsets if best_value + o >= min_val]
                if len(valid_offsets) < 5:
                    # Add more positive offsets to get 5 values
                    extra_positive = 5 - len(valid_offsets)
                    current_offset = 2
                    while len(valid_offsets) < 5 and current_offset <= 10:
                        current_offset += 1
                        if best_value + current_offset not in [best_value + o for o in valid_offsets]:
                            valid_offsets.append(current_offset)
                raw_values = [best_value + offset for offset in valid_offsets[:5]]
            else:
                raw_values = [best_value + offset for offset in offsets]
        else:
            # For larger integers, use percentage-based approach
            factors = [0.8, 0.9, 1.0, 1.1, 1.2]
            raw_values = [int(round(best_value * factor)) for factor in factors]

            # Check for duplicates
            unique_check = sorted(list(set(raw_values)))
            if len(unique_check) < 4:
                # Fallback to fixed offsets
                raw_values = [best_value - 2, best_value - 1, best_value, best_value + 1, best_value + 2]

        raw_values = [int(val) for val in raw_values]
    else:
        # For continuous parameters, use percentage factors
        factors = [0.8, 0.9, 1.0, 1.1, 1.2]
        raw_values = [best_value * factor for factor in factors]

    # Apply bounds to raw values and prevent constraint violations
    bounded_values = []
    for val in raw_values:
        # Apply min constraint first
        if min_val is not None:
            val = max(min_val, val)
        # Apply max constraint
        if max_val is not None:
            val = min(max_val, val)
        bounded_values.append(val)

    # Remove duplicates while preserving order
    unique_values = []
    seen = set()
    for val in bounded_values:
        if val not in seen:
            unique_values.append(val)
            seen.add(val)

    unique_values = sorted(unique_values)

    if len(unique_values) >= n_values:
        # Select evenly distributed values
        indices = np.linspace(0, len(unique_values) - 1, n_values).astype(int)
        final_values = [unique_values[i] for i in indices]
    else:
        # Need more values - generate them intelligently
        if len(unique_values) == 1:
            center = unique_values[0]
            if param_type == 'integer':
                # For integers, create small variations around center
                target_values = [center-2, center-1, center, center+1, center+2]
            else:
                # For continuous, create small percentage variations
                target_values = [center*0.95, center*0.975, center, center*1.025, center*1.05]

            final_values = []
            for val in target_values:
                # Apply constraints
                if min_val is not None:
                    val = max(min_val, val)
                if max_val is not None:
                    val = min(max_val, val)
                if param_type == 'integer':
                    val = int(round(val))
                if val not in final_values:
                    final_values.append(val)

            final_values = sorted(final_values)
        else:
            # Interpolate between existing unique values
            min_unique = min(unique_values)
            max_unique = max(unique_values)

            if param_type == 'integer':
                if max_unique > min_unique:
                    final_values = []
                    step = max(1, (max_unique - min_unique) // (n_values - 1))
                    for i in range(n_values):
                        val = min_unique + i * step
                        val = min(max_unique, val)
                        if val not in final_values:
                            final_values.append(val)

                    # Fill remaining slots if needed
                    while len(final_values) < n_values and len(final_values) > 0:
                        last_val = final_values[-1]
                        if last_val < max_unique:
                            final_values.append(last_val + 1)
                        else:
                            break
                else:
                    final_values = unique_values
            else:
                # For continuous, use linear interpolation
                final_values = list(np.linspace(min_unique, max_unique, n_values))

    # Final bounds check and constraint validation
    result = []
    for val in final_values[:n_values]:  # Ensure we don't exceed n_values
        # Apply constraints one more time
        if min_val is not None:
            val = max(min_val, val)
        if max_val is not None:
            val = min(max_val, val)
        if param_type == 'integer':
            val = int(round(val))

        # Avoid duplicates
        if val not in result:
            result.append(val)

    # Ensure we have enough values and fill if needed
    result = sorted(result)
    if len(result) < n_values and min_val is not None and max_val is not None:
        if param_type == 'integer':
            # Use full range if available
            full_range = list(range(int(min_val), int(max_val) + 1))
            if len(full_range) >= n_values:
                indices = np.linspace(0, len(full_range) - 1, n_values).astype(int)
                result = [full_range[i] for i in indices]
        else:
            # Use linear spacing for continuous
            result = list(np.linspace(min_val, max_val, n_values))

    return sorted(result[:n_values])

# ============================================================================
# PARAMETER UTILITIES
# ============================================================================
_NO_DEFAULT = object()

def get_param_value(params: Dict[str, Any], base_key: str,
                   default: Any = _NO_DEFAULT, verbose: bool = False) -> Any:
    """Get parameter value checking for all possible prefixes (estimator__svr__, estimator__, svr__, direct)"""
    # Check in order of specificity (most specific first)

    # 1. Double-nested for SVM flux models: estimator__svr__C
    double_nested_key = f'estimator__svr__{base_key}'
    if double_nested_key in params:
        return params[double_nested_key]

    # 2. SVR prefix for SVM keff models: svr__C
    svr_key = f'svr__{base_key}'
    if svr_key in params:
        return params[svr_key]

    # 3. Estimator prefix for other multi-output models: estimator__C
    estimator_key = f'estimator__{base_key}'
    if estimator_key in params:
        return params[estimator_key]

    # 4. Direct key for single-output models: C
    if base_key in params:
        return params[base_key]

    # Handle default case
    if default is not _NO_DEFAULT:
        if verbose:
            print(f"   Parameter '{base_key}' not found, using default: {default}")
        return default
    else:
        available_keys = list(params.keys())
        raise KeyError(f"Parameter '{base_key}' not found. Available keys: {available_keys}")

def get_optional_param(params: Dict[str, Any], base_key: str, default: Any) -> Any:
    """Get optional parameter value silently"""
    return get_param_value(params, base_key, default, verbose=False)

# ============================================================================
# PARAMETER CONSTRAINTS SYSTEM
# ============================================================================
class MLParameterConstraints:
    """Defines constraints for ML parameters to ensure valid values"""

        # Parameters that CANNOT be zero (must be > 0)
    CANNOT_BE_ZERO = {
        'learning_rate', 'epsilon', 'learning_rate_init',
        'alpha', 'C', 'n_estimators', 'max_depth', 'min_child_weight',
        'min_samples_split', 'min_samples_leaf'
    }

    # Parameters that CAN be zero (>= 0)
    CAN_BE_ZERO = {
        'reg_alpha', 'reg_lambda', 'gamma'
    }

    # Parameters that must be positive integers
    POSITIVE_INTEGERS = {
        'n_estimators', 'max_depth', 'min_child_weight', 'degree',
        'min_samples_split', 'min_samples_leaf'
    }

    # Parameters that must be in (0, 1] range
    UNIT_INTERVAL = {
        'subsample', 'colsample_bytree', 'colsample_bylevel', 'max_features', 'max_samples'
    }

    @classmethod
    def get_safe_min_value(cls, param_name: str, original_min: Optional[float] = None) -> Optional[float]:
        """Get the safe minimum value for a parameter considering ML constraints"""
        if param_name in cls.CANNOT_BE_ZERO:
            if original_min is None or original_min <= 0:
                # Set safe minimums for parameters that cannot be zero
                safe_minimums = {
                    'learning_rate': 0.001,
                    'epsilon': 0.00005,
                    'gamma': 0.00001,
                    'learning_rate_init': 0.0001,
                    'alpha': 0.0001,
                    'C': 0.1,
                    'n_estimators': 1,
                    'max_depth': 1,
                    'min_child_weight': 1,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
                return safe_minimums.get(param_name, 0.00001)
            else:
                return max(original_min, 0.00001)
        elif param_name in cls.CAN_BE_ZERO:
            return 0.0
        elif param_name in cls.POSITIVE_INTEGERS:
            return max(1, original_min) if original_min is not None else 1
        else:
            return original_min

    @classmethod
    def validate_value(cls, param_name: str, value: float) -> float:
        """Validate and adjust a parameter value according to ML constraints"""
        if param_name in cls.CANNOT_BE_ZERO and value <= 0:
            safe_min = cls.get_safe_min_value(param_name)
            print(f"   ⚠️  WARNING: {param_name}={value} invalid, using {safe_min}")
            return safe_min
        elif param_name in cls.POSITIVE_INTEGERS and value < 1:
            print(f"   ⚠️  WARNING: {param_name}={value} invalid, using 1")
            return 1
        elif param_name in cls.UNIT_INTERVAL and (value <= 0 or value > 1):
            clamped = max(0.1, min(1.0, value))
            print(f"   ⚠️  WARNING: {param_name}={value} invalid, using {clamped}")
            return clamped
        return value

# ============================================================================
# MODEL PARAMETER HANDLERS
# ============================================================================
class ModelParameterHandler(ABC):
    """Abstract base class for model-specific parameter handling"""

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for the model"""
        pass

    @abstractmethod
    def get_fixed_params(self) -> Dict[str, Any]:
        """Get fixed parameters that are not optimized"""
        pass

    @abstractmethod
    def get_random_distributions(self, needs_wrapper: bool) -> Dict[str, Any]:
        """Get parameter distributions for random search"""
        pass

    @abstractmethod
    def create_grid_params(self, best_params: Dict[str, Any], needs_wrapper: bool) -> Dict[str, Any]:
        """Create grid search parameters"""
        pass

    @abstractmethod
    def create_bayesian_spaces(self, best_params: Dict[str, Any], needs_wrapper: bool) -> Dict[str, Any]:
        """Create Bayesian search spaces"""
        pass

class XGBoostParameterHandler(ModelParameterHandler):
    """Parameter handler for XGBoost models"""

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            # Fixed parameters (not optimized)
            'colsample_bylevel': 1.0,  # Changed from 0.8 to 1.0 as discussed
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'gamma': 0.0,
            'min_child_weight': 1,
            'random_state': 42
        }

    def get_fixed_params(self) -> Dict[str, Any]:
        """Fixed parameters that are not optimized during hyperparameter search"""
        return {
            'colsample_bylevel': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'gamma': 0.0,
            'min_child_weight': 1,
            'random_state': 42
        }

    def get_random_distributions(self, needs_wrapper: bool) -> Dict[str, Any]:
        base_params = {
            # Only optimize these 5 parameters
            'n_estimators': randint(50, 10000),
            'max_depth': randint(2, 20),
            'learning_rate': uniform(0.001, 0.499),
            'subsample': uniform(0.3, 0.7),
            'colsample_bytree': uniform(0.3, 0.7)
            # Other parameters (random_state, etc.) are fixed
        }
        if needs_wrapper:
            return {f'estimator__{k}': v for k, v in base_params.items()}
        return base_params

    def create_grid_params(self, best_params: Dict[str, Any], needs_wrapper: bool) -> Dict[str, Any]:
        prefix = 'estimator__' if needs_wrapper else ''
        param_grid = {}

        # Only optimize these 5 parameters
        param_configs = [
            ('n_estimators', 'integer', 50, 10000),
            ('max_depth', 'integer', 2, 20),
            ('learning_rate', 'continuous', 0.001, 0.5),
            ('subsample', 'continuous', 0.3, 1.0),
            ('colsample_bytree', 'continuous', 0.3, 1.0)
        ]

        for param_name, param_type, min_val, max_val in param_configs:
            value = get_param_value(best_params, param_name)
            param_grid[f'{prefix}{param_name}'] = create_focused_grid(
                value, param_type, min_val, max_val)

        return param_grid

    def create_bayesian_spaces(self, best_params: Dict[str, Any], needs_wrapper: bool) -> Dict[str, Any]:
        prefix = 'estimator__' if needs_wrapper else ''
        search_spaces = {}
        bc = BoundsCalculator()

        # Integer parameters - only optimize these 2
        int_params = [
            ('n_estimators', 50, 10000),
            ('max_depth', 2, 20)
        ]

        for param_name, min_val, max_val in int_params:
            value = get_param_value(best_params, param_name)
            lower, upper = bc.get_safe_range_20_percent(value, 'integer', min_val, max_val, param_name)
            search_spaces[f'{prefix}{param_name}'] = Integer(lower, upper)

        # Continuous parameters with log-uniform prior - only learning_rate
        value = get_param_value(best_params, 'learning_rate')
        lower, upper = bc.get_safe_range_20_percent(value, 'continuous', 0.001, 0.5, 'learning_rate')
        search_spaces[f'{prefix}learning_rate'] = Real(lower, upper, prior='log-uniform')

        # Continuous parameters with uniform prior - only these 2
        uniform_params = [
            ('subsample', 0.3, 1.0),
            ('colsample_bytree', 0.3, 1.0)
        ]

        for param_name, min_val, max_val in uniform_params:
            value = get_param_value(best_params, param_name)
            lower, upper = bc.get_safe_range_20_percent(value, 'continuous', min_val, max_val, param_name)
            search_spaces[f'{prefix}{param_name}'] = Real(lower, upper)

        return search_spaces

class RandomForestParameterHandler(ModelParameterHandler):
    """Parameter handler for Random Forest models"""

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            # Fixed parameter (not optimized)
            'max_samples': None,  # Changed from 0.8 to None as suggested
            'random_state': 42
        }

    def get_fixed_params(self) -> Dict[str, Any]:
        """Fixed parameters that are not optimized during hyperparameter search"""
        return {
            'max_samples': None,
            'random_state': 42
        }

    def get_random_distributions(self, needs_wrapper: bool) -> Dict[str, Any]:
        base_params = {
            'n_estimators': randint(50, 1500),
            'max_depth': randint(10, 40),
            'min_samples_split': randint(5, 30),
            'min_samples_leaf': randint(2, 10),
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9]
            # max_samples and random_state are fixed parameters
        }
        if needs_wrapper:
            return {f'estimator__{k}': v for k, v in base_params.items()}
        return base_params

    def create_grid_params(self, best_params: Dict[str, Any], needs_wrapper: bool) -> Dict[str, Any]:
        param_grid = {}

        # Integer parameters
        int_params = [
            ('n_estimators', 50, 1500),
            ('max_depth', 10, 40),
            ('min_samples_split', 5, 30),
            ('min_samples_leaf', 2, 10)
        ]

        for param_name, min_val, max_val in int_params:
            value = get_param_value(best_params, param_name)
            param_grid[param_name] = create_focused_grid(
                value, 'integer', min_val, max_val)

        # max_features special handling
        max_features = get_param_value(best_params, 'max_features')
        if isinstance(max_features, (int, float)):
            param_grid['max_features'] = create_focused_grid(
                max_features, 'continuous', 0.3, 0.9)
        else:
            param_grid['max_features'] = [max_features, 0.5, 0.7]

        return param_grid

    def create_bayesian_spaces(self, best_params: Dict[str, Any], needs_wrapper: bool) -> Dict[str, Any]:
        search_spaces = {}
        bc = BoundsCalculator()

        # Define parameter configurations with specific bounds
        param_configs = [
            ('n_estimators', 50, 50, 1500),
            ('min_samples_split', 2, 5, 30),
            ('min_samples_leaf', 1, 2, 10),
            ('max_depth', 3, 10, 40)
        ]

        for param_name, delta, min_val, max_val in param_configs:
            value = get_param_value(best_params, param_name)
            lower, upper = bc.safe_integer_bounds(value-delta, value+delta, min_val, max_val)
            search_spaces[param_name] = Integer(lower, upper)

        # max_features special handling
        max_features = get_param_value(best_params, 'max_features')
        if isinstance(max_features, (int, float)):
            lower, upper = bc.safe_bounds(max_features*0.8, max_features*1.2, 0.3, 0.9)
            search_spaces['max_features'] = Real(lower, upper)
        else:
            search_spaces['max_features'] = Categorical([max_features, 0.5, 0.7])

        return search_spaces

class SVMParameterHandler(ModelParameterHandler):
    """Parameter handler for SVM models"""

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'C': 10.0,
            'gamma': 0.01,
            'epsilon': 0.01,
            'kernel': 'rbf',
            'shrinking': True,      # Matches Optuna for better performance
            'max_iter': -1       # Matches Optuna
        }

    def get_fixed_params(self) -> Dict[str, Any]:
        """Fixed parameters that are not optimized during hyperparameter search"""
        return {}

    def get_random_distributions(self, needs_wrapper: bool) -> Dict[str, Any]:
        base_params = {
            'C': uniform(1.0, 999.0),
            'epsilon': loguniform(0.00005, 0.1),
            # 'kernel': ['rbf', 'poly'],
            'kernel': ['rbf'],
            'gamma': loguniform(0.00001, 0.1),
            # 'degree': randint(3, 5),  # Now allow up to degree 5 with scaling
            # 'coef0': uniform(1.0, 9.0),
            'shrinking': [True],     # Keep False for consistency with Optuna
            'max_iter': [-1],     # Fixed like Optuna
            # random_state is a fixed parameter
        }
        # For SVM, always return clean parameters since we create Pipeline with svr__ prefix
        return base_params

    def create_grid_params(self, best_params: Dict[str, Any], needs_wrapper: bool) -> Dict[str, Any]:
        # For SVM, always use clean parameters since we create Pipeline with svr__ prefix
        prefix = ''

        # Base parameters for all kernels
        base_grid = {}
        for param_name, param_type, min_val, max_val in [
            ('C', 'continuous', 1.0, 1000.0),
            ('gamma', 'continuous', 0.00001, 0.1),
            ('epsilon', 'continuous', 0.00005, 0.1)
        ]:
            value = get_param_value(best_params, param_name)
            base_grid[f'{prefix}{param_name}'] = create_focused_grid(
                value, param_type, min_val, max_val)

        # Create kernel-specific grids
        best_kernel = get_param_value(best_params, 'kernel')

        # RBF grid
        rbf_grid = base_grid.copy()
        rbf_grid[f'{prefix}kernel'] = ['rbf']

        # # Poly grid
        # poly_grid = base_grid.copy()
        # poly_grid[f'{prefix}kernel'] = ['poly']

        # degree = get_param_value(best_params, 'degree')
        # coef0 = get_param_value(best_params, 'coef0')
        # poly_grid[f'{prefix}degree'] = create_focused_grid(degree, 'integer', 3, 5)  # Allow up to degree 5
        # poly_grid[f'{prefix}coef0'] = create_focused_grid(coef0, 'continuous', 1, 10)

        # # Return appropriate grid based on best kernel
        # if best_kernel == 'rbf':
        #     return [rbf_grid, poly_grid]
        # else:
        #     return [poly_grid, rbf_grid]
        base_grid[f'{prefix}kernel'] = ['rbf']
        return base_grid

    def create_bayesian_spaces(self, best_params: Dict[str, Any], needs_wrapper: bool) -> Dict[str, Any]:
        # For SVM, always use clean parameters since we create Pipeline with svr__ prefix
        prefix = ''
        search_spaces = {}
        bc = BoundsCalculator()

        # Continuous parameters with log-uniform prior
        log_params = [
            ('C', 1.0, 1000.0),
            ('gamma', 0.00001, 0.1),
            ('epsilon', 0.00005, 0.1)
        ]

        for param_name, min_val, max_val in log_params:
            value = get_param_value(best_params, param_name)

            # Add comprehensive safety checks for each parameter
            try:
                lower, upper = bc.get_safe_range_20_percent(value, 'continuous', min_val, max_val, param_name)

                # Critical validation: ensure lower < upper before creating Real object
                if lower >= upper:
                    print(f"   ⚠️  WARNING: Invalid bounds for {param_name}: [{lower}, {upper}]")
                    print(f"      Value: {value}, min_val: {min_val}, max_val: {max_val}")

                    # Emergency fallback: use conservative range around min_val
                    if param_name == 'epsilon':
                        lower, upper = 0.00005, 0.01  # Safe epsilon range
                    elif param_name == 'gamma':
                        lower, upper = 0.00001, 0.01  # Safe gamma range
                    elif param_name == 'C':
                        lower, upper = 1.0, 1000.0     # Safe C range
                    else:
                        lower, upper = min_val, min_val * 2

                    print(f"      Using fallback bounds: [{lower}, {upper}]")

                # Final safety check
                if lower >= upper:
                    raise ValueError(f"Cannot create valid bounds for {param_name}")

                # Additional constraint validation for ML parameters
                if param_name in ['gamma', 'epsilon'] and lower <= 0:
                    print(f"   ⚠️  WARNING: {param_name} cannot be <= 0, adjusting lower bound")
                    lower = min_val
                    if upper <= lower:
                        upper = lower * 2.0

                search_spaces[f'{prefix}{param_name}'] = Real(lower, upper, prior='log-uniform')
                print(f"   ✅ {param_name}: Real({lower:.6f}, {upper:.6f})")

            except Exception as e:
                print(f"   ❌ ERROR creating bounds for {param_name}: {str(e)}")
                print(f"      Using safe default bounds")

                # Ultimate fallback to safe defaults
                if param_name == 'C':
                    search_spaces[f'{prefix}{param_name}'] = Real(1.0, 1000.0, prior='log-uniform')
                elif param_name == 'gamma':
                    search_spaces[f'{prefix}{param_name}'] = Real(0.00001, 0.1, prior='log-uniform')
                elif param_name == 'epsilon':
                    search_spaces[f'{prefix}{param_name}'] = Real(0.00005, 0.1, prior='log-uniform')

        # Kernel selection
        # search_spaces[f'{prefix}kernel'] = Categorical(['rbf', 'poly'])
        search_spaces[f'{prefix}kernel'] = Categorical(['rbf'])

        # # Poly-specific parameters with safety checks
        # try:
        #     degree = get_param_value(best_params, 'degree')
        #     coef0 = get_param_value(best_params, 'coef0')

        #     degree_lower, degree_upper = bc.safe_integer_bounds(degree-1, degree+1, 3, 5)  # Allow up to degree 5
        #     coef0_lower, coef0_upper = bc.safe_bounds(coef0*0.5, coef0*2.0, 1, 10)

        #     # Validate integer bounds
        #     if degree_lower >= degree_upper:
        #         print(f"   ⚠️  WARNING: Invalid degree bounds, using fallback")
        #         degree_lower, degree_upper = 3, 5

        #     # Validate continuous bounds
        #     if coef0_lower >= coef0_upper:
        #         print(f"   ⚠️  WARNING: Invalid coef0 bounds, using fallback")
        #         coef0_lower, coef0_upper = 1.0, 10.0

        #     search_spaces[f'{prefix}degree'] = Integer(degree_lower, degree_upper)
        #     search_spaces[f'{prefix}coef0'] = Real(coef0_lower, coef0_upper)

        #     print(f"   ✅ degree: Integer({degree_lower}, {degree_upper})")
        #     print(f"   ✅ coef0: Real({coef0_lower:.3f}, {coef0_upper:.3f})")

        # except Exception as e:
        #     print(f"   ❌ ERROR creating poly parameter bounds: {str(e)}")
        #     print(f"      Using safe default bounds")
        #     search_spaces[f'{prefix}degree'] = Integer(3, 5)  # Allow up to degree 5
        #     search_spaces[f'{prefix}coef0'] = Real(1.0, 10.0)

        return search_spaces

class NeuralNetParameterHandler(ModelParameterHandler):
    """Parameter handler for Neural Network models"""

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'hidden_layer_sizes': (100,),
            'learning_rate_init': 0.001,
            'alpha': 0.001,
            'activation': 'relu',
            'solver': 'adam',
            'random_state': 42,
            'max_iter': 1000
        }

    def get_fixed_params(self) -> Dict[str, Any]:
        """Fixed parameters that are not optimized during hyperparameter search"""
        return {
            'random_state': 42,
            'max_iter': 1000
        }

    def get_random_distributions(self, needs_wrapper: bool) -> Dict[str, Any]:
        base_params = {
            'hidden_layer_sizes': [(100,), (200,), (100,50), (200,100), (300,200,100)],
            'learning_rate_init': uniform(0.0001, 0.0099),
            'alpha': uniform(0.0001, 0.0999),
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs']
            # random_state and max_iter are fixed parameters
        }
        if needs_wrapper:
            return {f'estimator__{k}': v for k, v in base_params.items()}
        return base_params

    def create_grid_params(self, best_params: Dict[str, Any], needs_wrapper: bool) -> Dict[str, Any]:
        prefix = 'estimator__' if needs_wrapper else ''
        param_grid = {}

        # Hidden layer sizes
        layers = get_param_value(best_params, 'hidden_layer_sizes')
        layer_variations = self._generate_layer_variations(layers)
        param_grid[f'{prefix}hidden_layer_sizes'] = layer_variations[:5]

        # Continuous parameters
        for param_name, min_val, max_val in [
            ('learning_rate_init', 0.0001, 0.01),
            ('alpha', 0.0001, 0.1)
        ]:
            value = get_param_value(best_params, param_name)
            param_grid[f'{prefix}{param_name}'] = create_focused_grid(
                value, 'continuous', min_val, max_val)

        # Categorical parameters
        param_grid[f'{prefix}activation'] = [get_param_value(best_params, 'activation')]

        return param_grid

    def create_bayesian_spaces(self, best_params: Dict[str, Any], needs_wrapper: bool) -> Dict[str, Any]:
        prefix = 'estimator__' if needs_wrapper else ''
        search_spaces = {}
        bc = BoundsCalculator()

        # Continuous parameters with log-uniform prior
        for param_name, min_val, max_val in [
            ('learning_rate_init', 0.0001, 0.01),
            ('alpha', 0.0001, 0.1)
        ]:
            value = get_param_value(best_params, param_name)
            lower, upper = bc.safe_bounds(value*0.5, value*2.0, min_val, max_val)
            search_spaces[f'{prefix}{param_name}'] = Real(lower, upper, prior='log-uniform')

        # Categorical parameters
        layers = get_param_value(best_params, 'hidden_layer_sizes')
        activation = get_param_value(best_params, 'activation')

        search_spaces[f'{prefix}hidden_layer_sizes'] = Categorical([layers])
        search_spaces[f'{prefix}activation'] = Categorical([activation])

        return search_spaces

    def _generate_layer_variations(self, layers: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Generate variations of neural network architectures"""
        layer_variations = []
        if isinstance(layers, tuple):
            base_layers = list(layers)
            deltas = [-50, -25, 0, 25, 50]
            for delta in deltas:
                new_layers = tuple(max(50, min(400, l + delta)) for l in base_layers)
                if new_layers not in layer_variations:
                    layer_variations.append(new_layers)

            # Add standard architectures if needed
            standard_architectures = [(100,), (200,), (100,50), (200,100), (300,), (100,100), (200,200)]
            for arch in standard_architectures:
                if arch not in layer_variations and len(layer_variations) < 5:
                    layer_variations.append(arch)
        else:
            layer_variations = [(100,), (200,), (100,50), (200,100), (300,)]

        # Ensure we have enough variations
        for size in [150, 250, 350]:
            if len(layer_variations) >= 5:
                break
            arch = (size,)
            if arch not in layer_variations:
                layer_variations.append(arch)

        return layer_variations

# ============================================================================
# PARAMETER HANDLER FACTORY
# ============================================================================
class ParameterHandlerFactory:
    """Factory for creating model-specific parameter handlers"""

    _handlers = {
        'xgboost': XGBoostParameterHandler,
        'random_forest': RandomForestParameterHandler,
        'svm': SVMParameterHandler,
        'neural_net': NeuralNetParameterHandler
    }

    @classmethod
    def create_handler(cls, model_type: str) -> ModelParameterHandler:
        """Create appropriate parameter handler for model type"""
        handler_class = cls._handlers.get(model_type)
        if handler_class is None:
            raise ValueError(f"Unknown model type: {model_type}")
        return handler_class()

# ============================================================================
# SCORING FUNCTIONS
# ============================================================================
def create_mape_scorer(use_log_flux: bool):
    """Create custom MAPE scorer for flux models"""
    from sklearn.metrics import make_scorer

    def mape_scorer(y_true, y_pred):
        """Calculate MAPE, handling log-transformed data"""
        if use_log_flux:
            y_true_linear = 10 ** y_true
            y_pred_linear = 10 ** y_pred
        else:
            y_true_linear = y_true
            y_pred_linear = y_pred

        non_zero_values = y_true_linear[y_true_linear != 0]
        if len(non_zero_values) > 0:
            epsilon = max(1e-10, np.percentile(np.abs(non_zero_values), 5))
        else:
            epsilon = 1e-10

        if len(y_true_linear.shape) > 1:
            all_errors = []
            for i in range(len(y_true_linear)):
                for j in range(y_true_linear.shape[1]):
                    if abs(y_true_linear[i, j]) > epsilon:
                        error = abs((y_pred_linear[i, j] - y_true_linear[i, j]) / y_true_linear[i, j]) * 100
                        all_errors.append(error)
            return -np.mean(all_errors) if all_errors else -1000
        else:
            mask = np.abs(y_true_linear) > epsilon
            if np.any(mask):
                mape = np.mean(np.abs((y_pred_linear[mask] - y_true_linear[mask]) / y_true_linear[mask])) * 100
                return -mape
            else:
                return -1000

    return make_scorer(mape_scorer, greater_is_better=True)

# ============================================================================
# CROSS-VALIDATION SETUP
# ============================================================================
def setup_cross_validation(X_train: np.ndarray, y_train: np.ndarray,
                          groups: Optional[np.ndarray] = None) -> Tuple[Any, int]:
    """Set up appropriate cross-validation strategy"""
    if groups is not None:
        n_unique_groups = len(np.unique(groups))

        print(f"\nAugmentation Analysis:")
        print(f"   - Total training samples: {len(X_train)}")
        print(f"   - Total unique configs: {n_unique_groups}")
        print(f"   - Average samples per config: {len(X_train) / n_unique_groups:.1f}")

        if len(X_train) % n_unique_groups == 0:
            samples_per_config = len(X_train) // n_unique_groups
            print(f"   - Confirmed: {samples_per_config}-fold augmentation detected")
        else:
            samples_per_config = len(X_train) / n_unique_groups
            print(f"   - Warning: Non-integer augmentation: {samples_per_config:.2f} samples per config")

        if n_unique_groups < 2:
            raise ValueError(f"GroupKFold requires at least 2 unique groups, got {n_unique_groups}")

        n_splits = min(10, n_unique_groups)
        n_splits = max(2, n_splits)

        if n_splits < 10:
            print(f"   - WARNING: Only {n_unique_groups} unique groups available, using {n_splits}-fold CV")

        cv = GroupKFold(n_splits=n_splits)
        print(f"   - Using GroupKFold with {n_unique_groups} unique configurations")
        print(f"   - Actual CV folds: {n_splits}")
        print(f"   - Preventing augmentation leakage in CV")

        # Expected split sizes
        print(f"\n🎯 Expected CV Split Sizes:")
        configs_per_test_fold = n_unique_groups // n_splits
        configs_per_train_fold = n_unique_groups - configs_per_test_fold
        expected_test_samples = configs_per_test_fold * samples_per_config
        expected_train_samples = configs_per_train_fold * samples_per_config
        print(f"   - Test fold: ~{configs_per_test_fold} configs × {samples_per_config:.1f} samples = ~{expected_test_samples:.0f} test samples")
        print(f"   - Train fold: ~{configs_per_train_fold} configs × {samples_per_config:.1f} samples = ~{expected_train_samples:.0f} train samples")

        # Verify first fold
        print(f"\n🔬 Actual CV Split Verification (Fold 1):")
        for i, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train, groups)):
            train_configs = len(np.unique(groups[train_idx]))
            test_configs = len(np.unique(groups[test_idx]))
            print(f"   - Fold {i+1}: {len(train_idx)} train samples ({train_configs} configs), {len(test_idx)} test samples ({test_configs} configs)")
            print(f"   - Train samples per config: {len(train_idx) / train_configs:.1f}")
            print(f"   - Test samples per config: {len(test_idx) / test_configs:.1f}")
            break
    else:
        cv = 10
        n_splits = 10
        print(f"   - WARNING: No groups provided - may have CV leakage!")
        print(f"   - Using regular {cv}-fold cross-validation")

    return cv, n_splits

# ============================================================================
# OPTIMIZATION STAGES
# ============================================================================
class OptimizationStage:
    """Base class for optimization stages"""

    def __init__(self, stage_name: str, stage_number: int, total_stages: int):
        self.stage_name = stage_name
        self.stage_number = stage_number
        self.total_stages = total_stages

    def print_header(self):
        """Print stage header"""
        print(f"\n{'='*60}")
        print(f"STAGE {self.stage_number}/{self.total_stages}: {self.stage_name.upper()}")
        print(f"{'='*60}")

    def safe_fit_with_progress(self, search_cv, X, y, groups=None, n_splits=10):
        """Safely fit search with timeout and progress tracking"""
        try:
            print(f"\n⏱️  Starting {self.stage_name} at {datetime.now().strftime('%H:%M:%S')}...")

            stage_start_time = time.time()

            # Calculate total fits
            if hasattr(search_cv, 'n_iter'):
                total_fits = search_cv.n_iter * n_splits
            else:
                if hasattr(search_cv, 'param_grid'):
                    if isinstance(search_cv.param_grid, dict):
                        total_combinations = 1
                        for param_values in search_cv.param_grid.values():
                            total_combinations *= len(param_values)
                    else:
                        total_combinations = sum(
                            np.prod([len(v) for v in grid.values()])
                            for grid in search_cv.param_grid
                        )
                    total_fits = total_combinations * n_splits
                else:
                    total_fits = n_splits

            print(f"   Starting {total_fits} model fits...")

            # Set verbose mode for underlying estimator
            if hasattr(search_cv.estimator, 'set_params'):
                try:
                    model_type = search_cv.estimator.__class__.__name__.lower()
                    if 'xgb' in model_type:
                        search_cv.estimator.set_params(verbosity=2)
                    elif 'random' in model_type:
                        search_cv.estimator.set_params(verbose=1)
                    elif 'svm' in model_type or 'svr' in model_type:
                        search_cv.estimator.set_params(verbose=True)
                    elif 'mlp' in model_type:
                        search_cv.estimator.set_params(verbose=True)
                except Exception as e:
                    print(f"   ⚠️  Could not set verbose mode: {str(e)[:50]}")

            # Fit with timeout
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                if platform.system() != 'Windows':
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(config.stage_timeout)
                else:
                    print(f"   ⚠️  Windows: No hard timeout - monitoring execution time manually")

                try:
                    if groups is not None:
                        search_cv.fit(X, y, groups=groups)
                    else:
                        search_cv.fit(X, y)

                    elapsed = time.time() - stage_start_time

                    if platform.system() == 'Windows' and elapsed > config.stage_timeout:
                        print(f"\n⚠️  {self.stage_name} completed but took {elapsed:.1f}s (>{config.stage_timeout}s)")
                    else:
                        print(f"\n✅ {self.stage_name} completed successfully in {elapsed:.1f}s!")

                    # Extract target type from scorer
                    if hasattr(search_cv, 'scoring') and callable(search_cv.scoring):
                        print(f"   Best MAPE: {abs(search_cv.best_score_):.2f}%")
                    else:
                        print(f"   Best MSE: {abs(search_cv.best_score_):.10f}")

                    return True, search_cv.best_params_, search_cv.best_score_, search_cv

                finally:
                    if platform.system() != 'Windows':
                        signal.alarm(0)

        except TimeoutException:
            print(f"\n⏱️  {self.stage_name} timed out after {config.stage_timeout}s")
            if hasattr(search_cv, 'cv_results_'):
                scores = search_cv.cv_results_['mean_test_score']
                if len(scores) > 0:
                    best_idx = np.argmax(scores)
                    best_params = search_cv.cv_results_['params'][best_idx]
                    best_score = scores[best_idx]
                    print(f"   Using partial results: {len(scores)} combinations tested")
                    print(f"   Best score so far: {abs(best_score):.4f}")
                    return True, best_params, best_score, search_cv
            return False, {}, float('-inf'), None

        except Exception as e:
            print(f"\n❌ {self.stage_name} failed with error: {str(e)[:100]}")
            return False, {}, float('-inf'), None

class RandomSearchStage(OptimizationStage):
    """Random search optimization stage"""

    def __init__(self):
        super().__init__("Random Search", 1, 3)

    def run(self, X_train, y_train, model_class, param_distributions,
            cv, n_splits, scoring, n_jobs, groups, n_iter=1000,
            fixed_params=None, model_type=None):
        """Execute random search stage"""
        self.print_header()
        print("Exploring parameter space broadly...")
        print(f"This stage tests {n_iter} random parameter combinations")

        stage_start = time.time()

        print(f"\n📊 Random Search Parameters:")
        print(f"   - Candidates to test: {n_iter}")
        print(f"   - Cross-validation folds: {n_splits}")
        print(f"   - Total fits: {n_iter * n_splits}")
        print(f"   - Timeout: {config.stage_timeout}s")

        # Create model instance with fixed parameters
        if fixed_params:
            model = model_class(**fixed_params)
            print(f"   - Using {len(fixed_params)} fixed parameters during optimization")
        else:
            model = model_class()

        # CRITICAL FIX: Add scaling for SVM and MultiOutputRegressor for flux
        if model_type == 'svm' and not hasattr(model, 'named_steps'):
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.multioutput import MultiOutputRegressor

            # Create Pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', model)
            ])

            # Check if we need MultiOutputRegressor for flux targets
            has_multi_output_data = len(y_train.shape) > 1 and y_train.shape[1] > 1
            if has_multi_output_data:
                model = MultiOutputRegressor(pipeline)
                print(f"   - Added StandardScaler pipeline + MultiOutputRegressor for SVM flux")

                # Update param_distributions to use 'estimator__svr__' prefix
                updated_distributions = {}
                for key, value in param_distributions.items():
                    if not key.startswith('estimator__') and not key.startswith('svr__'):
                        updated_distributions[f'estimator__svr__{key}'] = value
                    else:
                        updated_distributions[key] = value
                param_distributions = updated_distributions
            else:
                model = pipeline
                print(f"   - Added StandardScaler pipeline for SVM")

                # Update param_distributions to use 'svr__' prefix
                updated_distributions = {}
                for key, value in param_distributions.items():
                    if not key.startswith('svr__') and not key.startswith('estimator__'):
                        updated_distributions[f'svr__{key}'] = value
                    else:
                        updated_distributions[key] = value
                param_distributions = updated_distributions

        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=2,
            random_state=42
        )

        success, best_params, best_score, search_obj = self.safe_fit_with_progress(
            random_search, X_train, y_train, groups, n_splits
        )

        stage_time = time.time() - stage_start
        print(f"\nStage 1 took {stage_time/60:.1f} minutes")

        if success and best_params:
            print(f"Best params: {best_params}")

        return success, best_params, best_score, search_obj, stage_time

class GridSearchStage(OptimizationStage):
    """Grid search optimization stage"""

    def __init__(self):
        super().__init__("Grid Search", 2, 3)

    def run(self, X_train, y_train, model_class, model_type, best_random_params,
            cv, n_splits, scoring, n_jobs, groups, needs_wrapper,
            fixed_params=None):
        """Execute grid search stage"""
        self.print_header()
        print("Refining parameters with focused ±20% grid around best Random Search results...")
        print("This stage provides consistent, predictable fine-tuning around promising values")

        stage_start = time.time()

        # Create parameter grid
        handler = ParameterHandlerFactory.create_handler(model_type)
        param_grid = handler.create_grid_params(best_random_params, needs_wrapper)

        # Calculate total combinations
        if isinstance(param_grid, list):
            total_combinations = sum(
                np.prod([len(v) for v in grid.values()]) for grid in param_grid
            )
        else:
            total_combinations = np.prod([len(v) for v in param_grid.values()])

        print(f"\n🔍 Grid Search Parameters:")
        print(f"   - Parameter combinations: {total_combinations}")
        print(f"   - Cross-validation folds: {n_splits}")
        print(f"   - Total fits: {total_combinations * n_splits}")
        print(f"   - Timeout: {config.stage_timeout}s")

        # Create model instance with fixed parameters
        if fixed_params:
            model = model_class(**fixed_params)
            print(f"   - Using {len(fixed_params)} fixed parameters during optimization")
        else:
            model = model_class()

        # CRITICAL FIX: Add scaling for SVM and MultiOutputRegressor for flux
        if model_type == 'svm' and not hasattr(model, 'named_steps'):
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.multioutput import MultiOutputRegressor

            # Create Pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', model)
            ])

            # Check if we need MultiOutputRegressor for flux targets
            has_multi_output_data = len(y_train.shape) > 1 and y_train.shape[1] > 1
            if has_multi_output_data:
                model = MultiOutputRegressor(pipeline)
                print(f"   - Added StandardScaler pipeline + MultiOutputRegressor for SVM flux")

                # Update param_grid to use 'estimator__svr__' prefix
                if isinstance(param_grid, list):
                    updated_grids = []
                    for grid in param_grid:
                        updated_grid = {}
                        for key, value in grid.items():
                            if not key.startswith('estimator__') and not key.startswith('svr__'):
                                updated_grid[f'estimator__svr__{key}'] = value
                            else:
                                updated_grid[key] = value
                        updated_grids.append(updated_grid)
                    param_grid = updated_grids
                else:
                    updated_grid = {}
                    for key, value in param_grid.items():
                        if not key.startswith('estimator__') and not key.startswith('svr__'):
                            updated_grid[f'estimator__svr__{key}'] = value
                        else:
                            updated_grid[key] = value
                    param_grid = updated_grid
            else:
                model = pipeline
                print(f"   - Added StandardScaler pipeline for SVM")

                # Update param_grid to use 'svr__' prefix
                if isinstance(param_grid, list):
                    updated_grids = []
                    for grid in param_grid:
                        updated_grid = {}
                        for key, value in grid.items():
                            if not key.startswith('svr__') and not key.startswith('estimator__'):
                                updated_grid[f'svr__{key}'] = value
                            else:
                                updated_grid[key] = value
                        updated_grids.append(updated_grid)
                    param_grid = updated_grids
                else:
                    updated_grid = {}
                    for key, value in param_grid.items():
                        if not key.startswith('svr__') and not key.startswith('estimator__'):
                            updated_grid[f'svr__{key}'] = value
                        else:
                            updated_grid[key] = value
                    param_grid = updated_grid

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=2
        )

        success, best_params, best_score, search_obj = self.safe_fit_with_progress(
            grid_search, X_train, y_train, groups, n_splits
        )

        stage_time = time.time() - stage_start
        print(f"\nStage 2 took {stage_time/60:.1f} minutes")

        return success, best_params, best_score, search_obj, stage_time

class BayesianSearchStage(OptimizationStage):
    """Bayesian optimization stage"""

    def __init__(self):
        super().__init__("Bayesian Optimization", 3, 3)

    def run(self, X_train, y_train, model_class, model_type, best_params_so_far,
            cv, n_splits, scoring, n_jobs, groups, needs_wrapper, n_iter=100,
            fixed_params=None):
        """Execute Bayesian optimization stage"""
        self.print_header()
        print("Fine-tuning with intelligent search...")

        stage_start = time.time()

        # Create search spaces
        handler = ParameterHandlerFactory.create_handler(model_type)
        search_spaces = handler.create_bayesian_spaces(best_params_so_far, needs_wrapper)

        print(f"\n🎯 Bayesian Search Parameters:")
        print(f"   - Iterations: {n_iter}")
        print(f"   - Cross-validation folds: {n_splits}")
        print(f"   - Total fits: up to {n_iter * n_splits}")
        print(f"   - Timeout: {config.stage_timeout}s")

        # Create model instance with fixed parameters
        if fixed_params:
            model = model_class(**fixed_params)
            print(f"   - Using {len(fixed_params)} fixed parameters during optimization")
        else:
            model = model_class()

        # CRITICAL FIX: Add scaling for SVM and MultiOutputRegressor for flux
        if model_type == 'svm' and not hasattr(model, 'named_steps'):
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.multioutput import MultiOutputRegressor

            # Create Pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', model)
            ])

            # Check if we need MultiOutputRegressor for flux targets
            has_multi_output_data = len(y_train.shape) > 1 and y_train.shape[1] > 1
            if has_multi_output_data:
                model = MultiOutputRegressor(pipeline)
                print(f"   - Added StandardScaler pipeline + MultiOutputRegressor for SVM flux")

                # Update search_spaces to use 'estimator__svr__' prefix
                updated_spaces = {}
                for key, value in search_spaces.items():
                    if not key.startswith('estimator__') and not key.startswith('svr__'):
                        updated_spaces[f'estimator__svr__{key}'] = value
                    else:
                        updated_spaces[key] = value
                search_spaces = updated_spaces
            else:
                model = pipeline
                print(f"   - Added StandardScaler pipeline for SVM")

                # Update search_spaces to use 'svr__' prefix
                updated_spaces = {}
                for key, value in search_spaces.items():
                    if not key.startswith('svr__') and not key.startswith('estimator__'):
                        updated_spaces[f'svr__{key}'] = value
                    else:
                        updated_spaces[key] = value
                search_spaces = updated_spaces

        bayes_search = BayesSearchCV(
            model,
            search_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=2,
            random_state=42
        )

        success, best_params, best_score, search_obj = self.safe_fit_with_progress(
            bayes_search, X_train, y_train, groups, n_splits
        )

        stage_time = time.time() - stage_start
        print(f"\nStage 3 took {stage_time/60:.1f} minutes")

        return success, best_params, best_score, search_obj, stage_time

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def clean_optimization_parameters(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Clean parameter names by removing optimization prefixes (estimator__svr__, estimator__, svr__)"""
    clean_params = {}
    for key, value in params_dict.items():
        if key.startswith('estimator__svr__'):
            # Handle double-nested parameters for SVM flux models
            clean_key = key.replace('estimator__svr__', '')
            clean_params[clean_key] = value
        elif key.startswith('estimator__'):
            clean_key = key.replace('estimator__', '')
            clean_params[clean_key] = value
        elif key.startswith('svr__'):
            clean_key = key.replace('svr__', '')
            clean_params[clean_key] = value
        else:
            clean_params[key] = value
    return clean_params

# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================
def three_stage_optimization(X_train, y_train, model_class, model_type='xgboost',
                           n_jobs=-1, target_type='flux', use_log_flux=True, groups=None,
                           n_random_iter=None, n_bayesian_iter=None, fast_mode=False):
    """
    Three-stage hyperparameter optimization: Random → Grid → Bayesian

    Args:
        X_train: Training features
        y_train: Training targets
        model_class: Model class to optimize
        model_type: Type of model ('xgboost', 'random_forest', 'svm', 'neural_net')
        n_jobs: Number of parallel jobs
        target_type: 'flux' or 'keff' - determines optimization metric
        use_log_flux: Whether flux data is log-transformed
        groups: Array indicating which samples belong to same original config
        n_random_iter: Number of random search iterations
        n_bayesian_iter: Number of Bayesian search iterations
        fast_mode: If True, uses reduced iteration counts for quick testing

    Returns:
        tuple: (best_params, None)
    """
    # Configure iteration counts
    if fast_mode:
        actual_random_iter = config.fast_random_iter
        actual_bayesian_iter = config.fast_bayesian_iter
        print(f"   🚀 FAST MODE: Using reduced iteration counts for quick testing")
    else:
        actual_random_iter = n_random_iter if n_random_iter is not None else config.default_random_iter
        actual_bayesian_iter = n_bayesian_iter if n_bayesian_iter is not None else config.default_bayesian_iter

    print(f"\n{'='*60}")
    print(f"THREE-STAGE HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Model: {model_type}")
    print(f"Target: {target_type.upper()}")
    print(f"Optimization metric: {'MAPE' if target_type == 'flux' else 'MSE'}")
    print(f"Random search iterations: {actual_random_iter}")
    print(f"Bayesian search iterations: {actual_bayesian_iter}")
    print(f"Stage timeout: {config.stage_timeout}s, Total timeout: {config.total_timeout}s")
    print(f"{'='*60}")

    # Validate model class matches model type
    model_class_name = model_class.__name__.lower()
    expected_names = {
        'xgboost': ['xgbregressor', 'xgbclassifier'],
        'random_forest': ['randomforestregressor', 'randomforestclassifier'],
        'svm': ['svr', 'svc', 'linearsvr', 'linearsvc'],
        'neural_net': ['mlpregressor', 'mlpclassifier', 'neuralnetwork']
    }

    if model_type in expected_names:
        if not any(name in model_class_name for name in expected_names[model_type]):
            print(f"⚠️  WARNING: Model class '{model_class.__name__}' may not match model_type '{model_type}'")
            print(f"   Expected one of: {expected_names[model_type]}")

    optimization_start_time = time.time()
    best_params_so_far = {}
    best_score_so_far = float('-inf')

    # Determine if multi-output and needs wrapper
    has_multi_output_data = len(y_train.shape) > 1 and y_train.shape[1] > 1

    if has_multi_output_data:
        try:
            test_model = model_class()

            # Check if the model instance is already a MultiOutputRegressor wrapper
            from sklearn.multioutput import MultiOutputRegressor

            if isinstance(test_model, MultiOutputRegressor):
                print(f"   - Multi-output: Model is already wrapped with MultiOutputRegressor")
                needs_wrapper = True  # Parameters need estimator__ prefix
            elif hasattr(test_model, '_more_tags'):
                tags = test_model._more_tags()
                native_multioutput = tags.get('multioutput', False) or tags.get('multioutput_only', False)
                if native_multioutput:
                    print(f"   - Multi-output: Native support detected, no wrapper needed")
                    needs_wrapper = False
                else:
                    print(f"   - Multi-output: Using MultiOutputRegressor wrapper")
                    needs_wrapper = True
            else:
                # Fallback: check if model type typically needs wrapper
                native_multioutput = model_type in ['random_forest']  # Only RF has native support
                if native_multioutput:
                    print(f"   - Multi-output: Native support detected for {model_type}")
                    needs_wrapper = False
                else:
                    print(f"   - Multi-output: Using MultiOutputRegressor wrapper for {model_type}")
                    needs_wrapper = True
        except Exception as e:
            print(f"   - Multi-output: Error detecting wrapper ({str(e)[:50]}), using safe default")
            needs_wrapper = True
    else:
        needs_wrapper = False

    print(f"\n📊 Data Info:")
    print(f"   - Training samples: {X_train.shape[0]}")
    print(f"   - Features: {X_train.shape[1]}")
    print(f"   - Output type: {'Multi-output' if has_multi_output_data else 'Single output'}")
    if has_multi_output_data:
        print(f"   - Number of outputs: {y_train.shape[1]}")
    print(f"   - Using {n_jobs} parallel jobs")

    # Set up cross-validation
    cv, n_splits = setup_cross_validation(X_train, y_train, groups)

    # Set up scoring
    if target_type == 'flux':
        scoring = create_mape_scorer(use_log_flux)
        print(f"   - Using MAPE scoring (log_flux={use_log_flux})")
    else:
        scoring = 'neg_mean_squared_error'
        print(f"   - Using MSE scoring")

    # Get parameter handler
    handler = ParameterHandlerFactory.create_handler(model_type)

    # Extract fixed parameters that should not be optimized
    fixed_params = handler.get_fixed_params()
    print(f"\n🔒 Fixed Parameters (not optimized):")
    for param, value in fixed_params.items():
        print(f"   - {param}: {value}")

    # ========================================================================
    # STAGE 1: RANDOM SEARCH
    # ========================================================================
    random_stage = RandomSearchStage()
    param_distributions = handler.get_random_distributions(needs_wrapper)

    success1, best_random_params, best_random_score, random_search_obj, stage1_time = \
        random_stage.run(X_train, y_train, model_class, param_distributions,
                        cv, n_splits, scoring, n_jobs, groups, actual_random_iter,
                        fixed_params=fixed_params, model_type=model_type)

    if success1 and best_random_params:
        best_params_so_far = best_random_params
        best_score_so_far = best_random_score
    else:
        print(f"\n❌ Stage 1 failed to find any parameters. Returning default parameters.")
        default_params = handler.get_default_params()
        # Always return clean parameters (no prefixes) for final model training
        return default_params, None

    # Check total timeout
    if time.time() - optimization_start_time > config.total_timeout:
        print(f"\n⏱️  Total timeout reached. Returning best parameters found so far.")
        return clean_optimization_parameters(best_params_so_far), None

    # ========================================================================
    # STAGE 2: GRID SEARCH
    # ========================================================================
    grid_stage = GridSearchStage()

    success2, best_grid_params, best_grid_score, grid_search_obj, stage2_time = \
        grid_stage.run(X_train, y_train, model_class, model_type,
                      best_random_params, cv, n_splits, scoring, n_jobs,
                      groups, needs_wrapper, fixed_params=fixed_params)

    if success2 and best_grid_params:
        if best_grid_score > best_score_so_far:
            best_params_so_far = best_grid_params
            best_score_so_far = best_grid_score
            print(f"\n✅ Grid search improved performance!")
        else:
            print(f"\n Grid search did not improve performance.")

    # Check total timeout
    if time.time() - optimization_start_time > config.total_timeout:
        print(f"\n⏱️  Total timeout reached. Returning best parameters found so far.")
        return clean_optimization_parameters(best_params_so_far), None

    # ========================================================================
    # STAGE 3: BAYESIAN OPTIMIZATION
    # ========================================================================
    bayes_stage = BayesianSearchStage()

    success3, best_bayes_params, best_bayes_score, bayes_search_obj, stage3_time = \
        bayes_stage.run(X_train, y_train, model_class, model_type,
                       best_params_so_far, cv, n_splits, scoring, n_jobs,
                       groups, needs_wrapper, actual_bayesian_iter,
                       fixed_params=fixed_params)

    if success3 and best_bayes_params:
        if best_bayes_score > best_score_so_far:
            best_params_so_far = best_bayes_params
            best_score_so_far = best_bayes_score
            print(f"\n✅ Bayesian search improved performance!")
        else:
            print(f"\n Bayesian search did not improve performance.")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_time = time.time() - optimization_start_time

    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total optimization time: {total_time/60:.1f} minutes")
    print(f"  - Stage 1 (Random):   {stage1_time/60:.1f} minutes")
    print(f"  - Stage 2 (Grid):     {stage2_time/60:.1f} minutes")
    print(f"  - Stage 3 (Bayesian): {stage3_time/60:.1f} minutes")

    if target_type == 'flux':
        print(f"\nFinal best MAPE: {abs(best_score_so_far):.2f}%")
    else:
        print(f"\nFinal best MSE: {abs(best_score_so_far):.10f}")

    # Prepare final parameters - fixed params were used throughout optimization
    handler = ParameterHandlerFactory.create_handler(model_type)
    default_params = handler.get_default_params()

    # Clean up parameters for final model and ensure complete parameter set
    if needs_wrapper and model_type in ['xgboost', 'svm', 'neural_net']:
        clean_params = clean_optimization_parameters(best_params_so_far)

        # Ensure all default parameters are included (optimized + fixed)
        for param, value in default_params.items():
            if param not in clean_params:
                clean_params[param] = value

        print(f"\nOptimal parameters (fixed parameters were used throughout optimization):")
        optimized_params = {k: v for k, v in clean_params.items() if k not in fixed_params}
        fixed_params_used = {k: v for k, v in clean_params.items() if k in fixed_params}

        print(f"   Optimized parameters:")
        for param, value in optimized_params.items():
            print(f"   - {param}: {value}")
        print(f"   Fixed parameters:")
        for param, value in fixed_params_used.items():
            print(f"   - {param}: {value}")

        return clean_params, None
    else:
        # Clean up parameters for final model (handle SVM pipeline parameters)
        final_params = clean_optimization_parameters(best_params_so_far)

        # Ensure all default parameters are included (optimized + fixed)
        for param, value in default_params.items():
            if param not in final_params:
                final_params[param] = value

        print(f"\nOptimal parameters (fixed parameters were used throughout optimization):")
        optimized_params = {k: v for k, v in final_params.items() if k not in fixed_params}
        fixed_params_used = {k: v for k, v in final_params.items() if k in fixed_params}

        print(f"   Optimized parameters:")
        for param, value in optimized_params.items():
            print(f"   - {param}: {value}")
        print(f"   Fixed parameters:")
        for param, value in fixed_params_used.items():
            print(f"   - {param}: {value}")

        return final_params, None

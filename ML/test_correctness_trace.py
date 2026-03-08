"""
Trace every data handoff in the restructured pipeline and verify correctness.

This isn't about shapes or leakage -- it's about checking that the actual
VALUES flowing through each stage are what they should be.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_restructure import (
    _get_feature_layout,
    restructure_for_position_pooling,
    restructure_single_for_prediction,
)


def make_traceable_data():
    """Create data where every value is unique and traceable.

    Feature vector for sample i:
      global:   [i+0.01, i+0.02, i+0.03, i+0.04, i+0.05]
      local_p0: [i+0.10, i+0.11, i+0.12, i+0.13, i+0.14]
      local_p1: [i+0.20, i+0.21, i+0.22, i+0.23, i+0.24]
      local_p2: [i+0.30, i+0.31, i+0.32, i+0.33, i+0.34]
      local_p3: [i+0.40, i+0.41, i+0.42, i+0.43, i+0.44]
      nci_p0:   [i+0.50, i+0.51, i+0.52]
      nci_p1:   [i+0.60, i+0.61, i+0.62]
      nci_p2:   [i+0.70, i+0.71, i+0.72]
      nci_p3:   [i+0.80, i+0.81, i+0.82]

    Flux targets for sample i:
      [i+0.91, i+0.92, i+0.93, i+0.94]  (one per position)
    """
    N = 3
    X = np.zeros((N, 37))
    y = np.zeros((N, 4))

    for i in range(N):
        base = float(i * 10)  # 0, 10, 20
        # Global (5)
        X[i, 0:5] = [base+0.01, base+0.02, base+0.03, base+0.04, base+0.05]
        # Local pos 0-3 (5 each)
        for p in range(4):
            offset = 5 + p * 5
            X[i, offset:offset+5] = [base + (p+1)*0.1 + j*0.01 for j in range(5)]
        # NCI pos 0-3 (3 each)
        for p in range(4):
            offset = 25 + p * 3
            X[i, offset:offset+3] = [base + (p+5)*0.1 + j*0.01 for j in range(3)]
        # Targets
        y[i] = [base+0.91, base+0.92, base+0.93, base+0.94]

    groups = np.array([0, 0, 1])  # samples 0,1 same group; sample 2 different
    return X, y, groups


def test_restructured_values_exact():
    """For each restructured row, trace every feature back to its source."""
    X, y, groups = make_traceable_data()
    Xp, yp, gp = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    print("Checking every value in every restructured row...")
    errors = []

    for orig_idx in range(X.shape[0]):
        base = float(orig_idx * 10)
        orig_global = X[orig_idx, 0:5]

        for my_pos in range(4):
            row_idx = orig_idx * 4 + my_pos
            row = Xp[row_idx]

            # 1. Global features [0:5] should be unchanged
            if not np.array_equal(row[0:5], orig_global):
                errors.append(f"Sample {orig_idx} pos {my_pos}: global features wrong")

            # 2. MY local [5:10] should be position my_pos's local from original
            orig_my_local = X[orig_idx, 5 + my_pos*5 : 5 + my_pos*5 + 5]
            if not np.array_equal(row[5:10], orig_my_local):
                errors.append(f"Sample {orig_idx} pos {my_pos}: MY local wrong. "
                            f"Expected {orig_my_local}, got {row[5:10]}")

            # 3. MY NCI [10:13] should be position my_pos's NCI from original
            orig_my_nci = X[orig_idx, 25 + my_pos*3 : 25 + my_pos*3 + 3]
            if not np.array_equal(row[10:13], orig_my_nci):
                errors.append(f"Sample {orig_idx} pos {my_pos}: MY NCI wrong. "
                            f"Expected {orig_my_nci}, got {row[10:13]}")

            # 4. OTHER local [13:28] should be the 3 other positions in order
            other_positions = [p for p in range(4) if p != my_pos]
            for j, other_pos in enumerate(other_positions):
                expected = X[orig_idx, 5 + other_pos*5 : 5 + other_pos*5 + 5]
                actual = row[13 + j*5 : 13 + j*5 + 5]
                if not np.array_equal(actual, expected):
                    errors.append(f"Sample {orig_idx} pos {my_pos}: OTHER local[{j}] "
                                f"(pos {other_pos}) wrong. Expected {expected}, got {actual}")

            # 5. OTHER NCI [28:37] should be the 3 other positions in order
            for j, other_pos in enumerate(other_positions):
                expected = X[orig_idx, 25 + other_pos*3 : 25 + other_pos*3 + 3]
                actual = row[28 + j*3 : 28 + j*3 + 3]
                if not np.array_equal(actual, expected):
                    errors.append(f"Sample {orig_idx} pos {my_pos}: OTHER NCI[{j}] "
                                f"(pos {other_pos}) wrong. Expected {expected}, got {actual}")

            # 6. Target should be the flux for my_pos
            expected_target = y[orig_idx, my_pos]
            if yp[row_idx] != expected_target:
                errors.append(f"Sample {orig_idx} pos {my_pos}: target wrong. "
                            f"Expected {expected_target}, got {yp[row_idx]}")

            # 7. Group should match original
            if gp[row_idx] != groups[orig_idx]:
                errors.append(f"Sample {orig_idx} pos {my_pos}: group wrong. "
                            f"Expected {groups[orig_idx]}, got {gp[row_idx]}")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise AssertionError(f"{len(errors)} value errors found!")
    else:
        print(f"  Checked {X.shape[0] * 4} rows × 37 features each = "
              f"{X.shape[0] * 4 * 37} individual values")
        print("  PASS: Every single value traces correctly to its source")


def test_prediction_restructure_exact_match():
    """Verify prediction restructure produces IDENTICAL features to training restructure."""
    X, y, groups = make_traceable_data()
    Xp_train, _, _ = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')
    Xp_pred = restructure_single_for_prediction(X, 'fill', 'separate')

    if not np.array_equal(Xp_train, Xp_pred):
        diff_mask = Xp_train != Xp_pred
        diff_count = np.sum(diff_mask)
        diff_locs = np.argwhere(diff_mask)
        print(f"  MISMATCH at {diff_count} locations:")
        for loc in diff_locs[:10]:
            print(f"    [{loc[0]}, {loc[1]}]: train={Xp_train[loc[0],loc[1]]}, "
                  f"pred={Xp_pred[loc[0],loc[1]]}")
        raise AssertionError("Training and prediction restructure produce different features!")

    print("  PASS: Prediction restructure is byte-identical to training restructure")


def test_roundtrip_reconstruct():
    """From restructured predictions, verify we can reconstruct the original y_flux."""
    X, y, groups = make_traceable_data()
    _, yp, _ = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    # Simulate what predict_flux does: reshape 1D predictions to (N, 4)
    N = X.shape[0]
    y_reconstructed = yp.reshape(N, 4)

    if not np.array_equal(y_reconstructed, y):
        print(f"  Original y:\n{y}")
        print(f"  Reconstructed y:\n{y_reconstructed}")
        raise AssertionError("Reshape doesn't recover original targets!")

    print("  PASS: reshape(N, 4) perfectly recovers original flux targets")


def test_model_sees_consistent_features():
    """
    The critical test: features the model sees during Optuna CV scoring
    must be identical to what it sees during final training and prediction.

    Simulate the exact sequence:
    1. Original X (from DataHandler)
    2. Restructure for CV scoring (inside Optuna objective)
    3. Restructure for final training (in model_trainer)
    4. Restructure for prediction (inside predict_flux)
    All three must produce the same feature matrix.
    """
    X, y, groups = make_traceable_data()

    # Step 2: What Optuna's objective does
    Xp_optuna, _, _ = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    # Step 3: What model_trainer does before _create_and_train_model
    Xp_final, _, _ = restructure_for_position_pooling(X, y, groups, 'fill', 'separate')

    # Step 4: What predict_flux does internally
    Xp_predict = restructure_single_for_prediction(X, 'fill', 'separate')

    assert np.array_equal(Xp_optuna, Xp_final), \
        "Optuna CV features != final training features"
    assert np.array_equal(Xp_final, Xp_predict), \
        "Final training features != prediction features"

    print("  PASS: Optuna CV, final training, and prediction all see identical features")


if __name__ == '__main__':
    test_restructured_values_exact()
    print()
    test_prediction_restructure_exact_match()
    print()
    test_roundtrip_reconstruct()
    print()
    test_model_sees_consistent_features()
    print("\nAll correctness trace tests passed.")

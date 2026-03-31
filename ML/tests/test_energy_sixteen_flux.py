"""Tests for energy_sixteen flux mode: 16 columns = 4 positions × (tot, th, epi, fast)."""
import os
import numpy as np

# Column order: for position index p in 0..3, cols 4*p+0..3 are tot, th, epi, fast.


def test_prepare_energy_sixteen_shape_and_order():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from execution.data_handler import DataHandler

    dh = DataHandler()
    lattice = np.full((8, 8), 'F', dtype=object)
    lattice[2, 2] = 'I_1'
    position_order = [(2, 2), (2, 3), (2, 4), (2, 5)]
    flux_dict = {'I_1': 1.0e14}
    energy_dict = {'I_1': {'thermal': 0.5, 'epithermal': 0.3, 'fast': 0.2}}
    out = dh._prepare_energy_sixteen_flux_values(lattice, flux_dict, energy_dict, position_order)
    assert len(out) == 16
    tot0, th0, epi0, fast0 = out[0], out[1], out[2], out[3]
    assert abs(tot0 - 1.0e14) < 1.0
    assert abs(th0 - 0.5e14) < 1.0
    assert abs(epi0 - 0.3e14) < 1.0
    assert abs(fast0 - 0.2e14) < 1.0


def test_load_energy_sixteen_smoke():
    train = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.txt')
    if not os.path.exists(train):
        return
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from execution.data_handler import DataHandler

    dh = DataHandler()
    X, y_flux, y_keff, groups, lattices, _ = dh.load_and_prepare_data(
        train, 'physics', flux_mode='energy_sixteen', max_runs=33)
    assert y_flux.shape[1] == 16
    assert X.shape[0] == y_flux.shape[0]


if __name__ == '__main__':
    test_prepare_energy_sixteen_shape_and_order()
    test_load_energy_sixteen_smoke()
    print('PASS: test_energy_sixteen_flux')

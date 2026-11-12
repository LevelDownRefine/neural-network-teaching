import numpy as np
import pytest

from lif.simulate import Parameters, inputs, simulate


def test_constant_subthreshold_and_threshold_current_do_not_spike():
    params = Parameters()
    for current_na in (0, 1, 1.5):
        _, current = inputs(params, current_na=current_na)
        voltage, spikes = simulate(current, params)
        assert not spikes.any()
        assert np.isfinite(voltage).all()
        if current_na == 0:
            assert np.allclose(voltage, params.rest_mv)


def test_superthreshold_reset_and_refractory():
    params = Parameters()
    _, current = inputs(params)
    voltage, spikes = simulate(current, params)
    times = np.flatnonzero(spikes)
    assert len(times) > 1
    assert np.all(voltage[times] == params.reset_mv)
    assert np.all(np.diff(times) > round(params.refractory_ms / params.dt_ms))


def test_invalid_refractory_is_rejected():
    with pytest.raises(ValueError, match="multiple"):
        simulate([0], Parameters(refractory_ms=2.05))


def test_reaching_threshold_counts_as_spike():
    params = Parameters(tau_ms=1, dt_ms=1, refractory_ms=0)
    voltage, spikes = simulate([1.5], params)
    assert spikes.tolist() == [True]
    assert voltage.tolist() == [params.reset_mv]

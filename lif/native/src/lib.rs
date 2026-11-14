//! FP32 LIF 前向仿真；阈值检测、立即重置及整步不应期。
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

pub type Trace = (Vec<f32>, Vec<u8>);

pub fn simulate_core(
    current: &[f32],
    beta: f32,
    scale: f32,
    rest: f32,
    threshold_relative: f32,
    refractory_steps: usize,
) -> Result<Trace, &'static str> {
    if !beta.is_finite()
        || !(0.0..=1.0).contains(&beta)
        || !scale.is_finite()
        || scale <= 0.0
        || !rest.is_finite()
        || !threshold_relative.is_finite()
        || threshold_relative <= 0.0
    {
        return Err("invalid LIF coefficients");
    }
    if current.is_empty() || current.iter().any(|sample| !sample.is_finite()) {
        return Err("current must be nonempty and finite");
    }

    let mut voltage = Vec::with_capacity(current.len());
    let mut spikes = Vec::with_capacity(current.len());
    let mut u = 0.0_f32;
    let mut remaining = 0_usize;
    for &sample in current {
        let mut spike = 0_u8;
        if remaining > 0 {
            remaining -= 1;
            u = 0.0;
        } else {
            // 与 PyTorch FP32 分别乘、加的舍入顺序一致；不要换成 mul_add。
            let increment = scale * sample;
            let leaked = beta * u;
            u = leaked + increment;
            if !u.is_finite() {
                return Err("membrane update overflowed");
            }
            if u >= threshold_relative {
                spike = 1;
                u = 0.0;
                remaining = refractory_steps;
            }
        }
        let absolute_voltage = u + rest;
        if !absolute_voltage.is_finite() {
            return Err("voltage overflowed");
        }
        voltage.push(absolute_voltage);
        spikes.push(spike);
    }
    Ok((voltage, spikes))
}

#[cfg(feature = "python")]
#[pyfunction]
fn simulate(
    py: Python<'_>,
    current: Vec<f32>,
    beta: f32,
    scale: f32,
    rest: f32,
    threshold_relative: f32,
    refractory_steps: usize,
) -> PyResult<Trace> {
    // 整段时间循环留在 Rust 中，释放 GIL，让 Qt 主线程继续响应。
    py.detach(move || {
        simulate_core(
            &current,
            beta,
            scale,
            rest,
            threshold_relative,
            refractory_steps,
        )
    })
    .map_err(PyValueError::new_err)
}

#[cfg(feature = "python")]
#[pymodule]
fn lif_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(simulate, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leak_and_integration() {
        let (voltage, spikes) = simulate_core(&[2.0, 0.0], 0.5, 1.0, -65.0, 15.0, 0).unwrap();
        assert_eq!(voltage, [-63.0, -64.0]);
        assert_eq!(spikes, [0, 0]);
    }

    #[test]
    fn equality_spikes_and_resets() {
        let (voltage, spikes) = simulate_core(&[1.5], 0.0, 10.0, -65.0, 15.0, 0).unwrap();
        assert_eq!(voltage, [-65.0]);
        assert_eq!(spikes, [1]);
    }

    #[test]
    fn full_refractory_period() {
        let (voltage, spikes) = simulate_core(&[2.0; 4], 0.0, 10.0, -65.0, 15.0, 2).unwrap();
        assert_eq!(voltage, [-65.0; 4]);
        assert_eq!(spikes, [1, 0, 0, 1]);
    }

    #[test]
    fn zero_refractory_allows_next_step_spike() {
        let (_, spikes) = simulate_core(&[2.0; 3], 0.0, 10.0, -65.0, 15.0, 0).unwrap();
        assert_eq!(spikes, [1, 1, 1]);
    }

    #[test]
    fn critical_current_does_not_spike() {
        let (_, spikes) = simulate_core(&vec![1.5; 20_000], 0.995, 0.05, -65.0, 15.0, 20).unwrap();
        assert!(spikes.iter().all(|&spike| spike == 0));
    }

    #[test]
    fn rejects_invalid_inputs() {
        for input in [&[][..], &[f32::NAN][..], &[f32::INFINITY][..]] {
            assert!(simulate_core(input, 0.5, 1.0, -65.0, 15.0, 0).is_err());
        }
        for beta in [-1.0, 1.1, f32::NAN] {
            assert!(simulate_core(&[1.0], beta, 1.0, -65.0, 15.0, 0).is_err());
        }
        assert!(simulate_core(&[f32::MAX], 0.5, 2.0, -65.0, 15.0, 0).is_err());
    }
}

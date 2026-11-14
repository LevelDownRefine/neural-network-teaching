//! 不依赖 Python 的输入生成、仿真和发放率统计。
use crate::simulate_core;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
#[cfg_attr(feature = "python", pyo3(from_item_all))]
#[cfg_attr(feature = "serialization", derive(serde::Serialize))]
pub struct Parameters {
    pub sinusoidal: bool,
    pub current: f64,
    pub amplitude: f64,
    pub frequency: f64,
    pub phase: f64,
    pub threshold: f64,
    pub tau: f64,
    pub resistance: f64,
    pub refractory: f64,
    pub duration: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            sinusoidal: false,
            current: 2.0,
            amplitude: 0.8,
            frequency: 5.0,
            phase: 0.0,
            threshold: -50.0,
            tau: 20.0,
            resistance: 10.0,
            refractory: 2.0,
            duration: 2000.0,
        }
    }
}

pub const DT: f64 = 0.1;
pub const REST: f64 = -65.0;

pub struct Experiment {
    pub current: Vec<f32>,
    pub voltage: Vec<f32>,
    pub spikes: Vec<u8>,
    pub rate: Vec<f64>,
}

impl Experiment {
    pub fn run(p: &Parameters) -> Result<Self, &'static str> {
        if ![
            p.current,
            p.amplitude,
            p.frequency,
            p.phase,
            p.threshold,
            p.tau,
            p.resistance,
            p.refractory,
            p.duration,
        ]
        .iter()
        .all(|x| x.is_finite())
        {
            return Err("Parameters must be finite");
        }
        let steps = (p.duration / DT).round_ties_even();
        let refractory_steps = (p.refractory / DT).round_ties_even();
        if !(1.0..=100_000.0).contains(&steps)
            || (p.duration / DT - steps).abs() > 1e-7
            || p.refractory < 0.0
            || (p.refractory / DT - refractory_steps).abs() > 1e-7
            || p.tau < DT
            || p.resistance <= 0.0
            || p.threshold <= REST
            || p.amplitude < 0.0
            || p.frequency < 0.0
        {
            return Err("Invalid time step or neuron parameters");
        }
        let current: Vec<f32> = (0..steps as usize)
            .map(|k| {
                let modulation = if p.sinusoidal {
                    p.amplitude
                        * (std::f64::consts::TAU * p.frequency * (k as f64 * DT) / 1000.0 + p.phase)
                            .sin()
                } else {
                    0.0
                };
                (p.current + modulation) as f32
            })
            .collect();
        let (voltage, spikes) = simulate_core(
            &current,
            (1.0 - DT / p.tau) as f32,
            (DT / p.tau * p.resistance) as f32,
            REST as f32,
            (p.threshold - REST) as f32,
            refractory_steps as usize,
        )?;
        let window = (100.0 / DT).round_ties_even() as usize;
        let mut count = 0_usize;
        let rate = spikes
            .iter()
            .enumerate()
            .map(|(k, &spike)| {
                count += spike as usize;
                if k >= window {
                    count -= spikes[k - window] as usize;
                }
                count as f64 / ((k + 1).min(window) as f64 * DT / 1000.0)
            })
            .collect();
        Ok(Self {
            current,
            voltage,
            spikes,
            rate,
        })
    }

    pub fn count(&self) -> usize {
        self.spikes.iter().map(|&s| s as usize).sum()
    }

    pub fn first_spike(&self) -> Option<f64> {
        self.spikes
            .iter()
            .position(|&s| s != 0)
            .map(|i| (i + 1) as f64 * DT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_experiments_match_reference_counts() {
        let p = Parameters::default();
        let result = Experiment::run(&p).unwrap();
        assert_eq!(result.count(), 67);
        assert!((result.first_spike().unwrap() - 27.7).abs() < 1e-10);
        let sine = Experiment::run(&Parameters {
            sinusoidal: true,
            ..p
        })
        .unwrap();
        assert_eq!(sine.count(), 60);
        assert!(sine.rate.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn zero_input_and_invalid_parameters() {
        let mut p = Parameters {
            current: 0.0,
            ..Parameters::default()
        };
        let result = Experiment::run(&p).unwrap();
        assert_eq!(result.count(), 0);
        assert!(result.voltage.iter().all(|&v| v == -65.0));
        p.refractory = 0.15;
        assert!(Experiment::run(&p).is_err());
        p.refractory = 2.0;
        p.duration = 0.0;
        assert!(Experiment::run(&p).is_err());
    }
}

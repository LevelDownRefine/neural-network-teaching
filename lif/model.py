"""LIF 模型及离散时间仿真，电位 mV、时间 ms、电流 nA。"""

from dataclasses import dataclass

import numpy as np
import snntorch as snn
import torch


class LIFNeuron(torch.nn.Module):
    def __init__(self, dt, tau_m, R, V_rest, V_th, refractory_ms):
        super().__init__()
        if not all(np.isfinite(x) for x in (dt, tau_m, R, V_rest, V_th, refractory_ms)):
            raise ValueError("LIF parameters must be finite")
        if dt <= 0 or tau_m <= 0 or dt > tau_m or R <= 0:
            raise ValueError("require 0 < dt <= tau_m and R > 0")
        if refractory_ms < 0 or not np.isclose(
            refractory_ms / dt, round(refractory_ms / dt), rtol=0, atol=1e-7
        ):
            raise ValueError("refractory_ms must be a nonnegative multiple of dt")
        if V_rest >= V_th:
            raise ValueError("rest potential must be below threshold")
        self.dt, self.tau_m, self.R = dt, tau_m, R
        self.V_rest = V_rest
        self.refractory_steps = round(refractory_ms / dt)
        self.surrogate = snn.surrogate.fast_sigmoid()
        self.neuron = snn.Leaky(
            beta=torch.tensor(self._decay_factor(), dtype=torch.float32),
            threshold=self._relative_voltage(V_th),
            reset_mechanism="zero",
            reset_delay=False,
            spike_grad=self._at_threshold,
        )
        self.register_buffer(
            "U",
            torch.tensor(0.0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "remaining", torch.zeros((), dtype=torch.long), persistent=False
        )

    def _at_threshold(self, voltage_from_threshold):
        # 前向达到阈值即发放；反向使用 snnTorch 的代理梯度。
        return self.surrogate(voltage_from_threshold) + (
            voltage_from_threshold == 0
        ).to(voltage_from_threshold.dtype)

    def _relative_voltage(self, V):
        return V - self.V_rest

    def _decay_factor(self):
        return 1 - self.dt / self.tau_m

    def _input_increment(self, I):
        return self.dt / self.tau_m * self.R * I

    def reset_state(self):
        self.U = torch.zeros_like(self.U)
        self.remaining = torch.zeros_like(self.remaining)
        self.neuron.reset_mem()

    def forward(self, I):
        if not isinstance(I, torch.Tensor):
            I = torch.as_tensor(I, dtype=self.U.dtype, device=self.U.device)
        else:
            I = I.to(dtype=self.U.dtype, device=self.U.device)
        if I.ndim != 0:
            raise ValueError("single-neuron input must be a scalar")
        X = self._input_increment(I)
        spike, U = self.neuron(X, self.U)
        active = self.remaining == 0
        spike = torch.where(active, spike, torch.zeros_like(spike))
        self.U = torch.where(active, U, torch.zeros_like(U))
        self.remaining = torch.where(
            active,
            spike.detach().to(torch.long) * self.refractory_steps,
            self.remaining - 1,
        )
        return self.U + self.V_rest, spike


def simulate_current(current, duration_ms, dt, neuron):
    if not np.isfinite(dt) or dt <= 0 or dt != neuron.dt:
        raise ValueError("dt must match the neuron time step")
    if not np.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("duration_ms must be positive and finite")
    if round(duration_ms / dt) < 1 or not np.isclose(
        duration_ms / dt, round(duration_ms / dt), rtol=0, atol=1e-7
    ):
        raise ValueError("duration_ms must be a multiple of dt")
    steps = round(duration_ms / dt)
    I = np.asarray(current)
    if I.ndim == 0:
        I = np.full(steps, I)
    if not np.isfinite(I).all():
        raise ValueError("current must be finite")
    if I.shape != (steps,):
        raise ValueError("I must be a scalar or one value per time step")
    voltages, spikes = [], []
    for I_k in I:
        V, spike = neuron(I_k)
        voltages.append(V)
        spikes.append(spike)
    # 只在绘图前将 Tensor 转换为 NumPy。
    return torch.stack(voltages).detach().cpu().numpy(), torch.stack(
        spikes
    ).detach().cpu().numpy()


def firing_rate(spikes, dt):
    # 回看约 100 ms；起始阶段按已有时长计算。
    window_steps = max(1, round(100 / dt))
    cumulative = np.concatenate(([0], np.cumsum(spikes)))
    end = np.arange(1, len(spikes) + 1)
    start = np.maximum(0, end - window_steps)
    return (cumulative[end] - cumulative[start]) / ((end - start) * dt / 1000)


@dataclass(frozen=True)
class SimulationParameters:
    mode: str = "constant"
    current: float = 2.0
    amplitude: float = 0.8
    frequency: float = 5.0
    phase: float = 0.0
    threshold: float = -50.0
    rest: float = -65.0
    tau: float = 20.0
    resistance: float = 10.0
    refractory: float = 2.0
    dt: float = 0.1
    duration: float = 2000.0


def simulate_experiment(parameters):
    p = parameters
    if not np.isfinite(p.duration) or p.duration <= 0:
        raise ValueError("duration must be positive and finite")
    if not all(np.isfinite(x) for x in (p.current, p.amplitude, p.frequency, p.phase)):
        raise ValueError("input parameters must be finite")
    if p.amplitude < 0 or p.frequency < 0:
        raise ValueError("amplitude and frequency must be nonnegative")
    neuron = LIFNeuron(p.dt, p.tau, p.resistance, p.rest, p.threshold, p.refractory)
    try:
        input_times = np.arange(round(p.duration / p.dt)) * p.dt
        current = np.full_like(input_times, p.current)
        if p.mode == "sinusoidal":
            current += p.amplitude * np.sin(
                2 * np.pi * p.frequency * input_times / 1000 + p.phase
            )
        elif p.mode != "constant":
            raise ValueError("unknown input mode")
        with torch.inference_mode():
            voltage, spikes = simulate_current(current, p.duration, p.dt, neuron)
        return {
            "input_time_ms": input_times,
            "time_ms": input_times + p.dt,
            "current": current,
            "voltage": voltage,
            "spikes": spikes,
            "rate": firing_rate(spikes, p.dt),
        }
    finally:
        # snnTorch 会全局保留神经元；桌面仿真结束后只移除本次临时对象。
        snn.SpikingNeuron.instances.remove(neuron.neuron)

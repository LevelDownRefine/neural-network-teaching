"""Simulate a single LIF neuron with physical units."""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
import torch


@dataclass(frozen=True)
class Parameters:
    rest_mv: float = -65.0
    reset_mv: float = -65.0
    threshold_mv: float = -50.0
    tau_ms: float = 20.0
    resistance_mohm: float = 10.0
    refractory_ms: float = 2.0
    dt_ms: float = 0.1
    duration_ms: float = 2000.0


def lif_step(neuron, membrane, current_na, refractory, params):
    """Advance one time step; return membrane, spike and refractory steps."""
    reset = params.reset_mv - params.rest_mv
    if refractory:
        return torch.tensor(reset, dtype=torch.float64), False, refractory - 1

    drive = torch.tensor(
        params.dt_ms / params.tau_ms * params.resistance_mohm * float(current_na),
        dtype=torch.float64,
    )
    before_reset = neuron.beta * membrane + drive
    spike, membrane = neuron(drive, membrane)
    # snnTorch uses >; the assignment specifies reaching the threshold.
    if spike.item() or before_reset.item() >= params.threshold_mv - params.rest_mv:
        return (
            torch.tensor(reset, dtype=torch.float64),
            True,
            int(round(params.refractory_ms / params.dt_ms)),
        )
    return membrane, False, 0


def simulate(current_na, params=Parameters()):
    """Euler LIF: mV, nA, MOhm and ms; post-spike reset is immediate."""
    if (
        params.dt_ms <= 0
        or params.tau_ms <= 0
        or params.dt_ms > params.tau_ms
        or params.refractory_ms < 0
    ):
        raise ValueError(
            "time step and time constant must be positive; refractory must be nonnegative"
        )
    ratio = params.refractory_ms / params.dt_ms
    if not np.isclose(ratio, round(ratio)):
        raise ValueError("refractory period must be a multiple of time step")
    current = np.asarray(current_na, dtype=float)
    neuron = snn.Leaky(
        beta=torch.tensor(1 - params.dt_ms / params.tau_ms, dtype=torch.float64),
        threshold=params.threshold_mv - params.rest_mv,
        reset_mechanism="zero",
        reset_delay=False,
    )
    membrane = torch.tensor(params.reset_mv - params.rest_mv, dtype=torch.float64)
    neuron = neuron.double()
    refractory = 0
    voltages = np.empty(len(current))
    spikes = np.zeros(len(current), dtype=bool)
    with torch.no_grad():
        for i, value in enumerate(current):
            membrane, spikes[i], refractory = lif_step(
                neuron, membrane, value, refractory, params
            )
            voltages[i] = membrane.item() + params.rest_mv
    return voltages, spikes


def inputs(params, periodic=False, current_na=2.0):
    n = round(params.duration_ms / params.dt_ms)
    time_ms = np.arange(n) * params.dt_ms
    current = np.full(n, current_na)
    if periodic:
        current += 0.8 * np.cos(2 * np.pi * 5 * time_ms / 1000)
    return time_ms, current


def plot_signal(time_ms, current, voltage, spikes, params, output):
    window = min(len(spikes), max(1, round(50 / params.dt_ms)))
    rate = np.convolve(spikes.astype(float), np.ones(window), mode="same") / (
        window * params.dt_ms / 1000
    )
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(time_ms, current)
    axes[0].set_ylabel("Current (nA)")
    axes[1].plot(time_ms + params.dt_ms, voltage)
    axes[1].axhline(params.threshold_mv, color="red", linestyle="--")
    axes[1].set_ylabel("Voltage (mV)")
    axes[2].eventplot(time_ms[spikes] + params.dt_ms)
    axes[2].set_ylabel("Spikes")
    axes[3].plot(time_ms + params.dt_ms, rate)
    axes[3].set_ylabel("Rate (Hz)")
    axes[3].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_experiments(params, output):
    groups = {
        "threshold": [
            (str(value), replace(params, threshold_mv=value), 2.0)
            for value in (-55.0, -50.0, -45.0)
        ],
        "current": [(str(value), params, value) for value in (1.0, 1.5, 2.0, 2.5)],
        "tau": [
            (str(value), replace(params, tau_ms=value), 2.0)
            for value in (10.0, 20.0, 40.0)
        ],
    }
    summary = {}
    for name, cases in groups.items():
        fig, axes = plt.subplots(
            len(cases), 1, sharex=True, figsize=(10, 2.4 * len(cases))
        )
        summary[name] = []
        for axis, (value, case, current_na) in zip(axes, cases):
            time_ms, current = inputs(case, current_na=current_na)
            voltage, spikes = simulate(current, case)
            spike_times = time_ms[spikes] + case.dt_ms
            summary[name].append(
                {
                    "value": float(value),
                    "spikes": int(spikes.sum()),
                    "average_hz": float(spikes.sum() / (case.duration_ms / 1000)),
                    "first_spike_ms": float(spike_times[0])
                    if len(spike_times)
                    else None,
                }
            )
            axis.plot(time_ms + case.dt_ms, voltage, linewidth=0.8)
            axis.axhline(case.threshold_mv, color="red", linestyle="--", linewidth=0.8)
            axis.set_ylabel(f"{name}={value}\nVoltage (mV)")
        axes[-1].set_xlabel("Time (ms)")
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=160)
        plt.close(fig)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("lif/output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    params = Parameters()
    for kind in (False, True):
        time_ms, current = inputs(params, periodic=kind)
        voltage, spikes = simulate(current, params)
        plot_signal(
            time_ms,
            current,
            voltage,
            spikes,
            params,
            args.output / ("periodic.png" if kind else "constant.png"),
        )
    summary = run_experiments(params, args.output)
    print(json.dumps(summary, indent=2))
    print(f"Saved figures to {args.output}")


if __name__ == "__main__":
    main()

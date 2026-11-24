"""用作者的 PyTorch 参考递推比较固定门值 PDN 与作业原版 Delta。"""

import hashlib
import importlib.util
import json
import math
from itertools import repeat
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from associative_memory.experiment1.run import (
    BASE_SEED,
    KEY_DIMS,
    LEARNING_RATE,
    METRICS,
    PAIR_COUNTS,
    REPEATS,
    VALUE_DIM,
    measure,
    pair_tensors,
    sample_pairs,
    write_csv,
)

from . import OUTPUT_DIR
from .optimized_writes import evaluate, fit_delta
from .preconditioned_curves import METHODS, plot_accuracy, plot_memory

OUTPUT = OUTPUT_DIR / "preconditioned_delta"
COMMIT = "7bd753279af87b39114149a104c5bde9bf67145f"
REF_PATH = "3rdparty/flash-linear-attention/fla/ops/precond_gated_delta_rule/naive.py"
REF_BLOB = "4a2273f25a89b509332e8755391e78cc594a16b0"
X, EPS, CENTER = 1.5, 1e-6, -0.2
CHUNK = 20
PROFILE_PAIRS = 32


def load_official_rule():
    source = Path(__file__).with_name("reference") / "pdn" / "naive.py"
    content = source.read_bytes().replace(b"\r\n", b"\n")
    blob = hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()
    if blob != REF_BLOB:
        raise ValueError(f"Unexpected Git blob for the official write rule: {blob}")
    spec = importlib.util.spec_from_file_location("official_pdn_reference", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_recurrent_precond_gated_delta_rule


def make_inputs(d, p, index):
    return pair_tensors(*sample_pairs(d, p, index))


@torch.no_grad()
def official_write(rule, keys, values, state=(None, None), x=X):
    """每条流使用一个头。对零初始状态或流式传入的 S、A 继续写入。"""
    b, t, _ = keys.shape
    gates = keys.new_zeros(b, t, 1)
    qk = keys.unsqueeze(2)
    _, memory, diagonal = rule(
        q=qk,
        k=qk,
        v=values.unsqueeze(2),
        g=gates,
        g_atk=gates,
        beta=torch.full_like(gates, LEARNING_RATE),
        beta_atk=torch.ones_like(gates),
        scale=1.0,
        initial_state=state[0],
        initial_A_state=state[1],
        output_final_state=True,
        x=x,
        eps=EPS,
        log_atk_scale=CENTER,
    )
    return memory, diagonal


@torch.no_grad()
def scalar_control(keys, values):
    """独立的一阶 Delta 对照。单位双极键的每个坐标平方相同。"""
    batch, p, d = keys.shape
    memory = keys.new_zeros(batch, d, VALUE_DIM)
    for i in range(p):
        r = math.log((i + 1) / d + EPS) - CENTER
        step = LEARNING_RATE * math.exp(-math.log(X) * r / (1 + abs(r)))
        key = keys[:, i]
        prediction = torch.bmm(key.unsqueeze(1), memory).squeeze(1)
        memory.baddbmm_(
            key.unsqueeze(-1), (values[:, i] - prediction).unsqueeze(1), alpha=step
        )
    return torch.bmm(keys, memory)


def paired_interval(differences):
    gain = 100 * differences
    avg = gain.mean()
    half_width = 1.96 * gain.std(ddof=1) / np.sqrt(len(gain))
    return float(avg), float(avg - half_width), float(avg + half_width)


def compare(d, p, rule, audit):
    results = np.empty((REPEATS, len(METHODS), len(METRICS)))
    for start in range(0, REPEATS, CHUNK):
        indices = range(start, min(start + CHUNK, REPEATS))
        pairs = [make_inputs(d, p, i) for i in indices]
        keys, values = (torch.stack(items) for items in zip(*pairs))
        memory, diagonal = official_write(rule, keys, values)
        expected = torch.full_like(diagonal, p / d)
        torch.testing.assert_close(diagonal, expected, atol=1e-5, rtol=5e-6)
        if not torch.equal(diagonal, diagonal[..., :1].expand_as(diagonal)):
            raise AssertionError("The diagonal should be isotropic for bipolar keys")
        # 按作业协议在最后一次写入后读出所有记忆，逐时刻的官方输出不参与评分。
        scores = torch.bmm(keys, memory[:, 0])
        control = scalar_control(keys, values)
        torch.testing.assert_close(scores, control, atol=2e-5, rtol=1e-5)
        audit["max_abs_pdn_minus_scalar_score"] = max(
            audit["max_abs_pdn_minus_scalar_score"],
            (scores - control).abs().max().item(),
        )
        audit["scalar_bit_disagreements"] += int(
            ((scores >= 0) != (control >= 0)).sum()
        )
        audit["tested_bits"] += scores.numel()
        if start == 0:
            cut = p // 2
            first = official_write(rule, keys[:1, :cut], values[:1, :cut])
            resumed = official_write(rule, keys[:1, cut:], values[:1, cut:], first)
            for state, output_state in zip((memory, diagonal), resumed):
                torch.testing.assert_close(
                    output_state, state[:1], atol=2e-6, rtol=2e-6
                )
            neutral, _ = official_write(rule, keys[:1], values[:1], x=1.0)
        for j, (i, (k, v)) in enumerate(zip(indices, pairs)):
            delta = fit_delta(k, v, repeat(LEARNING_RATE, p))
            delta_metrics, delta_scores = evaluate(delta, k, v.numpy())
            if i == 0:
                np.testing.assert_allclose(
                    (k @ neutral[0, 0]).numpy(), delta_scores, atol=1e-5, rtol=1e-5
                )
            results[i, 0] = delta_metrics
            results[i, 1] = measure(
                scores[j].double().numpy(),
                v.numpy(),
                torch.linalg.vector_norm(memory[j, 0].double()).item(),
            )
    rows = []
    for j, method in enumerate(METHODS):
        row = dict(d=d, m=VALUE_DIM, P=p, load=p / d, method=method, repeats=REPEATS)
        for k, metric in enumerate(METRICS):
            row[f"{metric}_mean"] = results[:, j, k].mean()
            row[f"{metric}_sd"] = results[:, j, k].std(ddof=1)
        gains = paired_interval(
            results[:, j, 0] - results[:, METHODS.index("Delta"), 0]
        )
        for suffix, value in zip(("pp", "ci95_low_pp", "ci95_high_pp"), gains):
            row[f"gain_vs_delta_{suffix}"] = value
        rows.append(row)
    print(
        f"d={d:3} P={p:3}   Delta / PDN   "
        + "   ".join(f"{100 * row['bit_accuracy_mean']:.3f}%" for row in rows),
        flush=True,
    )
    return rows


def stream_write(method, rule, keys, values):
    if method == "Delta":
        return fit_delta(keys, values, repeat(LEARNING_RATE, len(keys)))
    if method != "PDN":
        raise ValueError(f"Unknown write method: {method}")
    state = (None, None)
    for key, value in zip(keys, values):
        state = official_write(rule, key.view(1, 1, -1), value.view(1, 1, -1), state)
    return state


def tensor_peak(method, rule, keys, values):
    """统计已有输入之外的 CPU 张量分配。覆盖状态初始化和每次只传一对的写入。"""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as prof:
        result = stream_write(method, rule, keys, values)
    events = sorted(
        (e for e in prof.profiler.kineto_results.events() if e.name() == "[memory]"),
        key=lambda e: e.start_ns(),
    )
    current = peak = 0
    for event in events:
        current += event.nbytes()
        peak = max(peak, current)
    # 用写完后仍存活的张量核对分配事件的净增量。
    if method == "PDN":
        live = list(result)
    else:
        live = [result.W]
        if result.W.grad is not None:
            live.append(result.W.grad)
    if not events or current != sum(t.numel() * t.element_size() for t in live):
        raise AssertionError(
            f"Inconsistent CPU allocation trace for {method}: {current}"
        )
    return peak


def measure_memory(rule):
    rows = []
    for d in KEY_DIMS:
        keys, values = make_inputs(d, PROFILE_PAIRS, 0)
        itemsize = keys.element_size()
        for method in METHODS:
            extra = d if method == "PDN" else 0
            stream_write(
                method, rule, keys[:2], values[:2]
            )  # 先预热，排除首次导入和优化器初始化
            peaks = [tensor_peak(method, rule, keys, values) for _ in range(2)]
            retained = itemsize * (d * VALUE_DIM + extra)
            if min(peaks) < retained or peaks[0] != peaks[1]:
                raise AssertionError(
                    f"Check memory profile for {method}, d={d}: {peaks}"
                )
            rows.append(
                dict(
                    d=d,
                    m=VALUE_DIM,
                    method=method,
                    dtype="float32",
                    weight_bytes=itemsize * d * VALUE_DIM,
                    extra_state_bytes=itemsize * extra,
                    persistent_write_bytes=retained,
                    single_stream_cpu_tensor_peak_bytes=peaks[0],
                    profile_pairs=PROFILE_PAIRS,
                    profile_batch=1,
                    profile_repeats=len(peaks),
                )
            )
    return rows


def main():
    torch.set_num_threads(1)
    rule = load_official_rule()
    audit = {
        "max_abs_pdn_minus_scalar_score": 0.0,
        "scalar_bit_disagreements": 0,
        "tested_bits": 0,
    }
    rows = [
        row for d in KEY_DIMS for p in PAIR_COUNTS for row in compare(d, p, rule, audit)
    ]
    memory = measure_memory(rule)
    info = dict(
        repository="https://github.com/ntumm120/preconditioned-deltanet",
        commit=COMMIT,
        official_path=REF_PATH,
        git_blob=REF_BLOB,
        official_function=rule.__name__,
        torch=str(torch.__version__),
        numpy=np.__version__,
        device="cpu",
        dtype="float32",
        threads=1,
        seed=BASE_SEED,
        repeats=REPEATS,
        keys=KEY_DIMS,
        pairs=PAIR_COUNTS,
        value_dim=VALUE_DIM,
        passes=1,
        scoring="all original full cues against the frozen final W",
        pdn=dict(
            heads=1,
            g=0,
            g_atk=0,
            beta=LEARNING_RATE,
            beta_atk=1,
            scale=1.0,
            x=X,
            eps=EPS,
            log_atk_scale=CENTER,
            initial_states="zero",
            trained_gates=False,
        ),
        delta_eta=LEARNING_RATE,
        methods=METHODS,
        baseline_check="covered by small deterministic regression tests",
        paired_interval="mean across 200 independent batches +/- 1.96 paired SE, per comparison",
        peak_measurement="single-stream CPU tensor allocation peak over 32 one-pair writes, two identical runs",
        validation=audit,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT) as tmp:
        staging = Path(tmp)
        write_csv(rows, staging / "accuracy_by_condition.csv")
        write_csv(memory, staging / "write_memory.csv")
        plot_accuracy(rows, staging / "bit_accuracy.png")
        plot_memory(memory, staging / "write_memory.png")
        (staging / "run_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for path in staging.iterdir():
            path.replace(OUTPUT / path.name)
    print(json.dumps(audit), flush=True)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()

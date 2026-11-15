"""Evaluate the published SpikCommander checkpoint at several SHD input bin widths."""

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import quote
from zipfile import ZipFile

import h5py
import numpy as np
import torch
from prepare_shd import DEFAULT_OUT, download_verified, file_digest
from torch.nn.utils.rnn import pad_sequence

MODEL_REVISION = "a51266caba285d52cc022012dc7fb979a426f7eb"
SJ_REVISION = "ea2f815c27da6f8f68f0f78095091576ccb6121e"
SJ_ARCHIVE = f"spikingjelly-{SJ_REVISION}"
WEIGHT = (
    "seed=312-shd-1depths-10ms-bins=5-lr_w=0.01-heads=8_Best_Acc_2025-07-17_10-58-08.pt"
)
WEIGHT_PATH = f"checkpoints/SHD/{WEIGHT}"
AUTHOR_FILES = {
    "best_config_SHD_former.py": "148c55d4e91d104af9086dc067fad17366d7780f",
    "spikcommder.py": "c24c83ceb4caa617178d1eafca69314285947ee5",
    "module/conv.py": "b89259894b6abcb3cf2fd7ecbfd9ab0743c0a4fe",
    "module/spikcommander_backbone.py": "6c37c26433e638c715ab8be7a23805a5cde857ab",
    WEIGHT_PATH: "55f1ac64dd134846f4669c788d1bd9be160c7e24",
}
KEY_PREFIXES = {"spike_embed.": "see.", "blocks.0.mlp.": "blocks.0.scrmlp."}
INPUT_WINDOWS_MS = (5, 10, 20)


def git_blob_hash(path):
    content = path.read_bytes()
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def prepare_reference(cache):
    author = cache / "author"
    source = (
        f"https://raw.githubusercontent.com/JackieWang9811/SCommander/{MODEL_REVISION}/"
    )
    for name, digest in AUTHOR_FILES.items():
        download_verified(source + quote(name), author / name, digest, git_blob_hash)
    archive = download_verified(
        f"https://codeload.github.com/fangwei123456/spikingjelly/zip/{SJ_REVISION}",
        cache / f"{SJ_ARCHIVE}.zip",
        "a4a396651e8e8578a96673cbc2e0a3e132fc0254b1355d7b426894652e5f8ff7",
    )
    return author, archive


def unpack_spikingjelly(archive, destination):
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            parts = PurePosixPath(member.filename).parts
            if (
                member.is_dir()
                or parts[:2] != (SJ_ARCHIVE, "spikingjelly")
                or ".." in parts
            ):
                continue
            target = destination.joinpath(*parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zipped.read(member))


def attach_forward_methods(backbone):
    # Use the two published forward bodies that were left at module scope.
    path = Path(backbone.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    signatures = [[arg.arg for arg in node.args.args] for node in functions]
    if signatures != [["self", "x", "attention_mask"], ["self", "x"]]:
        raise RuntimeError("Unexpected forward definitions in the author's backbone")
    for layer, node in zip(
        (backbone.MSTASA_v_branch, backbone.SCRMLP), functions, strict=True
    ):
        namespace = {}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            vars(backbone),
            namespace,
        )
        layer.forward = namespace["forward"]


def count_frames(times, units, bin_ms, channels_per_group):
    """Match SpikingJelly's duration integration and the author's channel grouping."""
    elapsed = 1000 * times
    elapsed -= elapsed[0]
    steps = int(np.ceil(elapsed[-1] / bin_ms))
    frame_index = np.minimum((elapsed // bin_ms).astype(np.int64), steps - 1)
    channels = 700 // channels_per_group
    grouped_units = units.astype(np.int64) // channels_per_group
    counts = np.bincount(
        frame_index * channels + grouped_units, minlength=steps * channels
    )
    if counts.sum() != len(times):
        raise RuntimeError("Integration changed the event count")
    return torch.as_tensor(counts.reshape(steps, channels), dtype=torch.float32)


def predict(model, frames, device):
    padded = pad_sequence(frames, batch_first=True).to(device)
    lengths = torch.tensor([len(sample) for sample in frames], device=device)
    mask = torch.arange(padded.size(1), device=device)[None, :] < lengths[:, None]
    logits = model(padded, mask)
    return logits.softmax(dim=2).sum(dim=0).argmax(dim=1)


@torch.inference_mode()
def evaluate(test_file, author, library, device):
    sys.path[:0] = [str(library), str(author)]
    from best_config_SHD_former import Config
    from module import spikcommander_backbone
    from spikcommder import SpikCommander
    from spikingjelly.activation_based import functional

    attach_forward_methods(spikcommander_backbone)
    config = Config()
    config.backend = "torch"
    config.n_hidden_neurons = config.n_hidden_neurons_list[0]
    config.hidden_dims = config.mlp_ratio * config.n_hidden_neurons
    model = SpikCommander(config).to(device)
    checkpoint = author / WEIGHT_PATH
    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    for key in list(weights):
        for old, new in KEY_PREFIXES.items():
            if key.startswith(old):
                weights[new + key.removeprefix(old)] = weights.pop(key)
                break
    model.load_state_dict(weights, strict=True)
    model.eval()

    window_results = []
    with h5py.File(test_file, "r") as data:
        labels = data["labels"][:].astype(np.int64)
        if len(labels) != 2264 or np.unique(labels).size != 20:
            raise ValueError("Expected the complete 20-class official SHD test split")
        ordered = np.argsort(labels, kind="stable")
        for window_ms in INPUT_WINDOWS_MS:
            correct = 0
            for start in range(0, len(ordered), config.batch_size):
                indices = ordered[start : start + config.batch_size]
                frames = [
                    count_frames(
                        data["spikes/times"][i],
                        data["spikes/units"][i],
                        window_ms,
                        config.n_bins,
                    )
                    for i in indices
                ]
                target = torch.from_numpy(labels[indices]).to(device)
                correct += int(predict(model, frames, device).eq(target).sum().item())
                functional.reset_net(model)
            window_results.append(
                {
                    "input_bin_ms": window_ms,
                    "correct": correct,
                    "samples": len(labels),
                    "accuracy": correct / len(labels),
                }
            )

    baseline = next(
        row for row in window_results if row["input_bin_ms"] == config.time_step
    )
    return {
        "model": "SpikCommander (1L-8-128)",
        "commit": MODEL_REVISION,
        "spikingjelly_commit": SJ_REVISION,
        "checkpoint": WEIGHT,
        "checkpoint_sha256": file_digest(checkpoint),
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_channels": 700 // config.n_bins,
        "batch_size": config.batch_size,
        "order": "label, then HDF5 sample index",
        **baseline,
        "window_results": window_results,
        "device": str(device),
        "neuron_backend": config.backend,
        "torch_version": str(torch.__version__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    test_file = out / "data" / "shd_test.h5"
    if not test_file.is_file():
        parser.error("Run prepare_shd.py with the same output directory first.")
    if args.threads < 1:
        parser.error("--threads must be positive")

    torch.set_num_threads(args.threads)
    cache = out / "spikcommander_reference"
    author, archive = prepare_reference(cache)
    with TemporaryDirectory(dir=cache) as directory:
        library = Path(directory)
        unpack_spikingjelly(archive, library)
        metrics = evaluate(test_file, author, library, torch.device(args.device))

    with NamedTemporaryFile(dir=out, suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        temporary.replace(out / "spikcommander.json")
    finally:
        temporary.unlink(missing_ok=True)
    for row in metrics["window_results"]:
        print(
            f"{row['input_bin_ms']:2d} ms  Top-1 accuracy: {row['accuracy']:.2%} "
            f"({row['correct']:,} / {row['samples']:,})",
            flush=True,
        )


if __name__ == "__main__":
    main()

"""Measure inference power after connecting an external meter."""

import hashlib
import json
from pathlib import Path

import torch
from utils.benchmark_common import create_parser, run
from utils.inference import inference_engine
from utils.io import atomic_write_json


def main(argv=None):
    parser = create_parser()
    artifacts = parser.add_mutually_exclusive_group()
    artifacts.add_argument("--save-compiled", type=Path)
    artifacts.add_argument("--load-compiled", type=Path)
    args = parser.parse_args(argv)
    if (args.save_compiled or args.load_compiled) and args.engine != "inductor":
        parser.error("compiled artifacts require --engine inductor")

    def prepare(predict, batch, device):
        if not (args.save_compiled or args.load_compiled):
            return inference_engine(predict, args.engine)
        metadata = {
            "model": args.model,
            "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            "batch_size": args.batch_size,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        }
        if args.save_compiled:
            args.save_compiled.mkdir(parents=True, exist_ok=False)
            compiled = torch.compile(
                predict, backend="inductor", fullgraph=True
            ).aot_compile(((batch,), {}))
            compiled.save_compiled_function(
                str(args.save_compiled / "compiled.pt"),
                external_data=predict.aot_external_data,
            )
            atomic_write_json(args.save_compiled / "metadata.json", metadata)
            print(json.dumps(metadata, indent=2))
            return None
        saved = json.loads(
            (args.load_compiled / "metadata.json").read_text(encoding="utf-8")
        )
        if saved != metadata:
            raise ValueError(
                "compiled artifact metadata does not match model, weights, batch or environment"
            )
        with (args.load_compiled / "compiled.pt").open("rb") as f:
            return torch.compiler.load_compiled_function(
                f, external_data=predict.aot_external_data
            )

    def wait_for_start():
        input("预热完成，请连接测试电源，然后按回车开始测量：")

    result_fields = (
        {"compiled_artifact": str(args.load_compiled)} if args.load_compiled else None
    )
    run(
        args,
        prepare=prepare,
        before_measure=wait_for_start,
        result_fields=result_fields,
    )


if __name__ == "__main__":
    main()

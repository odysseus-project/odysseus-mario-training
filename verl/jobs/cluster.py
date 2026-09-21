"""Small CPU/GPU probes used by the Slurm example; does not submit jobs."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    address = commands.add_parser("address")
    address.add_argument("hostname")
    probe = commands.add_parser("probe")
    probe.add_argument("--gpus", type=int, required=True)
    ready = commands.add_parser("wait")
    ready.add_argument("--nodes", type=int, required=True)
    ready.add_argument("--gpus", type=int, required=True, help="GPUs per node")
    ready.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.command == "address":
        print(socket.gethostbyname(args.hostname))
        return
    if args.command == "probe":
        import torch
        import verl

        expected = Path(__file__).resolve().parents[1] / "verl/__init__.py"
        if Path(verl.__file__).resolve() != expected:
            raise RuntimeError(f"Wrong verl checkout: {verl.__file__}; expected {expected}")
        if torch.cuda.device_count() < args.gpus:
            raise RuntimeError(f"{socket.gethostname()}: need {args.gpus} GPUs, found {torch.cuda.device_count()}")
        print(json.dumps({"host": socket.gethostname(), "verl": str(expected),
                          "gpus": torch.cuda.device_count(), "torch_cuda": torch.version.cuda,
                          "versions": {name: importlib.metadata.version(name)
                                       for name in ("torch", "vllm", "ray", "transformers", "flash-attn")}}))
        return

    import ray

    deadline = time.monotonic() + args.timeout
    ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
    try:
        while time.monotonic() < deadline:
            nodes = [node for node in ray.nodes() if node["Alive"]]
            if len(nodes) == args.nodes and all(node["Resources"].get("GPU", 0) >= args.gpus for node in nodes):
                print(f"Ray ready: {len(nodes)} nodes, at least {args.gpus} GPUs each")
                return
            time.sleep(2)
        raise TimeoutError(f"Ray cluster did not reach {args.nodes} nodes with {args.gpus} GPUs each")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

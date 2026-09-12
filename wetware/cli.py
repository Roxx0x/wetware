"""Command line interface. `wetware --help`."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .data import download, load, load_sample
from .tasks import TASKS


def _brain(sample: bool):
    return load_sample() if sample else load()


def cmd_download(a) -> int:
    path = download(force=a.force)
    print(f"connectome cached at {path}")
    return 0


def cmd_info(a) -> int:
    W, ids = _brain(a.sample)
    n = W.shape[0]
    nnz = int(np.count_nonzero(W))
    print(json.dumps({
        "source": "synthetic sample" if a.sample else "larval Drosophila connectome (Winding 2023)",
        "neurons": n,
        "connections": nnz,
        "density": round(nnz / (n * n), 4),
        "mean_synapses_per_connection": round(float(W[W != 0].mean()), 2),
    }, indent=2))
    return 0


def cmd_run(a) -> int:
    if a.task not in TASKS:
        raise SystemExit(f"unknown task {a.task!r}; choose from {', '.join(TASKS)}")
    W, _ = _brain(a.sample)
    result = TASKS[a.task](W)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wetware", description="Compute on a real fruit-fly brain.")
    p.add_argument("--sample", action="store_true", help="use the offline synthetic stand-in, not the real brain")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch and cache the real connectome")
    d.add_argument("--force", action="store_true")
    d.set_defaults(fn=cmd_download)

    i = sub.add_parser("info", help="connectome stats")
    i.set_defaults(fn=cmd_info)

    r = sub.add_parser("run", help="train a readout on a task and report the score")
    r.add_argument("task", choices=list(TASKS))
    r.set_defaults(fn=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""`python -m photoniq {audit,train}` command line entry point."""
import argparse
import json
import tempfile
from pathlib import Path
from .data import file_sha256, load_experimental, unpack_source
from .train import run_experiment


def main():
    p = argparse.ArgumentParser(prog='photoniq', description='Real-data UV–Vis thin-film ML; no simulated training data.')
    sub = p.add_subparsers(dest='action', required=True)
    for key in ('audit', 'train'):
        s = sub.add_parser(key)
        s.add_argument('--archive', type=Path, required=True, help='Original Mendeley Download All ZIP (v3)')
        s.add_argument('--output', type=Path, default=Path('runs/photoniq_v1'))
        s.add_argument('--seed', type=int, default=42)
        s.add_argument('--grid-points', type=int, default=128)
        s.add_argument('--min-match-fraction', type=float, default=.65)
        if key == 'train':
            s.add_argument('--epochs', type=int, default=150)
            s.add_argument('--patience', type=int, default=20)
            s.add_argument('--bootstrap-iterations', type=int, default=1000)
            s.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
    args = p.parse_args()
    if args.action == 'audit':
        with tempfile.TemporaryDirectory(prefix='photoniq_audit_') as tmp:
            unpack_source(args.archive, Path(tmp) / 'raw')
            d = load_experimental(Path(tmp)/'raw', args.grid_points, args.min_match_fraction)
            print(json.dumps(dict(**d.audit, source_sha256=file_sha256(args.archive)), indent=2))
    else:
        run_experiment(args.archive, args.output, epochs=args.epochs, patience=args.patience,
                       seed=args.seed, grid_points=args.grid_points,
                       min_match_fraction=args.min_match_fraction,
                       n_boot=args.bootstrap_iterations, device=args.device)


if __name__ == '__main__':
    main()

"""CLI entry point for PLS analysis."""

import argparse
import logging
import sys
from pathlib import Path


METHOD_ALIASES = {
    "c": "correlational",
    "correlational": "correlational",
    "d": "discriminatory",
    "discriminatory": "discriminatory",
}


def _error(message: str) -> None:
    """Print error message to stderr and exit with code 2."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


def pls_main(argv=None):
    """PLS covariance analysis with statistical testing and visualisation.

    Args:
        argv: Command-line arguments. None uses sys.argv (normal CLI usage);
              pass a list for testing.
    """
    # Shared flags available on every subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Enable verbose logging output")

    parser = argparse.ArgumentParser(
        prog="plsdo",
        description="PLS covariance analysis with statistical testing and visualisation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- plsdo run ---
    run_parser = subparsers.add_parser("run", help="Run PLS analysis",
                                       parents=[common])
    run_parser.add_argument("--method", "-m", required=True,
                            help="correlational/c or discriminatory/d")
    run_parser.add_argument("--x", dest="x_path", default=None,
                            help="X matrix CSV (required for correlational)")
    run_parser.add_argument("--y", dest="y_path", required=True,
                            help="Y matrix CSV")
    run_parser.add_argument("--demographics", required=True,
                            help="Demographics CSV")
    run_parser.add_argument("--output", required=True, help="Output directory")
    run_parser.add_argument("--group-col", default=None,
                            help="Group column name (shorthand for YAML)")
    run_parser.add_argument("--groups", dest="groups_path", default=None,
                            help="Groups YAML config file")
    run_parser.add_argument("--subject-id", default=None,
                            help="Subject ID column name")
    run_parser.add_argument("--x-meta", default=None,
                            help="X metadata CSV")
    run_parser.add_argument("--y-meta", default=None,
                            help="Y metadata CSV")
    run_parser.add_argument("--n-perms", default=10000, type=int,
                            help="Number of permutations (default: 10000)")
    run_parser.add_argument("--n-bootstraps", default=10000, type=int,
                            help="Number of bootstrap resamples (default: 10000)")
    run_parser.add_argument("--seed", default=42, type=int,
                            help="Random seed (default: 42)")
    run_parser.add_argument("--all-plots", action="store_true", default=False,
                            help="Generate all plots including diagnostics")
    run_parser.add_argument("--format", dest="img_format", default="svg",
                            choices=["svg", "png"],
                            help="Image format (default: svg)")
    run_parser.add_argument("--dpi", default=300, type=int,
                            help="Image DPI (default: 300)")

    # --- plsdo cross-validate ---
    cv_parser = subparsers.add_parser("cross-validate",
                                       help="Cross-validate discriminatory PLS model",
                                       parents=[common])
    cv_parser.add_argument("--y", dest="y_path", required=True,
                           help="Y matrix CSV")
    cv_parser.add_argument("--demographics", required=True,
                           help="Demographics CSV")
    cv_parser.add_argument("--output", required=True, help="Output directory")
    cv_parser.add_argument("--group-col", required=True,
                           help="Group column name")
    cv_parser.add_argument("--subject-id", default=None,
                           help="Subject ID column name")
    cv_parser.add_argument("--n-folds", default=5, type=int,
                           help="Number of CV folds (default: 5)")
    cv_parser.add_argument("--n-repeats", default=100, type=int,
                           help="Number of CV repeats (default: 100)")
    cv_parser.add_argument("--n-components", default=None, type=int,
                           help="Number of PLS components (default: n_groups - 1)")
    cv_parser.add_argument("--n-permutations", default=1000, type=int,
                           help="Number of permutations for CV test (default: 1000)")
    cv_parser.add_argument("--seed", default=42, type=int,
                           help="Random seed (default: 42)")
    cv_parser.add_argument("--all-plots", action="store_true", default=False,
                           help="Generate all plots including diagnostics")
    cv_parser.add_argument("--format", dest="img_format", default="svg",
                           choices=["svg", "png"],
                           help="Image format (default: svg)")
    cv_parser.add_argument("--dpi", default=300, type=int,
                           help="Image DPI (default: 300)")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(1)
    elif args.command == "run":
        _dispatch_run(args)
    elif args.command == "cross-validate":
        _dispatch_cross_validate(args)


def _dispatch_run(args):
    """Validate run arguments and dispatch to pipeline."""
    from plsdo.pipeline import run_pipeline

    # --- Resolve method ---
    method_lower = args.method.lower()
    if method_lower not in METHOD_ALIASES:
        _error(
            f"Unknown method '{args.method}'. "
            f"Use correlational/c or discriminatory/d."
        )
    method = METHOD_ALIASES[method_lower]

    # --- Validate method-specific constraints ---
    if method == "correlational" and args.x_path is None:
        _error("Correlational PLS requires --x.")
    if method == "discriminatory" and args.x_path is not None:
        _error(
            "Discriminatory PLS builds X from --group-col. "
            "Do not provide --x."
        )
    if method == "discriminatory" and args.group_col is None and args.groups_path is None:
        _error(
            "Discriminatory PLS requires --group-col or --groups."
        )
    if args.group_col is not None and args.groups_path is not None:
        _error(
            "--group-col and --groups are mutually exclusive."
        )

    run_pipeline(
        method=method,
        y_path=Path(args.y_path),
        demographics_path=Path(args.demographics),
        output_dir=Path(args.output),
        x_path=Path(args.x_path) if args.x_path else None,
        group_col=args.group_col,
        groups_path=Path(args.groups_path) if args.groups_path else None,
        subject_id=args.subject_id,
        x_meta_path=Path(args.x_meta) if args.x_meta else None,
        y_meta_path=Path(args.y_meta) if args.y_meta else None,
        n_perms=args.n_perms,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
        img_format=args.img_format,
        dpi=args.dpi,
        all_plots=args.all_plots,
    )


def _dispatch_cross_validate(args):
    """Validate cross-validate arguments and dispatch to pipeline."""
    from plsdo.pipeline import cross_validate_pipeline

    cross_validate_pipeline(
        y_path=Path(args.y_path),
        demographics_path=Path(args.demographics),
        output_dir=Path(args.output),
        group_col=args.group_col,
        subject_id=args.subject_id,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        n_components=args.n_components,
        n_permutations=args.n_permutations,
        seed=args.seed,
        img_format=args.img_format,
        dpi=args.dpi,
        all_plots=args.all_plots,
    )

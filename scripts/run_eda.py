"""
Phase 1 entry point: load -> validate -> EDA.

Usage:
    python scripts/run_eda.py --data-path data/raw/your_dataset.csv --target target
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_default_config
from src.data.loader import load_csv
from src.data.validator import DataValidator, save_quality_report
from src.eda.analysis import generate_eda_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 loading, validation, and EDA.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = parse_args()
    config = get_default_config()

    data_path = Path(args.data_path) if args.data_path else config.data.raw_data_path
    target_column = args.target or config.data.target_column
    output_dir = Path(args.output_dir) if args.output_dir else config.reporting.eda_output_dir

    df = load_csv(data_path)

    validator = DataValidator(
        target_column=target_column, valid_target_values=config.data.valid_target_values
    )
    report = validator.generate_quality_report(df)
    save_quality_report(report, output_dir / "data_quality_report.json")

    if report.has_errors():
        raise SystemExit(f"Data quality errors found, aborting EDA: {report.errors}")

    generate_eda_report(
        df=df,
        target_column=target_column,
        numerical_columns=report.numerical_columns,
        categorical_columns=report.categorical_columns,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
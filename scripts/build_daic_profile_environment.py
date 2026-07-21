from __future__ import annotations

import argparse
import json
from pathlib import Path

from normalize_daic_phq8_environment import (
    DEFAULT_DAIC_DIR,
    DEFAULT_SCHEMA_PATH,
    normalize_environment,
    write_readme,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the DAIC profile-grounded environment using the same artifact "
            "contract as MDD-5K: patient profiles, split group files, canonical "
            "evidence units, and surface-to-canonical links. DAIC uses exactly "
            "the eight PHQ-8 slots and Depressed/control labels."
        )
    )
    parser.add_argument("--daic-dir", type=Path, default=DEFAULT_DAIC_DIR)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = normalize_environment(args.daic_dir, args.schema)
    write_readme(args.daic_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Concatenate TSVs that share the same header.

Usage:
  python postprocess_output.py /path/to/dir PREFIX [-o /path/to/output.tsv]

Notes:
- Searches ONLY the given directory (non-recursive).
- Matches files whose names start with PREFIX and end with .tsv
  e.g., PREFIX_1.tsv, PREFIX_foo.tsv
- Assumes all inputs share the same first-line header; warns if not.
- Overwrites the output file if it exists.
"""

import argparse
import sys
from pathlib import Path


def find_tsvs(input_dir: Path, prefix: str):
    """Return a sorted list of .tsv files in input_dir whose names start with prefix."""
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix == ".tsv" and p.name.startswith(prefix)
    )


def concat_tsvs(input_dir: Path, prefix: str, output_path: Path) -> int:
    """Concatenate matching TSVs into output_path. Returns number of files processed."""
    files = find_tsvs(input_dir, prefix)
    if not files:
        raise SystemExit(f"No .tsv files in '{input_dir}' starting with '{prefix}'.")

    # Stream-write to avoid loading large files into memory
    with output_path.open("w", encoding="utf-8") as out:
        # Write header + body from the first file
        with files[0].open("r", encoding="utf-8") as f0:
            header = f0.readline()
            if header == "":
                raise SystemExit(f"First file is empty: {files[0]}")
            out.write(header)
            for line in f0:
                out.write(line)

        # Append body (skip header) from the rest
        for path in files[1:]:
            with path.open("r", encoding="utf-8") as f:
                h = f.readline()
                if h != header:
                    sys.stderr.write(
                        f"Warning: header in {path.name} differs from the first file; using the first header.\n"
                    )
                for line in f:
                    out.write(line)

    return len(files)


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate TSV files with shared header."
    )
    parser.add_argument("path", type=Path, help="Directory containing the TSV files")
    parser.add_argument("prefix", help="Filename prefix to match (files must start with this)")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .tsv path (default: <path>/<prefix>_classified.tsv)"
    )
    args = parser.parse_args()

    input_dir = args.path
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    output_path = args.output or (input_dir / f"{args.prefix}_classified.tsv")

    try:
        count = concat_tsvs(input_dir, args.prefix, output_path)
    except KeyboardInterrupt:
        raise SystemExit("\nAborted.")
    print(f"Concatenated {count} file(s) → {output_path}")

if __name__ == "__main__":
    main()


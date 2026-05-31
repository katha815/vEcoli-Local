#!/usr/bin/env python3
"""Render Mermaid (.mmd) files to SVG/PNG via Kroki (no Docker/Node required).
Sample usage:
    bash -lc 'cd /home/katha/dev/wcEcoli/visualization && python render_mermaid.py causality_flowchart.mmd -f png'
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.error
import urllib.request


def render_mermaid(
    input_path: pathlib.Path, output_path: pathlib.Path, output_format: str
) -> None:
    source = input_path.read_text(encoding="utf-8")
    url = f"https://kroki.io/mermaid/{output_format}"
    request = urllib.request.Request(
        url,
        data=source.encode("utf-8"),
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "Accept": "image/svg+xml,image/png,*/*",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kroki HTTP {error.code}: {message}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Network error contacting Kroki: {error}") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Mermaid file to SVG/PNG using Kroki",
        epilog=(
            "Sample usage:\n"
            "  bash -lc 'cd /home/katha/dev/wcEcoli/visualization && "
            "python render_mermaid.py causality_flowchart.mmd -f png'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=pathlib.Path, help="Input .mmd file")
    parser.add_argument(
        "-f",
        "--format",
        choices=("svg", "png"),
        default="svg",
        help="Output image format (default: svg)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Output file path (defaults to input stem + selected extension)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output or input_path.with_suffix(f".{args.format}")

    try:
        render_mermaid(input_path, output_path, args.format)
    except Exception as error:
        print(f"Failed to render Mermaid: {error}", file=sys.stderr)
        return 1

    print(f"Rendered: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

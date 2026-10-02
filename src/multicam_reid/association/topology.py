"""Explicit, undirected overlap permissions shared by replay and live CLIs."""

from argparse import ArgumentParser
from collections.abc import Sequence


def validate_overlap_pairs(pairs: Sequence[tuple[str, str]], camera_ids: Sequence[str]) -> None:
    """Reject typoed or disabled cameras rather than silently weakening the policy."""
    selected = set(camera_ids)
    if any(source not in selected or target not in selected for source, target in pairs):
        raise ValueError("overlap pairs must reference selected camera IDs")


def parse_overlap_pairs(
    values: Sequence[str],
    camera_ids: Sequence[str],
    *,
    numeric: bool = False,
) -> tuple[tuple[str, str], ...]:
    """Parse ``--overlap 2,3 4,5`` or ``--overlap PHONE,LAPTOP``.

    WiseNET's numeric shorthand is converted to its internal C-prefixed IDs.
    Permissions are symmetric but deliberately not transitive.
    """
    pairs = set()
    for value in values:
        parts = tuple(part.strip() for part in value.split(","))
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"invalid overlap pair {value!r}; use CAMERA_A,CAMERA_B")
        if numeric:
            parts = tuple(f"C{int(part)}" if part.isdigit() else part for part in parts)
        if parts[0] == parts[1]:
            raise ValueError("overlap pairs must contain two different cameras")
        pairs.add(tuple(sorted(parts)))
    result = tuple(sorted(pairs))
    validate_overlap_pairs(result, camera_ids)
    return result


def add_overlap_arguments(parser: ArgumentParser) -> None:
    """Use consistent opt-in syntax across all surface entry points."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--overlap",
        nargs="+",
        action="extend",
        metavar="A,B",
        help="Allow simultaneous matching only for these camera pairs (e.g. 2,3 4,5).",
    )
    group.add_argument(
        "--no-overlap",
        dest="overlap",
        action="store_const",
        const=[],
        help="Clear configured overlap pairs and use strict handoff matching.",
    )
    parser.add_argument(
        "--no-reconcile",
        action="store_true",
        help="Keep overlap matching but disable reconsideration of already-issued IDs.",
    )

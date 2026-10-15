"""Shared SPICE parsing and wire-net aliases for physical source views."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

WIRE_R = re.compile(r"^R(wire|R)_", re.I)
WIRE_C = re.compile(r"^C(wire|Cg)_", re.I)


@dataclass
class Subckt:
    name: str
    pins: List[str]
    parent: Optional["Subckt"]
    lines: List[List[str]] = field(default_factory=list)
    children: Dict[str, "Subckt"] = field(default_factory=dict)

    def resolve(self, name: str) -> "Subckt":
        scope: Optional[Subckt] = self
        while scope is not None:
            if name in scope.children:
                return scope.children[name]
            scope = scope.parent
        raise SystemExit(f"error: subcircuit {name} not defined in scope of {self.name}")


def parse(text: Iterable[str]) -> Subckt:
    """Parse PySpice deck lines into a tree of (possibly nested) subcircuits."""
    root = Subckt("<top>", [], None)
    cur = root
    logical: List[str] = []
    for raw in text:
        raw = raw.rstrip("\n")
        if raw.startswith("+") and logical:
            logical[-1] += " " + raw[1:]
        else:
            logical.append(raw)
    for line in logical:
        tok = line.split()
        if not tok or tok[0].startswith("*"):
            continue
        key = tok[0].lower()
        if key == ".subckt":
            child = Subckt(tok[1], tok[2:], cur)
            cur.children[tok[1]] = child
            cur = child
        elif key == ".ends":
            assert cur.parent is not None, "unbalanced .ends"
            cur = cur.parent
        elif not key.startswith("."):
            cur.lines.append(tok)
    return root


_PARSED: Dict[tuple, Subckt] = {}


def parse_file(path) -> Subckt:
    """Parse a deck once per file state; large banks read hundreds of views.

    Only the most recent deck is retained, which bounds memory to one tree.
    """
    from pathlib import Path
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _PARSED:
        _PARSED.clear()
        _PARSED[key] = parse(path.read_text().splitlines())
    return _PARSED[key]


def instance_parts(tok: List[str]) -> Tuple[List[str], str]:
    if any("=" in t for t in tok[1:]):
        raise SystemExit(f"error: instance parameters are not supported: {' '.join(tok)}")
    return tok[1:-1], tok[-1]


class Nets:
    """Union-find of net names merged by shorted wiring resistors."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, n: str) -> str:
        self.parent.setdefault(n, n)
        while self.parent[n] != n:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for n in list(self.parent):
            out.setdefault(self.find(n), []).append(n)
        return out

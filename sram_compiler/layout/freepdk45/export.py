#!/usr/bin/env python3
"""Export the SRAM macro of an OpenYield deck as a pure schematic (LVS) netlist.

OpenYield decks always carry interconnect parasitics: distributed wordline,
bitline and control-line RC (``Rwire_*``/``Cwire_*``) and local pin stubs
(``RR_*``/``CCg_*``). This script removes them: wiring resistors become shorts
(their nets are merged), wiring capacitors are dropped. Any other resistor or
capacitor inside the macro is an error, so an intentional R or C is never
silently shorted.

The macro is the array, replica column, replica wordline loads, row decoder,
wordline drivers, write-data buffers, precharge, write drivers and sense
amplifiers. TIME_CONTROL (with the address and data registers), the testbench
sources and the startup clamps stay outside; nets they share with the macro
become ports.

Usage (deck from ``python -m sram_compiler.per_device_mc.run --operation write``):

    python -m sram_compiler.layout.freepdk45.export deck.sp sram8x8_macro.sp
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

WIRE_R = re.compile(r"^R(wire|R)_", re.I)
WIRE_C = re.compile(r"^C(wire|Cg)_", re.I)
# Top-level instances that form the macro (name -> pattern).
MACRO_INSTANCES = [
    r"XSRAM_6T_CORE_\d+x\d+", r"Xsram_\d+x1_replica_column", r"XRWL", r"XRWL_LOAD_\d+",
    r"XDECODER", r"XWL_DRV_\d+", r"XDIN_BUF_\d+", r"XPRECHARGE_(\d+|RBL)",
    r"XWRITEDRIVER_\d+", r"XREPLICA_WDRV_LOAD", r"XSENSEAMP_\d+", r"XREPLICA_SENSEAMP",
]
# Top-level elements that are deliberately left outside the macro.
OUTSIDE = [r"XTIME_CONTROL", r"MCSINIT_[NP]", r"V\w+", r"Rwire_\w+", r"Cwire_\w+"]
# Macro port name for each merged OpenYield net; order is the port order.
PORTS = [
    (r"VDD", "VDD"), (r"VSS", "VSS"),
    (r"A_dff(\d+)", r"A\1"), (r"wl_en", "WL_EN"), (r"PRE", "PRE"),
    (r"s_en", "S_EN"), (r"sa_iso", "SA_ISO"), (r"w_en", "W_EN"),
    (r"DIN_dff(\d+)", r"DIN\1"), (r"SA_Q(\d+)", r"Q\1"), (r"SA_QB(\d+)", r"QB\1"),
    (r"RBL", "RBL"), (r"RWL", "RWL"),
]
# Suffixes OpenYield adds to the segments of one wire.
WIRE_SUFFIX = re.compile(r"(_line)?(_periph)?(_(tap|wire)\d+|_far|_seg\d+|_end)?$")


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


def strip_body(sub: Subckt, keep: Sequence[List[str]]) -> Tuple[Dict[str, str], List[List[str]]]:
    """Short wiring R and drop wiring C in ``keep``; return net renaming and devices."""
    nets = Nets()
    devices: List[List[str]] = []
    for tok in keep:
        kind = tok[0][0].upper()
        if kind == "R":
            if not WIRE_R.match(tok[0]):
                raise SystemExit(f"error: non-wiring resistor in {sub.name}: {' '.join(tok)}")
            nets.union(tok[1], tok[2])
        elif kind == "C":
            if not WIRE_C.match(tok[0]):
                raise SystemExit(f"error: non-wiring capacitor in {sub.name}: {' '.join(tok)}")
        elif kind in "MX":
            devices.append(tok)
        else:
            raise SystemExit(f"error: unsupported element in {sub.name}: {' '.join(tok)}")
    rename: Dict[str, str] = {}
    for members in nets.groups().values():
        pins = [m for m in members if m in sub.pins]
        if len(pins) > 1:
            raise SystemExit(f"error: wiring shorts pins {pins} of {sub.name}")
        if pins:
            rep = pins[0]
        else:
            bases = sorted({WIRE_SUFFIX.sub("", m) for m in members} & set(members))
            if len(bases) != 1:
                raise SystemExit(f"error: cannot name merged net {sorted(members)} in {sub.name}")
            rep = bases[0]
        for m in members:
            rename[m] = rep
    return rename, devices


def nm(value: str) -> str:
    """Format a SPICE length (``1.35e-07`` or ``135n``) as whole nanometres."""
    scale = {"u": 1e-6, "n": 1e-9}.get(value[-1:].lower(), 1.0)
    nano = float(value[:-1] if scale != 1.0 else value) * scale * 1e9
    if abs(nano - round(nano)) > 1e-6:
        raise SystemExit(f"error: size {value} is not a whole nanometre")
    return f"{int(round(nano))}n"


class Emitter:
    """Flatten nested definitions into unique, structurally deduplicated subcircuits.

    PySpice scopes subcircuits, so one name (e.g. PINV) can carry different
    sizes in different parents. A name with one structure keeps its name; a name
    with several structures is qualified by its parent (WORDLINEDRIVER_PINV).
    """

    def __init__(self) -> None:
        self.index: Dict[Tuple, int] = {}        # signature -> definition number
        self.defs: List[Tuple[Subckt, List[Tuple[str, Optional[int]]]]] = []
        self.seen: Dict[int, int] = {}           # id(Subckt) -> definition number

    def visit(self, sub: Subckt) -> int:
        if id(sub) in self.seen:
            return self.seen[id(sub)]
        rename, devices = strip_body(sub, sub.lines)
        body: List[Tuple[str, Optional[int]]] = []
        for tok in devices:
            if tok[0][0] in "Mm":
                r = [rename.get(n, n) for n in tok[1:5]]
                if "0" in r:
                    raise SystemExit(f"error: device on global node 0 in {sub.name}")
                params = dict(p.lower().split("=", 1) for p in tok[6:])
                if set(params) != {"w", "l"}:
                    raise SystemExit(f"error: unexpected MOS parameters in {sub.name}: {tok}")
                body.append((f"{tok[0]} {' '.join(r)} {tok[5]} w={nm(params['w'])} l={nm(params['l'])}", None))
            else:
                conns, ref = instance_parts(tok)
                child = self.visit(sub.resolve(ref))
                body.append((f"{tok[0]} {' '.join(rename.get(n, n) for n in conns)}", child))
        sig = (tuple(sub.pins), tuple(body))
        if sig not in self.index:
            self.index[sig] = len(self.defs)
            self.defs.append((sub, body))
        self.seen[id(sub)] = self.index[sig]
        return self.index[sig]

    def names(self) -> List[str]:
        count = Counter(sub.name for sub, _ in self.defs)
        out: List[str] = []
        for sub, _ in self.defs:
            name = sub.name
            if count[name] > 1 and sub.parent is not None and sub.parent.parent is not None:
                name = f"{sub.parent.name}_{sub.name}"
            base, k = name, 2
            while name in out:
                name, k = f"{base}_{k}", k + 1
            out.append(name)
        return out

    def render(self, names: List[str]) -> List[str]:
        text: List[str] = []
        for (sub, body), name in zip(self.defs, names):
            text.append(f".subckt {name} {' '.join(sub.pins)}")
            text += [line if child is None else f"{line} {names[child]}" for line, child in body]
            text += [f".ends {name}", ""]
        return text


def flat_devices(inst_lines: Sequence[List[str]], scope: Subckt) -> Counter:
    """Count MOS (model, w, l) under ``inst_lines``, resolving the original tree."""
    out: Counter = Counter()
    for tok in inst_lines:
        if tok[0][0] in "Mm":
            p = dict(x.lower().split("=", 1) for x in tok[6:])
            out[(tok[5], nm(p["w"]), nm(p["l"]))] += 1
        elif tok[0][0] in "Xx":
            child = scope.resolve(instance_parts(tok)[1])
            out += flat_devices(child.lines, child)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("deck")
    ap.add_argument("output")
    ap.add_argument("--name", help="macro subcircuit name (default SRAM_<rows>X<cols>)")
    args = ap.parse_args()

    with open(args.deck, encoding="utf-8") as fh:
        root = parse(fh)
    macro, outside = [], []
    for tok in root.lines:
        if any(re.fullmatch(p, tok[0]) for p in MACRO_INSTANCES):
            macro.append(tok)
        elif any(re.fullmatch(p, tok[0]) for p in OUTSIDE):
            outside.append(tok)
        else:
            raise SystemExit(f"error: top-level element is neither macro nor outside: {tok[0]}")
    core = next((t for t in macro if t[0].startswith("XSRAM_6T_CORE_")), None)
    if core is None:
        raise SystemExit("error: no XSRAM_6T_CORE_<rows>x<cols> instance in the deck")
    rows, cols = re.fullmatch(r"XSRAM_6T_CORE_(\d+)x(\d+)", core[0]).groups()
    name = args.name or f"SRAM_{rows}X{cols}"

    wires = [t for t in outside if t[0][0] in "RC"]
    rename, _ = strip_body(Subckt("<top>", [], None), wires)
    by_name = {t[0]: t for t in macro}

    def top_net(n: str) -> str:
        if ":" in n:  # Xyce hierarchical node seen from outside: XINST:node
            inst, node = n.split(":", 1)
            if inst not in by_name:
                raise SystemExit(f"error: {n} refers into an instance outside the macro")
            conns, ref = instance_parts(by_name[inst])
            child = root.resolve(ref)
            child_rename, _ = strip_body(child, child.lines)
            pin = child_rename.get(node, node)
            if pin not in child.pins:
                raise SystemExit(f"error: {n} is internal to {ref}, not a pin")
            n = conns[child.pins.index(pin)]
        return rename.get(n, n)

    inside_nets = {top_net(n) for t in macro for n in instance_parts(t)[0]}
    touched = {top_net(n) for t in outside if t[0][0] not in "RC"
               for n in (instance_parts(t)[0] if t[0][0] in "Xx" else t[1:5] if t[0][0] in "Mm" else t[1:3])}
    port_map: Dict[str, str] = {}
    for pattern, repl in PORTS:
        for net in sorted(inside_nets, key=lambda s: (len(s), s)):
            if re.fullmatch(pattern, net):
                port_map[net] = re.sub(pattern, repl, net)
    missing = sorted((touched & inside_nets) - set(port_map))
    if missing:
        raise SystemExit(f"error: nets shared with the outside are not ports: {missing}")

    em = Emitter()
    children = [em.visit(root.resolve(instance_parts(tok)[1])) for tok in macro]
    names = em.names()
    body = []
    for tok, child in zip(macro, children):
        nets = [port_map.get(top_net(n), top_net(n)) for n in instance_parts(tok)[0]]
        body.append(f"{tok[0]} {' '.join(nets)} {names[child]}")
    ports = list(port_map.values())
    text = em.render(names) + [f".subckt {name} {' '.join(ports)}", *body, f".ends {name}"]

    # Device-count cross-check against the unstripped tree.
    expected = flat_devices(macro, root)
    emitted_root = parse(text)
    got = flat_devices(emitted_root.children[name].lines, emitted_root)
    if got != expected:
        raise SystemExit(f"error: device mismatch after stripping: {expected - got} / {got - expected}")

    header = [
        f"* {name}: OpenYield SRAM macro, schematic netlist without interconnect RC",
        f"* source deck: {os.path.basename(os.path.dirname(os.path.abspath(args.deck)))}/"
        f"{os.path.basename(args.deck)}",
        f"* {sum(got.values())} MOS; widths and lengths in nm; models NMOS_VTG / PMOS_VTG",
        "* ports: " + " ".join(ports),
        "",
    ]
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header + text + [".end", ""]))
    print(f"wrote {args.output}: {name}, {len(ports)} ports, {sum(got.values())} MOS")
    for (model, w, l), n in sorted(got.items()):
        print(f"  {model:9s} w={w:>6s} l={l}: {n}")


if __name__ == "__main__":
    main()

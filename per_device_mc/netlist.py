"""Generate independent model cards for every MOS in an OpenYield SPICE deck."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MC_PARAMS = ("vth0", "u0", "voff")


class SpiceParseError(RuntimeError):
    """Raised when the generated deck cannot be specialized safely."""


@dataclass
class Subckt:
    name: str
    ports: list[str]
    body: list[str] = field(default_factory=list)
    children: dict[str, "Subckt"] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelClone:
    model_name: str
    base_model: str
    hier_path: str
    mos_name: str
    subckt_name: str


def _remove_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        lines.append(stripped)
    return " ".join(lines)


def _convert_value(value: str) -> Any:
    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except ValueError:
        return value


def _parse_parameters(text: str) -> dict[str, Any]:
    text = re.sub(r"^\+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\n\+", "\n ", text)
    return {
        match.group(1): _convert_value(match.group(2))
        for match in re.finditer(r"(\w+)\s*=\s*([^\s]+)", text)
    }


def parse_spice_models(path: Path) -> dict[str, dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    models: dict[str, dict[str, Any]] = {}
    for section in re.split(r"\.model\s+", content, flags=re.IGNORECASE)[1:]:
        lines = section.strip().split("\n", 1)
        if not lines or not lines[0].strip():
            continue
        parts = lines[0].split()
        if len(parts) < 2:
            continue
        name, model_type = parts[0], parts[1]
        param_text = " ".join(parts[2:])
        if len(lines) > 1:
            param_text += " " + lines[1]
        models[name] = {
            "name": name,
            "type": model_type,
            "parameters": _parse_parameters(_remove_comments(param_text)),
        }
    if not models:
        raise SpiceParseError(f"No .model statements found in {path}")
    return models


def write_spice_models(models: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for model in models.values():
            stream.write(f".model  {model['name']}  {model['type']}")
            for index, (name, value) in enumerate(model["parameters"].items()):
                if index % 4 == 0:
                    stream.write("\n+")
                if isinstance(value, float):
                    value_text = (
                        f"{value:.3e}"
                        if abs(value) < 1e-3 or abs(value) > 1e6
                        else str(value)
                    )
                else:
                    value_text = str(value)
                stream.write(f"{name:>12} = {value_text:<26}")
            stream.write("\n\n")


def _safe_token(value: str, max_len: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_") or "x"
    if cleaned[0].isdigit():
        cleaned = "n_" + cleaned
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[: max_len - 13]}_{digest}"


def _short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _parse_subckts(lines: list[str]) -> Subckt:
    root = Subckt(name="__root__", ports=[])
    stack = [root]
    for raw in lines:
        stripped = raw.strip()
        if stripped.lower().startswith(".subckt "):
            parts = stripped.split()
            if len(parts) < 2:
                raise SpiceParseError(f"Malformed .subckt line: {raw}")
            subckt = Subckt(name=parts[1], ports=parts[2:])
            parent = stack[-1]
            parent.children[subckt.name] = subckt
            parent.order.append(subckt.name)
            stack.append(subckt)
        elif stripped.lower().startswith(".ends"):
            if len(stack) == 1:
                raise SpiceParseError(f"Unexpected .ends line: {raw}")
            stack.pop()
        else:
            stack[-1].body.append(raw)
    if len(stack) != 1:
        raise SpiceParseError("Unclosed .subckt block")
    return root


def _format_subckt(subckt: Subckt) -> list[str]:
    if subckt.name == "__root__":
        split_at = len(subckt.body)
        for index, line in enumerate(subckt.body):
            stripped = line.lstrip()
            if stripped and stripped[0].upper() in {
                "X",
                "M",
                "R",
                "C",
                "V",
                "I",
                "B",
                "E",
                "G",
            }:
                split_at = index
                break
        output = list(subckt.body[:split_at])
        for name in subckt.order:
            output.extend(_format_subckt(subckt.children[name]))
        output.extend(subckt.body[split_at:])
        return output

    output = [".subckt " + " ".join([subckt.name, *subckt.ports])]
    for name in subckt.order:
        output.extend(_format_subckt(subckt.children[name]))
    output.extend(subckt.body)
    output.append(f".ends {subckt.name}")
    return output


class PerDeviceSpecializer:
    def __init__(
        self, root: Subckt, base_models: dict[str, dict[str, Any]], vth_std: float
    ):
        self.root = root
        self.base_models = base_models
        self.vth_std = vth_std
        self.model_clones: dict[str, dict[str, Any]] = {}
        self.audit: list[ModelClone] = []
        self._cache: dict[tuple[str, str], Subckt] = {}

    def _clone_model(
        self, base_model: str, hier_path: str, mos_name: str, subckt_name: str
    ) -> str:
        if base_model not in self.base_models:
            raise SpiceParseError(
                f"MOS model '{base_model}' was not found in the PDK model file"
            )
        new_name = _safe_token(
            f"MC_{base_model}_{_short_hash(f'{hier_path}/{mos_name}/{base_model}')}",
            max_len=64,
        )
        if new_name in self.model_clones:
            return new_name

        source = self.base_models[base_model]
        params = dict(source["parameters"])
        for param in MC_PARAMS:
            value = params.get(param)
            if isinstance(value, (int, float)):
                sigma = abs(float(value)) * self.vth_std
                params[param] = f"{{AGAUSS({value}, {sigma:.5g}, 1)}}"
        self.model_clones[new_name] = {
            "name": new_name,
            "type": source["type"],
            "parameters": params,
        }
        self.audit.append(
            ModelClone(new_name, base_model, hier_path, mos_name, subckt_name)
        )
        return new_name

    def _specialize_mos(self, line: str, hier_path: str, subckt_name: str) -> str:
        parts = line.split()
        if len(parts) < 6:
            raise SpiceParseError(f"Malformed MOS line: {line}")
        parts[5] = self._clone_model(parts[5], hier_path, parts[0], subckt_name)
        return " ".join(parts)

    def _lookup_subckt(self, scope: Subckt, name: str) -> Subckt | None:
        return scope.children.get(name) or self.root.children.get(name)

    def _specialize_instance(
        self, line: str, scope: Subckt, hier_path: str
    ) -> tuple[str, Subckt | None]:
        parts = line.split()
        if len(parts) < 2:
            raise SpiceParseError(f"Malformed subcircuit instance: {line}")
        target_index = len(parts) - 1
        for index, token in enumerate(parts[1:], start=1):
            if token.lower().startswith("params:") or "=" in token:
                target_index = index - 1
                break
        target = self._lookup_subckt(scope, parts[target_index])
        if target is None:
            return line, None
        specialized = self._specialize_subckt(target, f"{hier_path}/{parts[0]}")
        parts[target_index] = specialized.name
        return " ".join(parts), specialized

    def _specialize_subckt(self, subckt: Subckt, hier_path: str) -> Subckt:
        cache_key = (subckt.name, hier_path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        output = Subckt(
            name=_safe_token(f"{subckt.name}_MC_{_short_hash(hier_path)}", max_len=96),
            ports=list(subckt.ports),
        )
        self._cache[cache_key] = output

        for line in subckt.body:
            prefix = line.lstrip()[:1].upper()
            if prefix == "M":
                output.body.append(self._specialize_mos(line, hier_path, subckt.name))
            elif prefix == "X":
                new_line, child = self._specialize_instance(line, subckt, hier_path)
                if child is not None and child.name not in output.children:
                    output.children[child.name] = child
                    output.order.append(child.name)
                output.body.append(new_line)
            else:
                output.body.append(line)
        return output

    def specialize(self) -> Subckt:
        output = Subckt(name="__root__", ports=[])
        output.children = dict(self.root.children)
        output.order = list(self.root.order)
        for line in self.root.body:
            prefix = line.lstrip()[:1].upper()
            if prefix == "M":
                output.body.append(self._specialize_mos(line, "__top__", "__top__"))
            elif prefix == "X":
                new_line, child = self._specialize_instance(line, self.root, "__top__")
                if child is not None and child.name not in output.children:
                    output.children[child.name] = child
                    output.order.append(child.name)
                output.body.append(new_line)
            else:
                output.body.append(line)
        return output


def _resolve_include(path_text: str, deck_base_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    return path.resolve() if path.is_absolute() else (deck_base_dir / path).resolve()


def _parse_include(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.lower().startswith(".include"):
        return None
    parts = stripped.split(maxsplit=1)
    return parts[1].strip().strip("'\"") if len(parts) == 2 else None


def _patch_deck(
    lines: list[str],
    *,
    base_model_path: Path,
    model_output_path: Path,
    deck_base_dir: Path,
    mc_runs: int,
) -> list[str]:
    patched: list[str] = []
    include_replaced = False
    base_model_path = base_model_path.resolve()

    for line in lines:
        lower = line.strip().lower()
        if lower.startswith(".sampling") or lower.startswith(".options samples"):
            continue
        include_text = _parse_include(line)
        if (
            include_text
            and _resolve_include(include_text, deck_base_dir) == base_model_path
        ):
            patched.append(f'.include "{model_output_path.resolve()}"')
            include_replaced = True
            continue
        patched.append(line)
    if not include_replaced:
        raise SpiceParseError(
            f"PDK include was not found in the generated deck: {base_model_path}"
        )
    title_index = next(
        (index for index, line in enumerate(patched) if line.strip()), -1
    )
    insert_at = title_index + 1
    patched[insert_at:insert_at] = [
        ".SAMPLING useExpr=true",
        f".options samples numsamples={mc_runs}",
    ]
    return patched


def _write_audit(path: Path, rows: list[ModelClone]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model_name",
                "base_model",
                "hier_path",
                "mos_name",
                "subckt_name",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def specialize_netlist(
    deck_text: str,
    *,
    base_model_path: Path,
    model_output_path: Path,
    mc_runs: int,
    vth_std: float,
    deck_base_dir: Path | None = None,
    audit_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a per-device MC deck and write its independent model cards."""
    if mc_runs <= 0:
        raise ValueError("mc_runs must be positive")
    if vth_std < 0:
        raise ValueError("vth_std must be non-negative")

    base_model_path = base_model_path.expanduser().resolve()
    if not base_model_path.is_file():
        raise FileNotFoundError(f"PDK model file does not exist: {base_model_path}")
    deck_base_dir = (deck_base_dir or Path.cwd()).expanduser().resolve()
    model_output_path = model_output_path.expanduser().resolve()

    root = _parse_subckts(deck_text.splitlines())
    specializer = PerDeviceSpecializer(
        root, parse_spice_models(base_model_path), vth_std
    )
    output_lines = _format_subckt(specializer.specialize())
    output_lines = _patch_deck(
        output_lines,
        base_model_path=base_model_path,
        model_output_path=model_output_path,
        deck_base_dir=deck_base_dir,
        mc_runs=mc_runs,
    )
    write_spice_models(specializer.model_clones, model_output_path)
    if audit_path is not None:
        _write_audit(audit_path.expanduser().resolve(), specializer.audit)

    by_base_model: dict[str, int] = {}
    for item in specializer.audit:
        by_base_model[item.base_model] = by_base_model.get(item.base_model, 0) + 1
    summary = {
        "base_model_file": str(base_model_path),
        "mc_model_file": str(model_output_path),
        "mc_runs": mc_runs,
        "vth_std": vth_std,
        "unique_mc_models": len(specializer.audit),
        "by_base_model": by_base_model,
    }
    return "\n".join(output_lines) + "\n", summary

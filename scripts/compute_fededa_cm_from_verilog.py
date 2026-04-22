#!/usr/bin/env python3
"""Compute FedEDA circuit metadata (size, Rent p, sigma) from gate-level Verilog.

Pipeline (paper-aligned intent):
1) Parse gate-level Verilog and extract module hierarchy + cell-level connectivity.
2) Keep standard-cell instances and drop physical-only cells (filler/tap/endcap/decap).
3) For each submodule graph, run recursive 2-way min-cut partitioning (METIS style).
4) Collect (N, T) points at each partition level:
   - N: number of gates in a block
   - T: number of cross-boundary terminals (nets crossing block boundary)
5) Fit log(T) = log(k) + p_i * log(N) for each submodule.
6) Circuit-level metadata:
   - p = weighted mean of p_i (weighted by submodule occurrence count)
   - sigma = weighted std of p_i
   - size = number of cell instances (prefer external size CSV if provided)

Notes:
- This script uses `pyverilog` for parsing and `pymetis` for min-cut partitioning.
- For flattened netlists (single module), the top module is the only submodule source.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import pymetis
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "pymetis is required. Install it with: pip install pymetis"
    ) from exc

from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.parser import VerilogParser


POWER_NET_NAMES = {
    "vdd",
    "vss",
    "vpwr",
    "vgnd",
    "vpb",
    "vnb",
    "vccd1",
    "vssd1",
}

PHYS_ONLY_KEYWORDS = (
    "__fill_",
    "__filler",
    "__tap",
    "tapvpwrvgnd",
    "__endcap",
    "__decap_",
)

CONST_TOKEN_RE = re.compile(
    r"^(?:[0-9]+|(?:[0-9]+)?'[bdhoBDHO][0-9a-fA-F_xXzZ?]+)$"
)
HEADER_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\(")
PIN_EXPR_RE = re.compile(
    r"\.\s*[A-Za-z_][A-Za-z0-9_$]*\s*\((.*?)\)\s*(?=,|\)\s*;)",
    flags=re.S,
)
NET_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*(?:\[[^\[\]]+\])?")


@dataclass
class ModuleGraph:
    module_name: str
    num_cells: int
    node_to_nets: List[List[int]]
    net_to_nodes: List[List[int]]
    adjacency: List[List[int]]


def normalize_design_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    base = os.path.basename(str(name).strip())
    marker = ".graph.bin-"
    if marker in base:
        base = base.split(marker, 1)[0] + ".graph.bin"
    if base.endswith(".graph.bin"):
        base = base[: -len(".graph.bin")]
    elif base.endswith(".synthesis_preroute.v"):
        base = base[: -len(".synthesis_preroute.v")]
    elif base.endswith(".v"):
        base = base[:-2]
    return base.lower().strip() if base else None


def safe_to_code(node, codegen: ASTCodeGenerator) -> str:
    try:
        text = codegen.visit(node)
    except Exception:
        text = str(node)
    return " ".join(str(text).strip().split())


def looks_constant_token(token: str) -> bool:
    t = token.strip()
    if not t:
        return True
    return bool(CONST_TOKEN_RE.match(t))


def extract_signal_tokens(node, codegen: ASTCodeGenerator) -> Set[str]:
    if node is None:
        return set()

    cls = node.__class__.__name__
    if cls in {"IntConst", "FloatConst", "StringConst"}:
        return set()

    if cls == "Identifier":
        name = str(getattr(node, "name", "")).strip()
        return {name} if name else set()

    # Keep bit/slice expressions as distinct net tokens for better fidelity.
    if cls in {"Pointer", "Partselect"}:
        tok = safe_to_code(node, codegen)
        return set() if looks_constant_token(tok) else {tok}

    if cls in {"Lvalue", "Rvalue"}:
        return extract_signal_tokens(getattr(node, "var", None), codegen)

    if cls == "Concat":
        out: Set[str] = set()
        for c in getattr(node, "list", []) or []:
            out.update(extract_signal_tokens(c, codegen))
        return out

    if cls == "Repeat":
        return extract_signal_tokens(getattr(node, "value", None), codegen)

    # Generic fallback: recurse children, then fallback to rendered token.
    out: Set[str] = set()
    has_children = False
    try:
        children = node.children()
    except Exception:
        children = []
    for c in children:
        has_children = True
        out.update(extract_signal_tokens(c, codegen))

    if out:
        return out

    if not has_children:
        tok = safe_to_code(node, codegen)
        return set() if looks_constant_token(tok) else {tok}
    return set()


def is_physical_only_cell(module_type: str, inst_name: str, keep_phys_only: bool) -> bool:
    if keep_phys_only:
        return False
    key = f"{module_type} {inst_name}".lower()
    return any(k in key for k in PHYS_ONLY_KEYWORDS)


def load_size_lookup(size_csv: str, size_column: str) -> Dict[str, float]:
    if not size_csv or not os.path.isfile(size_csv):
        return {}
    lookup: Dict[str, float] = {}
    with open(size_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            design_raw = row.get("design") or row.get("circuit") or row.get("name")
            key = normalize_design_name(design_raw)
            if not key:
                continue
            raw = row.get(size_column)
            if raw is None:
                continue
            try:
                value = float(raw)
            except Exception:
                continue
            if value > 0:
                lookup[key] = value
    return lookup


def discover_verilog_files(netlists_root: str, pattern: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(netlists_root):
        return out

    for entry in sorted(os.listdir(netlists_root)):
        dpath = os.path.join(netlists_root, entry)
        if not os.path.isdir(dpath):
            continue
        cands = sorted(glob.glob(os.path.join(dpath, pattern)))
        if not cands:
            cands = sorted(glob.glob(os.path.join(dpath, "*.v")))
        if not cands:
            continue
        design = normalize_design_name(entry) or normalize_design_name(cands[0])
        if not design:
            continue
        out.append((design, cands[0]))
    return out


def extract_module_hierarchy(modules: Dict[str, object]) -> Tuple[Dict[str, Counter], List[str]]:
    user_modules = set(modules.keys())
    child_counts: Dict[str, Counter] = defaultdict(Counter)
    instantiated = set()

    for module_name, module_def in modules.items():
        for item in getattr(module_def, "items", []) or []:
            if item.__class__.__name__ != "InstanceList":
                continue
            module_type = str(getattr(item, "module", "")).strip()
            if module_type in user_modules:
                cnt = len(getattr(item, "instances", []) or [])
                if cnt > 0:
                    child_counts[module_name][module_type] += cnt
                    instantiated.add(module_type)

    tops = [m for m in modules.keys() if m not in instantiated]
    if not tops and modules:
        tops = [next(iter(modules.keys()))]
    return child_counts, sorted(tops)


def accumulate_occurrence_counts(
    top_modules: Sequence[str],
    child_counts: Dict[str, Counter],
) -> Dict[str, int]:
    occ: Dict[str, int] = defaultdict(int)
    stack: List[Tuple[str, int, Tuple[str, ...]]] = [(m, 1, tuple()) for m in top_modules]

    while stack:
        module_name, mult, path = stack.pop()
        if module_name in path:
            # Defensive break for cyclic hierarchies.
            continue
        occ[module_name] += int(mult)
        children = child_counts.get(module_name, Counter())
        if not children:
            continue
        next_path = path + (module_name,)
        for child_name, cnt in children.items():
            if cnt <= 0:
                continue
            stack.append((child_name, int(mult) * int(cnt), next_path))
    return dict(occ)


def _build_graph_from_node_nets(
    module_name: str,
    node_nets_raw: List[Set[str]],
    ignore_net_degree_above: int,
    max_clique_degree: int,
    max_cells_per_module: int,
) -> ModuleGraph:
    if max_cells_per_module > 0 and len(node_nets_raw) > max_cells_per_module:
        total = len(node_nets_raw)
        cap = max_cells_per_module
        # Evenly sample to cap runtime on very large flattened modules.
        picks = []
        last = -1
        for i in range(cap):
            idx = int((i * total) / cap)
            if idx <= last:
                idx = min(total - 1, last + 1)
            picks.append(idx)
            last = idx
        node_nets_raw = [node_nets_raw[i] for i in picks]

    num_cells = len(node_nets_raw)
    if num_cells == 0:
        return ModuleGraph(
            module_name=module_name,
            num_cells=0,
            node_to_nets=[],
            net_to_nodes=[],
            adjacency=[],
        )

    net_to_nodes_tmp: Dict[str, List[int]] = defaultdict(list)
    for node_idx, nets in enumerate(node_nets_raw):
        for net in nets:
            net_to_nodes_tmp[net].append(node_idx)

    net_to_nodes: List[List[int]] = []
    node_to_nets: List[List[int]] = [[] for _ in range(num_cells)]

    for _net_name, nodes in net_to_nodes_tmp.items():
        uniq_nodes = sorted(set(nodes))
        deg = len(uniq_nodes)
        if deg < 2:
            continue
        if ignore_net_degree_above > 0 and deg > ignore_net_degree_above:
            continue
        net_id = len(net_to_nodes)
        net_to_nodes.append(uniq_nodes)
        for nidx in uniq_nodes:
            node_to_nets[nidx].append(net_id)

    adjacency: List[Set[int]] = [set() for _ in range(num_cells)]
    for pins in net_to_nodes:
        if len(pins) < 2:
            continue
        if len(pins) <= max_clique_degree:
            for i in range(len(pins)):
                a = pins[i]
                for j in range(i + 1, len(pins)):
                    b = pins[j]
                    if a == b:
                        continue
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        else:
            hub = pins[0]
            for b in pins[1:]:
                if hub == b:
                    continue
                adjacency[hub].add(b)
                adjacency[b].add(hub)

    return ModuleGraph(
        module_name=module_name,
        num_cells=num_cells,
        node_to_nets=node_to_nets,
        net_to_nodes=net_to_nodes,
        adjacency=[sorted(list(nei)) for nei in adjacency],
    )


def _extract_flat_node_nets_fast(
    verilog_path: str,
    keep_phys_only: bool,
    ignore_power_nets: bool,
) -> Tuple[str, List[Set[str]]]:
    module_name = normalize_design_name(verilog_path) or "top"
    node_nets_raw: List[Set[str]] = []

    in_stmt = False
    stmt_parts: List[str] = []

    def _emit(stmt: str):
        nonlocal node_nets_raw
        m = HEADER_RE.match(stmt)
        if not m:
            return
        module_type = m.group(1)
        inst_name = m.group(2)
        if module_type == "module":
            return
        if is_physical_only_cell(module_type, inst_name, keep_phys_only):
            return

        nets: Set[str] = set()
        for arg_expr in PIN_EXPR_RE.findall(stmt):
            expr = str(arg_expr).strip()
            if not expr:
                continue
            for tok in NET_TOKEN_RE.findall(expr):
                t = tok.strip()
                if not t or looks_constant_token(t):
                    continue
                if ignore_power_nets and t.lower() in POWER_NET_NAMES:
                    continue
                nets.add(t)
        node_nets_raw.append(nets)

    with open(verilog_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("module "):
                mm = re.match(r"^module\s+([A-Za-z_][A-Za-z0-9_$]*)\b", line)
                if mm:
                    module_name = mm.group(1)
                continue
            if line.startswith("endmodule"):
                if in_stmt and stmt_parts:
                    _emit(" ".join(stmt_parts))
                break

            if not in_stmt:
                if not HEADER_RE.match(line):
                    continue
                if line.startswith("module "):
                    continue
                in_stmt = True
                stmt_parts = [line]
                if ");" in line:
                    _emit(" ".join(stmt_parts))
                    in_stmt = False
                    stmt_parts = []
            else:
                stmt_parts.append(line)
                if ");" in line:
                    _emit(" ".join(stmt_parts))
                    in_stmt = False
                    stmt_parts = []

    if in_stmt and stmt_parts:
        _emit(" ".join(stmt_parts))

    return module_name, node_nets_raw


def build_module_graph(
    module_def,
    user_modules: Set[str],
    codegen: ASTCodeGenerator,
    keep_phys_only: bool,
    ignore_power_nets: bool,
    ignore_net_degree_above: int,
    max_clique_degree: int,
    max_cells_per_module: int,
) -> ModuleGraph:
    node_nets_raw: List[Set[str]] = []

    for item in getattr(module_def, "items", []) or []:
        if item.__class__.__name__ != "InstanceList":
            continue

        module_type = str(getattr(item, "module", "")).strip()
        instances = getattr(item, "instances", []) or []

        # Skip hierarchical child-module instantiations; keep only leaf/library cells.
        if module_type in user_modules:
            continue

        for inst in instances:
            inst_name = str(getattr(inst, "name", "")).strip()
            if is_physical_only_cell(module_type, inst_name, keep_phys_only):
                continue

            nets: Set[str] = set()
            for port in getattr(inst, "portlist", []) or []:
                arg = getattr(port, "argname", None)
                tokens = extract_signal_tokens(arg, codegen)
                for tok in tokens:
                    t = tok.strip()
                    if not t or looks_constant_token(t):
                        continue
                    if ignore_power_nets and t.lower() in POWER_NET_NAMES:
                        continue
                    nets.add(t)
            node_nets_raw.append(nets)

    return _build_graph_from_node_nets(
        module_name=str(getattr(module_def, "name", "")),
        node_nets_raw=node_nets_raw,
        ignore_net_degree_above=ignore_net_degree_above,
        max_clique_degree=max_clique_degree,
        max_cells_per_module=max_cells_per_module,
    )


def induced_adjacency(nodes: List[int], adjacency: List[List[int]]) -> Tuple[List[List[int]], int]:
    local_id = {n: i for i, n in enumerate(nodes)}
    out: List[List[int]] = []
    edge_sum = 0
    for n in nodes:
        local_neighbors: List[int] = []
        for nb in adjacency[n]:
            j = local_id.get(nb)
            if j is not None:
                local_neighbors.append(j)
        if local_neighbors:
            local_neighbors = sorted(set(local_neighbors))
        out.append(local_neighbors)
        edge_sum += len(local_neighbors)
    return out, edge_sum // 2


def count_boundary_terminals(
    part_nodes: List[int],
    node_to_nets: List[List[int]],
    net_to_nodes: List[List[int]],
) -> int:
    inside_counts: Dict[int, int] = defaultdict(int)
    for n in part_nodes:
        for net_id in node_to_nets[n]:
            inside_counts[net_id] += 1

    boundary = 0
    for net_id, inside_cnt in inside_counts.items():
        if inside_cnt < len(net_to_nodes[net_id]):
            boundary += 1
    return boundary


def collect_rent_points(
    graph: ModuleGraph,
    min_block_size: int,
    min_partition_size: int,
    max_depth: int,
) -> List[Tuple[int, int]]:
    n = graph.num_cells
    if n <= 2:
        return []

    points: List[Tuple[int, int]] = []
    stack: List[Tuple[List[int], int]] = [(list(range(n)), 0)]

    while stack:
        nodes, depth = stack.pop()
        if depth >= max_depth:
            continue
        if len(nodes) < max(min_block_size, 2):
            continue

        local_adj, local_edges = induced_adjacency(nodes, graph.adjacency)
        if local_edges <= 0:
            continue

        try:
            _, part = pymetis.part_graph(2, adjacency=local_adj)
        except Exception:
            continue

        left = [nodes[i] for i, p in enumerate(part) if p == 0]
        right = [nodes[i] for i, p in enumerate(part) if p == 1]
        if len(left) == 0 or len(right) == 0:
            continue
        if min(len(left), len(right)) < min_partition_size:
            continue

        for side in (left, right):
            N = len(side)
            T = count_boundary_terminals(side, graph.node_to_nets, graph.net_to_nodes)
            if N > 1 and T > 0:
                points.append((N, T))

        stack.append((left, depth + 1))
        stack.append((right, depth + 1))

    return points


def fit_rent_slope(points: Sequence[Tuple[int, int]], clip_01: bool) -> Optional[float]:
    xs: List[float] = []
    ys: List[float] = []

    for n, t in points:
        if n > 1 and t > 0:
            xs.append(math.log(float(n)))
            ys.append(math.log(float(t)))

    if len(xs) < 2:
        return None

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    var_x = sum((x - x_mean) ** 2 for x in xs)
    if var_x <= 1e-12:
        return None

    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    if clip_01:
        slope = min(1.0, max(0.0, float(slope)))
    return float(slope)


def instance_mean_std(vals: Sequence[Tuple[float, int]]) -> Tuple[float, float]:
    """Paper-style mean/std over submodule instances.

    Equivalent to expanding each module-level p_i by its occurrence count and
    applying:
        mu = (1/N) * sum_i p_i
        sigma = sqrt((1/N) * sum_i (p_i - mu)^2)
    where N is total number of submodule instances.
    """
    total_n = sum(int(n) for _, n in vals)
    if total_n <= 0:
        return 0.0, 0.0

    mu = sum(float(v) * int(n) for v, n in vals) / float(total_n)
    var = sum(((float(v) - mu) ** 2) * int(n) for v, n in vals) / float(total_n)
    return float(mu), float(math.sqrt(max(0.0, var)))


def parse_verilog_modules(verilog_path: str, parser: VerilogParser):
    with open(verilog_path, "r", errors="ignore") as f:
        text = f.read()
    ast = parser.parse(text)

    modules = {}
    for d in getattr(ast.description, "definitions", []) or []:
        if d.__class__.__name__ == "ModuleDef":
            modules[str(d.name)] = d
    return modules


def compute_design_metadata(
    design_name: str,
    verilog_path: str,
    parser: VerilogParser,
    codegen: ASTCodeGenerator,
    size_lookup: Dict[str, float],
    args,
) -> Dict[str, object]:
    if bool(getattr(args, "fast_flat_parser", True)):
        try:
            top_module_name, node_nets_raw = _extract_flat_node_nets_fast(
                verilog_path=verilog_path,
                keep_phys_only=args.keep_phys_only,
                ignore_power_nets=args.ignore_power_nets,
            )
            graph = _build_graph_from_node_nets(
                module_name=top_module_name,
                node_nets_raw=node_nets_raw,
                ignore_net_degree_above=args.ignore_net_degree_above,
                max_clique_degree=args.max_clique_degree,
                max_cells_per_module=args.max_cells_per_module,
            )

            p_i = None
            points: List[Tuple[int, int]] = []
            if graph.num_cells >= args.min_block_size:
                points = collect_rent_points(
                    graph=graph,
                    min_block_size=args.min_block_size,
                    min_partition_size=args.min_partition_size,
                    max_depth=args.max_depth,
                )
                if len(points) >= args.min_points_for_fit:
                    p_i = fit_rent_slope(points, clip_01=not args.no_clip_p)

            size_value = size_lookup.get(design_name)
            size_source = "size_csv"
            if size_value is None:
                size_value = float(graph.num_cells)
                size_source = "hierarchy_count"

            if p_i is not None:
                p_value, sigma_value = instance_mean_std([(float(p_i), 1)])
            else:
                p_value, sigma_value = None, None
            return {
                "design": design_name,
                "verilog": verilog_path,
                "top_modules": [top_module_name],
                "size": float(size_value) if size_value is not None else None,
                "size_source": size_source,
                "p": p_value,
                "sigma": sigma_value,
                "num_submodules_total": 1 if graph.num_cells > 0 else 0,
                "num_submodules_used": 1 if p_i is not None else 0,
                "submodules": [
                    {
                        "module": top_module_name,
                        "occurrence": 1,
                        "num_cells": int(graph.num_cells),
                        "num_nets_used": int(len(graph.net_to_nodes)),
                        "num_points": int(len(points)),
                        "p_i": p_i,
                    }
                ],
            }
        except Exception:
            # Fallback to pyverilog AST parser below.
            if getattr(args, "verbose", False):
                import traceback

                traceback.print_exc()

    modules = parse_verilog_modules(verilog_path, parser)
    if not modules:
        return {
            "design": design_name,
            "verilog": verilog_path,
            "size": size_lookup.get(design_name),
            "p": None,
            "sigma": None,
            "num_submodules_total": 0,
            "num_submodules_used": 0,
            "submodules": [],
            "error": "No module found in Verilog AST.",
        }

    if bool(getattr(args, "strict_hierarchical", False)) and len(modules) < 2:
        return {
            "design": design_name,
            "verilog": verilog_path,
            "size": size_lookup.get(design_name),
            "p": None,
            "sigma": None,
            "num_submodules_total": 0,
            "num_submodules_used": 0,
            "submodules": [],
            "error": (
                "strict_hierarchical requested, but input netlist is flattened "
                "(<2 modules). Provide hierarchical gate-level Verilog."
            ),
        }

    child_counts, tops = extract_module_hierarchy(modules)

    # Prefer filename-matched top when available.
    guessed_top = normalize_design_name(design_name)
    if guessed_top and guessed_top in modules:
        top_modules = [guessed_top]
    else:
        top_modules = tops if tops else [next(iter(modules.keys()))]

    occ = accumulate_occurrence_counts(top_modules, child_counts)
    user_modules = set(modules.keys())

    module_graphs: Dict[str, ModuleGraph] = {}
    module_reports: Dict[str, Dict[str, object]] = {}

    for module_name, module_def in modules.items():
        graph = build_module_graph(
            module_def=module_def,
            user_modules=user_modules,
            codegen=codegen,
            keep_phys_only=args.keep_phys_only,
            ignore_power_nets=args.ignore_power_nets,
            ignore_net_degree_above=args.ignore_net_degree_above,
            max_clique_degree=args.max_clique_degree,
            max_cells_per_module=args.max_cells_per_module,
        )
        module_graphs[module_name] = graph

        p_i = None
        points: List[Tuple[int, int]] = []
        if graph.num_cells >= args.min_block_size:
            points = collect_rent_points(
                graph=graph,
                min_block_size=args.min_block_size,
                min_partition_size=args.min_partition_size,
                max_depth=args.max_depth,
            )
            if len(points) >= args.min_points_for_fit:
                p_i = fit_rent_slope(points, clip_01=not args.no_clip_p)

        module_reports[module_name] = {
            "module": module_name,
            "occurrence": int(occ.get(module_name, 0)),
            "num_cells": int(graph.num_cells),
            "num_nets_used": int(len(graph.net_to_nodes)),
            "num_points": int(len(points)),
            "p_i": p_i,
        }

    # Size fallback from hierarchical direct-cell counts.
    size_from_hierarchy = 0.0
    for module_name, report in module_reports.items():
        w = int(report["occurrence"])
        if w <= 0:
            continue
        size_from_hierarchy += float(report["num_cells"]) * float(w)

    size_value = size_lookup.get(design_name)
    size_source = "size_csv"
    if size_value is None:
        size_value = float(size_from_hierarchy)
        size_source = "hierarchy_count"

    instance_pis: List[Tuple[float, int]] = []
    used_modules = 0
    total_modules = 0

    for module_name in sorted(module_reports.keys()):
        rep = module_reports[module_name]
        occ_w = int(rep["occurrence"])
        num_cells = int(rep["num_cells"])
        if occ_w <= 0 or num_cells <= 0:
            continue
        total_modules += 1
        p_i = rep["p_i"]
        if p_i is None:
            continue
        used_modules += 1
        instance_pis.append((float(p_i), int(occ_w)))

    if instance_pis:
        p_value, sigma_value = instance_mean_std(instance_pis)
    else:
        p_value, sigma_value = None, None

    submods_sorted = sorted(
        module_reports.values(),
        key=lambda x: (
            -int(x.get("occurrence", 0)),
            -int(x.get("num_cells", 0)),
            str(x.get("module", "")),
        ),
    )

    return {
        "design": design_name,
        "verilog": verilog_path,
        "top_modules": top_modules,
        "size": float(size_value) if size_value is not None else None,
        "size_source": size_source,
        "p": p_value,
        "sigma": sigma_value,
        "num_submodules_total": int(total_modules),
        "num_submodules_used": int(used_modules),
        "submodules": submods_sorted,
    }


def parse_args(argv: Optional[Sequence[str]] = None):
    default_root = "/root/autodl-tmp/netlists"
    parser = argparse.ArgumentParser(
        description="Compute FedEDA size/p/sigma metadata from gate-level Verilog."
    )
    parser.add_argument("--netlists-root", type=str, default=default_root)
    parser.add_argument("--verilog-pattern", type=str, default="*.synthesis_preroute.v")
    parser.add_argument(
        "--designs",
        type=str,
        default="",
        help="Optional comma-separated design names to process.",
    )

    parser.add_argument(
        "--size-csv",
        type=str,
        default=os.path.join(default_root, "cell_size_from_netlists.csv"),
    )
    parser.add_argument(
        "--size-column",
        type=str,
        default="size_cells_from_verilog_sky130",
    )

    parser.add_argument(
        "--out-json",
        type=str,
        default=os.path.join(default_root, "fededa_cm_stats.json"),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=os.path.join(default_root, "fededa_cm_stats.csv"),
    )

    parser.add_argument("--keep-phys-only", action="store_true")
    parser.add_argument("--ignore-power-nets", action="store_true", default=True)
    parser.add_argument("--no-ignore-power-nets", action="store_false", dest="ignore_power_nets")

    parser.add_argument("--min-block-size", type=int, default=128)
    parser.add_argument("--min-partition-size", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-points-for-fit", type=int, default=6)
    parser.add_argument("--max-clique-degree", type=int, default=12)
    parser.add_argument(
        "--max-cells-per-module",
        type=int,
        default=0,
        help="If >0, evenly sample at most this many cells per module before min-cut.",
    )
    parser.add_argument(
        "--fast-flat-parser",
        action="store_true",
        default=True,
        help="Use a fast line parser for flattened gate-level Verilog (default: on).",
    )
    parser.add_argument(
        "--no-fast-flat-parser",
        action="store_false",
        dest="fast_flat_parser",
        help="Disable fast line parser and force pyverilog AST parsing.",
    )
    parser.add_argument(
        "--strict-hierarchical",
        action="store_true",
        help="Require non-flatten hierarchical netlists (>=2 modules per design).",
    )
    parser.add_argument(
        "--ignore-net-degree-above",
        type=int,
        default=1000,
        help="Skip very high-fanout nets to stabilize partition runtime.",
    )
    parser.add_argument("--no-clip-p", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing out-json if present; skip already computed designs.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save JSON/CSV every N completed designs (default: 1).",
    )

    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _dump_outputs(
    out_json: str,
    out_csv: str,
    payload_meta: Dict[str, object],
    results: Dict[str, Dict[str, object]],
):
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    payload = {
        "meta": payload_meta,
        "designs": results,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            fieldnames = [
                "design",
                "verilog",
                "size",
                "size_source",
                "p",
                "sigma",
                "num_submodules_used",
                "num_submodules_total",
                "error",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for design in sorted(results.keys()):
                rec = results[design]
                writer.writerow(
                    {
                        "design": design,
                        "verilog": rec.get("verilog"),
                        "size": rec.get("size"),
                        "size_source": rec.get("size_source"),
                        "p": rec.get("p"),
                        "sigma": rec.get("sigma"),
                        "num_submodules_used": rec.get("num_submodules_used"),
                        "num_submodules_total": rec.get("num_submodules_total"),
                        "error": rec.get("error"),
                    }
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args(argv)
    if bool(getattr(args, "strict_hierarchical", False)):
        # Strict paper mode must run module hierarchy parser.
        args.fast_flat_parser = False

    designs_filter: Optional[Set[str]] = None
    if args.designs.strip():
        designs_filter = {
            normalize_design_name(x)
            for x in args.designs.split(",")
            if normalize_design_name(x)
        }

    verilog_files = discover_verilog_files(args.netlists_root, args.verilog_pattern)
    if designs_filter is not None:
        verilog_files = [(d, p) for d, p in verilog_files if d in designs_filter]

    if not verilog_files:
        print(f"No Verilog files found under: {args.netlists_root}")
        return 1

    size_lookup = load_size_lookup(args.size_csv, args.size_column)

    parser = VerilogParser()
    codegen = ASTCodeGenerator()

    results: Dict[str, Dict[str, object]] = {}
    if args.resume and os.path.isfile(args.out_json):
        try:
            old = json.load(open(args.out_json, "r"))
            old_designs = old.get("designs") if isinstance(old, dict) else None
            if isinstance(old_designs, dict):
                for k, v in old_designs.items():
                    nk = normalize_design_name(k)
                    if nk and isinstance(v, dict):
                        results[nk] = v
                print(f"Resume: loaded {len(results)} design(s) from existing JSON.")
        except Exception as exc:
            print(f"Resume warning: failed to read existing JSON: {exc}")

    pending = [(d, p) for d, p in verilog_files if d not in results]
    if not pending:
        payload_meta = {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "netlists_root": os.path.abspath(args.netlists_root),
            "size_csv": os.path.abspath(args.size_csv) if args.size_csv else None,
            "size_column": args.size_column,
            "method": "gate-level verilog -> module graph -> recursive min-cut -> rent fit",
            "params": {
                "keep_phys_only": bool(args.keep_phys_only),
                "ignore_power_nets": bool(args.ignore_power_nets),
                "min_block_size": int(args.min_block_size),
                "min_partition_size": int(args.min_partition_size),
                "max_depth": int(args.max_depth),
                "min_points_for_fit": int(args.min_points_for_fit),
                "max_clique_degree": int(args.max_clique_degree),
                "max_cells_per_module": int(args.max_cells_per_module),
                "fast_flat_parser": bool(args.fast_flat_parser),
                "ignore_net_degree_above": int(args.ignore_net_degree_above),
                "clip_p_to_01": not bool(args.no_clip_p),
                "resume": bool(args.resume),
                "save_every": int(max(1, args.save_every)),
            },
            "num_designs": len(results),
            "elapsed_sec": 0.0,
        }
        _dump_outputs(args.out_json, args.out_csv, payload_meta, results)
        print("All designs already computed; outputs refreshed.")
        print(f"Saved JSON: {args.out_json}")
        if args.out_csv:
            print(f"Saved CSV:  {args.out_csv}")
        return 0

    t_start = time.time()
    save_every = max(1, int(args.save_every))
    done_in_this_run = 0

    print(f"Processing {len(pending)} pending design(s) (total target={len(verilog_files)})...")
    for idx, (design_name, verilog_path) in enumerate(pending, start=1):
        dt0 = time.time()
        try:
            meta = compute_design_metadata(
                design_name=design_name,
                verilog_path=verilog_path,
                parser=parser,
                codegen=codegen,
                size_lookup=size_lookup,
                args=args,
            )
            results[design_name] = meta
            msg = (
                f"[{idx}/{len(pending)}] {design_name}: "
                f"size={meta.get('size')}, p={meta.get('p')}, sigma={meta.get('sigma')}, "
                f"submods={meta.get('num_submodules_used')}/{meta.get('num_submodules_total')}"
            )
            print(msg)
        except Exception as exc:
            results[design_name] = {
                "design": design_name,
                "verilog": verilog_path,
                "size": size_lookup.get(design_name),
                "p": None,
                "sigma": None,
                "num_submodules_total": 0,
                "num_submodules_used": 0,
                "submodules": [],
                "error": str(exc),
            }
            print(f"[{idx}/{len(pending)}] {design_name}: ERROR: {exc}")
            if args.verbose:
                import traceback

                traceback.print_exc()
        finally:
            print(f"  elapsed: {time.time() - dt0:.2f}s")
            done_in_this_run += 1

        if done_in_this_run % save_every == 0:
            payload_meta = {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "netlists_root": os.path.abspath(args.netlists_root),
                "size_csv": os.path.abspath(args.size_csv) if args.size_csv else None,
                "size_column": args.size_column,
                "method": "gate-level verilog -> module graph -> recursive min-cut -> rent fit",
                "params": {
                    "keep_phys_only": bool(args.keep_phys_only),
                    "ignore_power_nets": bool(args.ignore_power_nets),
                    "min_block_size": int(args.min_block_size),
                    "min_partition_size": int(args.min_partition_size),
                    "max_depth": int(args.max_depth),
                    "min_points_for_fit": int(args.min_points_for_fit),
                    "max_clique_degree": int(args.max_clique_degree),
                    "max_cells_per_module": int(args.max_cells_per_module),
                    "fast_flat_parser": bool(args.fast_flat_parser),
                    "ignore_net_degree_above": int(args.ignore_net_degree_above),
                    "clip_p_to_01": not bool(args.no_clip_p),
                    "resume": bool(args.resume),
                    "save_every": int(save_every),
                },
                "num_designs": len(results),
                "elapsed_sec": round(time.time() - t_start, 3),
            }
            _dump_outputs(args.out_json, args.out_csv, payload_meta, results)
            print(
                f"  checkpoint saved ({len(results)}/{len(verilog_files)} total designs) -> {args.out_json}"
            )

    payload_meta = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "netlists_root": os.path.abspath(args.netlists_root),
        "size_csv": os.path.abspath(args.size_csv) if args.size_csv else None,
        "size_column": args.size_column,
        "method": "gate-level verilog -> module graph -> recursive min-cut -> rent fit",
        "params": {
            "keep_phys_only": bool(args.keep_phys_only),
            "ignore_power_nets": bool(args.ignore_power_nets),
            "min_block_size": int(args.min_block_size),
            "min_partition_size": int(args.min_partition_size),
            "max_depth": int(args.max_depth),
            "min_points_for_fit": int(args.min_points_for_fit),
            "max_clique_degree": int(args.max_clique_degree),
            "max_cells_per_module": int(args.max_cells_per_module),
            "fast_flat_parser": bool(args.fast_flat_parser),
            "ignore_net_degree_above": int(args.ignore_net_degree_above),
            "clip_p_to_01": not bool(args.no_clip_p),
            "resume": bool(args.resume),
            "save_every": int(save_every),
        },
        "num_designs": len(results),
        "elapsed_sec": round(time.time() - t_start, 3),
    }
    _dump_outputs(args.out_json, args.out_csv, payload_meta, results)

    print(f"Saved JSON: {args.out_json}")
    if args.out_csv:
        print(f"Saved CSV:  {args.out_csv}")
    print(f"Done in {time.time() - t_start:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

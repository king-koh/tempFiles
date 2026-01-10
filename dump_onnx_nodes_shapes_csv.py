#!/usr/bin/env python3
# dump_onnx_nodes_shapes_csv.py
#
# Usage:
#   python3 dump_onnx_nodes_shapes_csv.py model.onnx out.csv
#
# Output:
#   - CSV: per-node shape-inference summary + flags (rank>=5, missing, 1D conv, var-var matmul)
#   - stdout: counts of rank>=5 tensors, missing shapes, 1D conv nodes, var-var matmul nodes

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import onnx
from onnx import TensorProto, shape_inference
from onnx.numpy_helper import to_array

# ---------- helpers: dtype / shape formatting ----------
def parse_constant_node_value(node: onnx.NodeProto) -> Optional[TypeShape]:
    """Constant node の attribute 'value' から dtype/shape を推定"""
    if node.op_type != "Constant":
        return None
    for a in node.attribute:
        if a.name == "value" and a.type == onnx.AttributeProto.TENSOR:
            arr = to_array(a.t)
            shape = [int(x) for x in arr.shape]
            # onnx.TensorProto の dtype(int) に戻すのは少し面倒なので、ここは a.t.data_type を使う
            return TypeShape(dtype=int(a.t.data_type), shape=shape)
    return None

def collect_initializer_names(m: onnx.ModelProto) -> Set[str]:
    return {init.name for init in m.graph.initializer if init.name}

def collect_constant_output_names(m: onnx.ModelProto) -> Set[str]:
    names = set()
    for n in m.graph.node:
        if n.op_type == "Constant":
            for o in n.output:
                if o:
                    names.add(o)
    return names

_DTYPE_MAP = {v: k for k, v in TensorProto.DataType.items()}  # int -> name

def dtype_to_str(dtype: Optional[int]) -> str:
    if dtype is None:
        return ""
    return _DTYPE_MAP.get(dtype, str(dtype))


@dataclass
class TypeShape:
    dtype: Optional[int]
    shape: Optional[List[Optional[int]]]  # None = unknown dim/rank

    @property
    def rank(self) -> Optional[int]:
        if self.shape is None:
            return None
        return len(self.shape)

    def fmt(self) -> str:
        """Return like: FLOAT[1,512,180,180] or FLOAT[?,?,?] or [] if totally unknown"""
        dt = dtype_to_str(self.dtype)
        if self.shape is None:
            # rank unknown
            return f"{dt}[]".strip()
        dims = []
        for d in self.shape:
            dims.append("?" if d is None else str(d))
        return f"{dt}[{','.join(dims)}]".strip()


def parse_value_info(vi: onnx.ValueInfoProto) -> TypeShape:
    t = vi.type.tensor_type
    dtype = t.elem_type if t.HasField("elem_type") else None
    if not t.HasField("shape"):
        return TypeShape(dtype=dtype, shape=None)

    dims: List[Optional[int]] = []
    for dim in t.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(None)
        else:
            dims.append(None)
    return TypeShape(dtype=dtype, shape=dims)


def build_valueinfo_map(m: onnx.ModelProto) -> Dict[str, TypeShape]:
    """Collect shapes for graph inputs/outputs/value_info + initializer + Constant outputs."""
    g = m.graph
    mp: Dict[str, TypeShape] = {}

    # 1) ValueInfo（shape_inference の成果）
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.name:
            mp[vi.name] = parse_value_info(vi)

    # 2) Initializer（重み/バイアスなど）
    for init in g.initializer:
        if not init.name:
            continue
        # init.dims は shape、init.data_type は dtype
        mp[init.name] = TypeShape(dtype=int(init.data_type),
                                 shape=[int(d) for d in init.dims])

    # 3) Constant node outputs（value attribute から）
    for n in g.node:
        if n.op_type != "Constant":
            continue
        ts = parse_constant_node_value(n)
        if ts is None:
            continue
        for o in n.output:
            if o and (o not in mp):
                mp[o] = ts

    return mp


def collect_constant_value_names(m: onnx.ModelProto) -> Set[str]:
    """Names that are compile-time constants: graph.initializer + Constant node outputs."""
    g = m.graph
    const_names: Set[str] = set()
    for init in g.initializer:
        if init.name:
            const_names.add(init.name)
    for n in g.node:
        if n.op_type == "Constant":
            for o in n.output:
                if o:
                    const_names.add(o)
    return const_names


# ---------- checks ----------

def has_rank_ge_5(ts: Optional[TypeShape]) -> bool:
    return (ts is not None) and (ts.rank is not None) and (ts.rank >= 5)


def is_1d_conv(node: onnx.NodeProto, vi_map: Dict[str, TypeShape]) -> bool:
    if node.op_type != "Conv":
        return False

    x_name = node.input[0] if len(node.input) > 0 else ""
    w_name = node.input[1] if len(node.input) > 1 else ""

    x_ts = vi_map.get(x_name)
    w_ts = vi_map.get(w_name)

    # 1D conv in ONNX: X rank=3 (N,C,L), W rank=3 (M,C,k)
    x_rank = x_ts.rank if x_ts else None
    w_rank = w_ts.rank if w_ts else None

    return (x_rank == 3) or (w_rank == 3)


def is_var_var_matmul(node: onnx.NodeProto, const_names: Set[str]) -> bool:
    if node.op_type != "MatMul":
        return False
    if len(node.input) < 2:
        return False

    a, b = node.input[0], node.input[1]
    # “var-var” = neither input is a known compile-time constant
    return (a not in const_names) and (b not in const_names)


def is_missing(ts: Optional[TypeShape]) -> bool:
    """True if we have no shape inference info for this value (or unknown rank)."""
    return (ts is None) or (ts.shape is None)


# ---------- main dump ----------

def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python3 dump_onnx_nodes_shapes_csv.py model.onnx out.csv", file=sys.stderr)
        return 2

    in_path = sys.argv[1]
    out_csv = sys.argv[2]

    model = onnx.load(in_path)

    # Shape inference
    try:
        model_si = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[WARN] shape_inference failed: {e}", file=sys.stderr)
        model_si = model

    vi_map = build_valueinfo_map(model_si)
    const_names = collect_constant_value_names(model_si)
    g = model_si.graph

    # Count tensors (value names) whose inferred rank >= 5
    rank5_value_names: Set[str] = {name for name, ts in vi_map.items() if has_rank_ge_5(ts)}

    # Summary counters
    varvar_count = 0
    conv1d_count = 0
    nodes_with_5d = 0

    nodes_with_missing = 0
    missing_value_names: Set[str] = set()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "node_idx",
            "node_name",
            "op_type",
            "inputs",
            "input_typeshapes",
            "outputs",
            "output_typeshapes",
            "missing_in",
            "missing_out",
            "has_missing_in",
            "has_missing_out",
            "has_5d_input",
            "has_5d_output",
            "is_1d_conv",
            "is_var_var_matmul",
            "max_output_rank",
        ])

        for i, node in enumerate(g.node):
            in_names = [x for x in node.input if x]
            out_names = [x for x in node.output if x]

            in_ts = [vi_map.get(n) for n in in_names]
            out_ts = [vi_map.get(n) for n in out_names]

            # missing counts
            miss_in = sum(1 for ts in in_ts if is_missing(ts))
            miss_out = sum(1 for ts in out_ts if is_missing(ts))
            has_miss_in = int(miss_in > 0)
            has_miss_out = int(miss_out > 0)

            if has_miss_in or has_miss_out:
                nodes_with_missing += 1
                for n, ts in zip(in_names, in_ts):
                    if is_missing(ts):
                        missing_value_names.add(n)
                for n, ts in zip(out_names, out_ts):
                    if is_missing(ts):
                        missing_value_names.add(n)

            has_5d_in = int(any(has_rank_ge_5(ts) for ts in in_ts))
            has_5d_out = int(any(has_rank_ge_5(ts) for ts in out_ts))
            if has_5d_in or has_5d_out:
                nodes_with_5d += 1

            is1d = int(is_1d_conv(node, vi_map))
            isvv = int(is_var_var_matmul(node, const_names))
            if is1d:
                conv1d_count += 1
            if isvv:
                varvar_count += 1

            # max output rank
            max_out_rank: Optional[int] = None
            for ts in out_ts:
                if ts is None or ts.rank is None:
                    continue
                if max_out_rank is None or ts.rank > max_out_rank:
                    max_out_rank = ts.rank

            w.writerow([
                i,
                node.name,
                node.op_type,
                ";".join(in_names),
                ";".join(ts.fmt() if ts else "[]" for ts in in_ts),
                ";".join(out_names),
                ";".join(ts.fmt() if ts else "[]" for ts in out_ts),
                miss_in,
                miss_out,
                has_miss_in,
                has_miss_out,
                has_5d_in,
                has_5d_out,
                is1d,
                isvv,
                "" if max_out_rank is None else max_out_rank,
            ])
    

    initializer_names = collect_initializer_names(model_si)
    constant_output_names = collect_constant_output_names(model_si)

    initializer_missing = sorted(n for n in missing_value_names if n in initializer_names)
    constant_missing = sorted(n for n in missing_value_names if n in constant_output_names)
    true_missing = sorted(
        n for n in missing_value_names
        if n not in initializer_names and n not in constant_output_names
    )

    print(f"[OK] wrote CSV: {out_csv}")
    print(f"[INFO] nodes: {len(g.node)}")
    print(f"[INFO] inferred value tensors (inputs/outputs/value_info): {len(vi_map)}")
    print(f"[INFO] rank>=5 tensors: {len(rank5_value_names)}")
    print(f"[INFO] nodes touching rank>=5 tensors: {nodes_with_5d}")
    print(f"[INFO] 1D Conv nodes: {conv1d_count}")
    print(f"[INFO] var-var MatMul nodes: {varvar_count}")
    print(f"[INFO] nodes with missing shape info: {nodes_with_missing}")
    print(f"[INFO] unique value names with missing shape info: {len(missing_value_names)}")

    print(f"[INFO] missing breakdown:")
    print(f"  initializer_missing: {len(initializer_missing)}")
    print(f"  constant_missing   : {len(constant_missing)}")
    print(f"  true_missing       : {len(true_missing)}")

    # 詳細を見たいとき用（多ければコメントアウト）
    if initializer_missing:
        print("  [initializer_missing]")
        for n in initializer_missing:
            print("    ", n)

    if constant_missing:
        print("  [constant_missing]")
        for n in constant_missing:
            print("    ", n)

    if true_missing:
        print("  [true_missing]  <-- CGC要注意")
        for n in true_missing:
            print("    ", n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

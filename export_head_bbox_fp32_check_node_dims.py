#!/usr/bin/env python3
import onnx
from onnx import numpy_helper

# onnx_path = onnx_path = "/Lidar_AI_Solution/CUDA-BEVFusion/qat/byUsingConfigs/head.bbox.fp32.onnx"
onnx_path = onnx_path = "/Lidar_AI_Solution/CUDA-BEVFusion/qat/onnx_fp16/head.bbox.onnx"

# -----------------------------
# ANSI colors (terminal)
# -----------------------------
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
BOLD = "\033[1m"
RESET = "\033[0m"


def tensor_shape_list_from_type(tensor_type):
    """TypeProto.Tensor shape を Python list へ。未知は None / str を混ぜる。"""
    if not tensor_type.HasField("shape"):
        return None

    dims = []
    for d in tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(str(d.dim_param))
        else:
            dims.append(None)
    return dims


def tensor_shape_str_from_list(dims):
    if dims is None:
        return "unknown"
    def one(x):
        if x is None:
            return "?"
        return str(x)
    return "[" + ", ".join(one(x) for x in dims) + "]"


def rank_of_dims(dims):
    """dims が分かれば rank を返す。shape不明なら None。"""
    if dims is None:
        return None
    return len(dims)


def has_unknown_dim(dims):
    if dims is None:
        return True
    return any((d is None) or isinstance(d, str) for d in dims)


def colorize_shape(name, dims, dtype_text, prefix=""):
    """
    rank>=5 を赤で強調。
    rank==4 を黄、未知含むshapeを青（参考）。
    """
    r = rank_of_dims(dims)
    shape_s = tensor_shape_str_from_list(dims)

    tag = ""
    colored = f"{prefix}{name:30s} shape={shape_s:25s} dtype={dtype_text}"

    if r is not None and r >= 5:
        tag = f"{BOLD}{RED}[RANK>={5}]{RESET} "
        colored = f"{tag}{BOLD}{RED}{colored}{RESET}"
    elif r == 4:
        tag = f"{YELLOW}[RANK=4]{RESET} "
        colored = f"{tag}{YELLOW}{colored}{RESET}"
    elif has_unknown_dim(dims):
        tag = f"{BLUE}[UNKNOWN_SHAPE]{RESET} "
        colored = f"{tag}{BLUE}{colored}{RESET}"

    return colored


def main():
    model = onnx.load(onnx_path)
    graph = model.graph

    rank5_or_more = []  # (section, name, dims, dtype)

    print("=" * 100)
    print("ONNX MODEL:", onnx_path)
    print("=" * 100)

    # ============================================================
    # Inputs
    # ============================================================
    print("\n[GRAPH INPUTS]")
    for i, inp in enumerate(graph.input):
        dims = tensor_shape_list_from_type(inp.type.tensor_type)
        dtype = inp.type.tensor_type.elem_type
        line = colorize_shape(inp.name, dims, str(dtype), prefix=f"{i:3d}: name=")
        print(line)
        if rank_of_dims(dims) is not None and rank_of_dims(dims) >= 5:
            rank5_or_more.append(("GRAPH_INPUT", inp.name, dims, dtype))

    # ============================================================
    # Outputs
    # ============================================================
    print("\n[GRAPH OUTPUTS]")
    for i, out in enumerate(graph.output):
        dims = tensor_shape_list_from_type(out.type.tensor_type)
        dtype = out.type.tensor_type.elem_type
        line = colorize_shape(out.name, dims, str(dtype), prefix=f"{i:3d}: name=")
        print(line)
        if rank_of_dims(dims) is not None and rank_of_dims(dims) >= 5:
            rank5_or_more.append(("GRAPH_OUTPUT", out.name, dims, dtype))

    # ============================================================
    # Initializers (weights / constants) - raw_dataは出さない
    # ============================================================
    print("\n[INITIALIZERS / WEIGHTS]")
    for i, init in enumerate(graph.initializer):
        # raw_dataを展開しない（to_arrayしない）
        dims = list(init.dims)  # TensorProtoのdims
        dtype = init.data_type
        line = colorize_shape(init.name, dims, str(dtype), prefix=f"{i:3d}: name=")
        print(line)
        if rank_of_dims(dims) is not None and rank_of_dims(dims) >= 5:
            rank5_or_more.append(("INITIALIZER", init.name, dims, dtype))

    # ============================================================
    # ValueInfo (intermediate tensors with shape info)
    # ============================================================
    print("\n[VALUE_INFO / INTERMEDIATE TENSORS]")
    if len(graph.value_info) == 0:
        print(" (none)")
    for i, v in enumerate(graph.value_info):
        dims = tensor_shape_list_from_type(v.type.tensor_type)
        dtype = v.type.tensor_type.elem_type
        line = colorize_shape(v.name, dims, str(dtype), prefix=f"{i:3d}: name=")
        print(line)
        if rank_of_dims(dims) is not None and rank_of_dims(dims) >= 5:
            rank5_or_more.append(("VALUE_INFO", v.name, dims, dtype))

    # ============================================================
    # Nodes
    # ============================================================
    print("\n[NODES]")
    for i, node in enumerate(graph.node):
        print("-" * 80)
        print(f"Node {i:4d}")
        print(f"  op_type : {node.op_type}")
        print(f"  name    : {node.name if node.name else '(none)'}")

        print("  inputs :")
        for j, inp in enumerate(node.input):
            print(f"    {j:2d}: {inp}")

        print("  outputs:")
        for j, out in enumerate(node.output):
            print(f"    {j:2d}: {out}")

        # raw_dataが出やすいので attributes の中身は隠して、名前とtypeだけ出す
        if node.attribute:
            print("  attributes:")
            for attr in node.attribute:
                print(f"    - {attr.name} (type={attr.type})")

    # ============================================================
    # Summary & rank>=5 list
    # ============================================================
    print("\n[SUMMARY]")
    print(f" #inputs      : {len(graph.input)}")
    print(f" #outputs     : {len(graph.output)}")
    print(f" #initializers: {len(graph.initializer)}")
    print(f" #value_info  : {len(graph.value_info)}")
    print(f" #nodes       : {len(graph.node)}")

    print("\n" + "=" * 100)
    print("[TENSORS WITH RANK >= 5]")
    print("=" * 100)
    if not rank5_or_more:
        print(" (none)")
    else:
        for sec, name, dims, dtype in rank5_or_more:
            shape_s = tensor_shape_str_from_list(dims)
            print(f"{BOLD}{RED}{sec:12s}{RESET}  {name:40s}  shape={shape_s}  dtype={dtype}")

    print("\n[OK] Listed ONNX graph and highlighted rank>=5 tensors.")


if __name__ == "__main__":
    main()

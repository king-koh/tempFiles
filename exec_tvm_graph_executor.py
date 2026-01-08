"""
import onnx
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor

onnx_path = "/Lidar_AI_Solution/CUDA-BEVFusion/qat/onnx_fp16/head.bbox.sim.onnx"  # ← FP32/FP16版を用意
model = onnx.load(onnx_path)

shape_dict = {"middle": (1, 512, 180, 180)}
mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

# IRを眺める（TVMが何に変換したか）
print(mod.astext(show_meta_data=False))

target = "llvm"  # CPU
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.cpu()
m = graph_executor.GraphModule(lib["default"](dev))

x = np.random.randn(*shape_dict["middle"]).astype("float32")
m.set_input("middle", x)
m.run()

# 出力が複数なら index を変えて取得
out0 = m.get_output(0).numpy()
print(out0.shape, out0.dtype)
"""

import onnx
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from collections import Counter

onnx_path = "/Lidar_AI_Solution/CUDA-BEVFusion/qat/onnx_fp16/head.bbox.sim.onnx"
model = onnx.load(onnx_path)

shape_dict = {"middle": (1, 512, 180, 180)}
mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

# -----------------------------
# 追加: cgc 関数の一覧/プロパティ表示
# -----------------------------
def op_hist(func: relay.Function) -> Counter:
    """関数内に含まれる Relay Op の出現回数を数える"""
    cnt = Counter()

    def visit(e):
        if isinstance(e, relay.Call):
            op = e.op
            if isinstance(op, tvm.ir.Op):
                cnt[op.name] += 1

    relay.analysis.post_order_visit(func.body, visit)
    return cnt

def summarize_func(name: str, func: relay.Function, topn_ops: int = 25):
    print("=" * 80)
    print(f"Function: @{name}")
    # type
    try:
        print("  Type :", func.checked_type)
    except Exception:
        print("  Type :", "(checked_type unavailable; try InferType pass)")

    # attrs
    attrs = {}
    if func.attrs is not None:
        for k in func.attrs.keys():
            try:
                attrs[k] = func.attrs[k]
            except Exception:
                attrs[k] = "<unprintable>"
    print("  Attrs:", attrs if attrs else "{}")

    # params
    try:
        for i, p in enumerate(func.params):
            print(f"  Param[{i}] {p.name_hint}: {p.checked_type}")
    except Exception:
        pass

    # op histogram
    cnt = op_hist(func)
    total_ops = sum(cnt.values())
    print(f"  Ops(total calls): {total_ops}")
    for op, n in cnt.most_common(topn_ops):
        print(f"    {op}: {n}")

def list_cgc_funcs(mod_p: tvm.IRModule, compiler_name="cgc"):
    """Partition後IRModuleから Compiler=compiler_name の関数だけ拾う"""
    cgc = []
    for gv, f in mod_p.functions.items():
        if not isinstance(f, relay.Function):
            continue
        name = getattr(gv, "name_hint", None)
        if not name:
            continue
        # PartitionGraph 後は func.attrs["Compiler"] が入るのが典型
        comp = None
        if f.attrs is not None and "Compiler" in f.attrs.keys():
            comp = str(f.attrs["Compiler"])
        if comp == compiler_name:
            cgc.append((name, f))
    return sorted(cgc, key=lambda x: x[0])

def annotate_and_partition(mod_in: tvm.IRModule, compiler_name="cgc") -> tvm.IRModule:
    """BYOC向けの annotate -> merge -> partition を実施"""
    seq = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.AnnotateTarget(compiler_name),
        relay.transform.MergeCompilerRegions(),
        relay.transform.PartitionGraph(),
        relay.transform.InferType(),
    ])
    with tvm.transform.PassContext(opt_level=3):
        return seq(mod_in)

print("=== Original Relay IR (head) ===")
print(mod.astext(show_meta_data=False)[:2000])  # 長いので先頭だけ（必要なら削って全文表示）

# Partitionして cgc関数を列挙
compiler_name = "cgc"  # 環境により "quadric" 等の場合もあります
try:
    mod_p = annotate_and_partition(mod, compiler_name=compiler_name)
    print("\n=== Partitioned Relay IR (head) ===")
    txt = mod_p.astext(show_meta_data=False)
    print(txt[:2000])  # 先頭だけ（必要なら全文に）
    cgc_funcs = list_cgc_funcs(mod_p, compiler_name=compiler_name)

    print("\n=== CGC function list ===")
    if not cgc_funcs:
        print(f"(none) Compiler='{compiler_name}' functions not found.")
        print("  - 可能性: compiler_name が違う / cgc統合が無い / annotateが効いていない")
        print("  - ヒント: mod_p.astext() で 'Compiler' を検索してください。")
    else:
        for name, f in cgc_funcs:
            summarize_func(name, f, topn_ops=30)

except Exception as e:
    print("\n[WARN] annotate/partition failed:", repr(e))
    print("  このTVM環境では AnnotateTarget/PartitionGraph 周りが使えない or 名前が違う可能性があります。")

# -----------------------------
# 元の: TVM(LLVM)でビルドして実行
# -----------------------------
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.cpu()
m = graph_executor.GraphModule(lib["default"](dev))

x = np.random.randn(*shape_dict["middle"]).astype("float32")
m.set_input("middle", x)
m.run()

out0 = m.get_output(0).numpy()
print("\n=== Run result ===")
print(out0.shape, out0.dtype)

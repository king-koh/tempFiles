#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np

onnx_path = "/Lidar_AI_Solution/CUDA-BEVFusion/qat/onnx_fp32/head.tr.fp32.onnx"

def main():
    # -----------------------------
    # Session 作成（CPUで十分）
    # -----------------------------
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )

    print("=" * 80)
    print("ONNX MODEL:", onnx_path)
    print("=" * 80)

    # -----------------------------
    # 入力情報
    # -----------------------------
    print("\n[Inputs]")
    inputs = sess.get_inputs()
    for i, inp in enumerate(inputs):
        print(f" {i}: name={inp.name}, shape={inp.shape}, dtype={inp.type}")

    # -----------------------------
    # 出力情報
    # -----------------------------
    print("\n[Outputs]")
    outputs = sess.get_outputs()
    for i, out in enumerate(outputs):
        print(f" {i}: name={out.name}, shape={out.shape}, dtype={out.type}")

    # -----------------------------
    # ダミー入力生成
    # -----------------------------
    # head.bbox の想定入力: middle = (1, 512, 180, 180)
    # shape は ONNX から取得したものを優先
    input_name = inputs[0].name
    input_shape = inputs[0].shape

    # None を 1 に置き換え（動的軸対策）
    concrete_shape = [
        1 if (s is None or isinstance(s, str)) else int(s)
        for s in input_shape
    ]

    print("\n[Dummy Input]")
    print(" name :", input_name)
    print(" shape:", concrete_shape)

    dummy_input = np.random.randn(*concrete_shape).astype(np.float32)

    # -----------------------------
    # 1回だけ推論
    # -----------------------------
    print("\n[Run inference once]")
    ort_outputs = sess.run(
        None,
        {input_name: dummy_input}
    )

    # -----------------------------
    # 出力 shape 確認
    # -----------------------------
    print("\n[Output shapes]")
    for out_meta, out_value in zip(outputs, ort_outputs):
        print(f" {out_meta.name}: shape={out_value.shape}, dtype={out_value.dtype}")

    print("\n[OK] onnxruntime inference succeeded.")

if __name__ == "__main__":
    main()

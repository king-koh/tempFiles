#!/usr/bin/env python3
"""
python3 export_head_bbox_fp32.py \
 --config /Lidar_AI_Solution/CUDA-BEVFusion/bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml \
 --ckpt   /Lidar_AI_Solution/CUDA-BEVFusion/model/resnet50/bevfusion-det.pth \
 --out    /Lidar_AI_Solution/CUDA-BEVFusion/qat/byUsingConfigs/head.bbox.fp32.onnx
"""

import argparse
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from typing import Tuple

class HeadBBoxWrapper(nn.Module):
    """
    Export-transfuser.py の head.bbox 出力に合わせたラッパ。
    入力:  middle (B, 512, H, W)
    出力:  score, rot, dim, reg(center), height, vel
    """
    def __init__(self, object_head: nn.Module):
        super().__init__()
        self.h = object_head
        self.register_buffer(
            "classes_eye",
            torch.eye(self.h.num_classes).float(),
            persistent=False
        )

    @torch.no_grad()
    def forward(self, middle: torch.Tensor):
        B = int(middle.shape[0])

        # --- shared conv ---
        lidar_feat = self.h.shared_conv(middle)   # (B, C, H, W)
        C = int(lidar_feat.shape[1])
        H = int(lidar_feat.shape[2])
        W = int(lidar_feat.shape[3])
        HW = H * W

        # --- flatten + positional embedding ---
        lidar_feat_flatten = lidar_feat.view(B, C, HW)  # (B, C, HW)

        # bev_pos は多くの実装で (HW, Dpos) or (1, HW, Dpos) を想定
        bev_pos = self.h.bev_pos.to(lidar_feat.dtype).to(lidar_feat.device)
        if bev_pos.dim() == 2:
            # (HW, Dpos) -> (B, HW, Dpos)
            bev_pos = bev_pos.unsqueeze(0).repeat(B, 1, 1)
        elif bev_pos.dim() == 3:
            # (1, HW, Dpos) or (B, HW, Dpos)
            if int(bev_pos.shape[0]) == 1:
                bev_pos = bev_pos.repeat(B, 1, 1)
        else:
            raise RuntimeError(f"Unexpected bev_pos dim: {bev_pos.dim()}")

        # --- dense heatmap (for proposals) ---
        dense_heatmap = self.h.heatmap_head(lidar_feat)  # (B, Nc, H, W)
        heatmap = dense_heatmap.detach().sigmoid()

        # --- simple NMS: max_pool ---
        padding = self.h.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.h.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner

        # dataset-specific exceptions (same idea as common TransFusion impl)
        ds = ""
        try:
            ds = self.h.test_cfg.get("dataset", "")
        except Exception:
            pass
        if ds == "nuScenes":
            if self.h.num_classes > 8:
                local_max[:, 8] = heatmap[:, 8]
            if self.h.num_classes > 9:
                local_max[:, 9] = heatmap[:, 9]
        elif ds == "Waymo":
            if self.h.num_classes > 1:
                local_max[:, 1] = heatmap[:, 1]
            if self.h.num_classes > 2:
                local_max[:, 2] = heatmap[:, 2]

        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(B, self.h.num_classes, HW)  # (B, Nc, HW)

        # --- topK proposals ---
        K = int(self.h.num_proposals)
        top_proposals = heatmap.view(B, -1).topk(k=K, dim=-1, largest=True)[1]  # (B, K)
        top_proposals_class = top_proposals // HW                               # (B, K)
        top_proposals_index = top_proposals % HW                                # (B, K)

        # --- gather query_feat ---
        query_feat = lidar_feat_flatten.gather(
            dim=-1,
            index=top_proposals_index[:, None, :].expand(-1, C, -1),
        )  # (B, C, K)

        # --- one_hot + class encoding ---
        one_hot = self.classes_eye.index_select(0, top_proposals_class.reshape(-1))[None].permute(0, 2, 1)  # (B, Nc, K)
        query_cat_encoding = self.h.class_encoding(one_hot)  # expect (B, C, K) in most impls
        query_feat = query_feat + query_cat_encoding

        # --- query_pos (B, K, Dpos) ---
        query_pos = bev_pos.gather(
            dim=1,
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
        )

        # --- query_heatmap_score (B, Nc, K) ---
        query_heatmap_score = heatmap.gather(
            dim=-1,
            index=top_proposals_index[:, None, :].expand(-1, self.h.num_classes, -1),
        )

        # --- transformer decoder loop ---
        for i in range(int(self.h.num_decoder_layers)):
            query_feat = self.h.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)
            res_layer = self.h.prediction_heads[i](query_feat)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

        # last layer preds
        pred_heatmap = res_layer["heatmap"]       # (B, Nc, K)
        pred_rot = res_layer["rot"]               # (B, ?, K)
        pred_dim = res_layer["dim"]               # (B, ?, K)
        pred_center = res_layer["center"]         # (B, 2, K) etc
        pred_height = res_layer["height"]         # (B, 1, K)
        pred_vel = res_layer.get("vel", torch.zeros_like(pred_center))  # ensure tensor

        # --- get_bboxes equivalent ---
        score = pred_heatmap.sigmoid()
        score = score * query_heatmap_score * one_hot
        reg = pred_center
        return score, pred_rot, pred_dim, reg, pred_height, pred_vel


def infer_middle_hw_from_cfg(cfg: Config) -> Tuple[int, int]:
    """
    あなたが pasted したフルconfigの test_cfg から H,W を確定する。
    典型：grid_size[0]/out_size_factor, grid_size[1]/out_size_factor
    """
    # 優先順位：model.heads.object.test_cfg -> test_cfg -> model.test_cfg
    test_cfg = None
    if "model" in cfg and "heads" in cfg.model and "object" in cfg.model.heads and "test_cfg" in cfg.model.heads.object:
        test_cfg = cfg.model.heads.object.test_cfg
    elif "test_cfg" in cfg:
        test_cfg = cfg.test_cfg
    elif "model" in cfg and "test_cfg" in cfg.model:
        test_cfg = cfg.model.test_cfg

    if test_cfg is None:
        raise RuntimeError("Cannot find test_cfg in cfg. Need grid_size and out_size_factor.")

    grid_size = test_cfg.get("grid_size", None)
    out_size_factor = test_cfg.get("out_size_factor", None)
    if grid_size is None or out_size_factor is None:
        raise RuntimeError(f"test_cfg must include grid_size and out_size_factor. got: {test_cfg}")

    gx, gy = int(grid_size[0]), int(grid_size[1])
    f = int(out_size_factor)
    if gx % f != 0 or gy % f != 0:
        raise RuntimeError(f"grid_size {grid_size} not divisible by out_size_factor {f}")

    H = gx // f
    W = gy // f
    return H, W

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="entry yaml (e.g. convfuser.yaml)")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pth (FP32) e.g. bevfusion-det.pth")
    ap.add_argument("--out", required=True, help="output onnx path e.g. head.bbox.onnx")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    args = ap.parse_args()

    # --- load full config like ptq.py ---
    configs.clear()  # important: avoid cache surprises
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # --- build model (NO dataset build) ---
    # 念のため pretrained 無効化
    if "model" in cfg and isinstance(cfg.model, dict):
        cfg.model.setdefault("pretrained", None)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg", None))
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    model.eval()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device).float()

    # object head
    object_head = model.heads.object.to(device).float().eval()

    # infer input middle shape from cfg
    H, W = infer_middle_hw_from_cfg(cfg)

    # in_channels (middle C) は head の in_channels を基本にする
    in_ch = 512
    try:
        in_ch = int(cfg.model.heads.object.in_channels)
    except Exception:
        pass

    print(f"[INFO] middle shape for export: (1, {in_ch}, {H}, {W})")
    print(f"[INFO] object head: num_classes={object_head.num_classes}, num_proposals={object_head.num_proposals}, "
          f"num_decoder_layers={object_head.num_decoder_layers}")

    wrapper = HeadBBoxWrapper(object_head).to(device).float().eval()

    middle = torch.randn(1, in_ch, H, W, device=device, dtype=torch.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ONNX export (FP32)
    torch.onnx.export(
        wrapper,
        middle,
        args.out,
        opset_version=args.opset,
        input_names=["middle"],
        output_names=["score", "rot", "dim", "reg", "height", "vel"],
        do_constant_folding=False,   # reshape/shape推論トラブル回避に有利
    )

    print(f"[OK] Exported FP32 head.bbox ONNX -> {args.out}")


if __name__ == "__main__":
    main()

# 以下で簡素化
# python3 -m onnxsim /Lidar_AI_Solution/CUDA-BEVFusion/qat/byUsingConfigs/head.bbox.fp32.onnx /Lidar_AI_Solution/CUDA-BEVFusion/qat/byUsingConfigs/head.bbox.fp32.sim.onnx  --overwrite-input-shape "middle:1,512,180,180" 

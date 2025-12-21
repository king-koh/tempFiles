#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _get_by_path(obj, path):
    """cfg / dict を混在で辿る。見つからなければ None"""
    cur = obj
    for k in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            try:
                cur = getattr(cur, k)
            except Exception:
                try:
                    cur = cur[k]
                except Exception:
                    return None
    return cur


def infer_middle_hw_from_cfg(cfg: Config, object_head: Optional[nn.Module] = None) -> Tuple[int, int]:
    """
    grid_size/out_size_factor から H,W を推定。
    test_cfg の位置は環境で揺れるので複数候補を探索。
    見つからない場合は object_head.bev_pos から HW を推定（平方なら H=W）。
    """
    candidates = [
        ["test_cfg"],
        ["model", "test_cfg"],
        ["model", "heads", "object", "test_cfg"],
        ["model", "bbox_head", "test_cfg"],
        ["model", "pts_bbox_head", "test_cfg"],
        ["model", "head", "test_cfg"],
    ]

    for p in candidates:
        tc = _get_by_path(cfg, p)
        if tc is None:
            continue
        try:
            if isinstance(tc, dict):
                grid_size = tc.get("grid_size", None)
                out_size_factor = tc.get("out_size_factor", None)
            else:
                grid_size = getattr(tc, "grid_size", None)
                out_size_factor = getattr(tc, "out_size_factor", None)
        except Exception:
            grid_size, out_size_factor = None, None

        if grid_size is not None and out_size_factor is not None:
            gx, gy = int(grid_size[0]), int(grid_size[1])
            f = int(out_size_factor)
            if gx % f != 0 or gy % f != 0:
                raise RuntimeError(f"grid_size {grid_size} not divisible by out_size_factor {f} (from {'/'.join(p)})")
            H = gx // f
            W = gy // f
            print(f"[INFO] test_cfg found at {'/'.join(p)}: grid_size={grid_size}, out_size_factor={out_size_factor} -> H,W={H},{W}")
            return H, W

    # fallback: bev_pos
    if object_head is not None and hasattr(object_head, "bev_pos"):
        bev_pos = object_head.bev_pos
        if bev_pos.dim() == 2:
            HW = int(bev_pos.shape[0])
        elif bev_pos.dim() == 3:
            HW = int(bev_pos.shape[1])
        else:
            raise RuntimeError(f"Cannot infer HW: unexpected bev_pos dim={bev_pos.dim()} shape={tuple(bev_pos.shape)}")

        s = int(math.sqrt(HW))
        if s * s == HW:
            print(f"[WARN] test_cfg not found; inferred H=W={s} from bev_pos HW={HW}")
            return s, s

        raise RuntimeError(f"test_cfg not found and bev_pos HW={HW} is not a perfect square; please specify H,W manually.")

    top_keys = list(cfg.keys()) if hasattr(cfg, "keys") else []
    model_keys = list(cfg.model.keys()) if hasattr(cfg, "model") and hasattr(cfg.model, "keys") else []
    raise RuntimeError(
        "Cannot find test_cfg (need grid_size/out_size_factor).\n"
        f"top-level keys: {top_keys}\n"
        f"model keys: {model_keys}\n"
        f"checked paths: {', '.join(['/'.join(p) for p in candidates])}"
    )


def disable_pretrained_init_cfg(model_cfg: dict):
    """
    export用途で不要な pretrained download を抑止（環境によりWebアクセスで止まるのを防ぐ）
    """
    if not isinstance(model_cfg, dict):
        return
    try:
        cam_bb = model_cfg.get("encoders", {}).get("camera", {}).get("backbone", {})
        if isinstance(cam_bb, dict) and "init_cfg" in cam_bb:
            cam_bb.pop("init_cfg", None)
    except Exception:
        pass


# ------------------------------------------------------------
# Split core (A: head_pre, B: transformer+pred+get_bboxes)
# ------------------------------------------------------------
class HeadPreCore(nn.Module):
    """
    A側：Transformer に入る前まで
    middle -> (query_feat, lidar_feat_flatten, query_pos, bev_pos, query_heatmap_score, one_hot)
    """
    def __init__(self, object_head: nn.Module):
        super().__init__()
        self.h = object_head
        self.register_buffer("classes_eye", torch.eye(self.h.num_classes).float(), persistent=False)

    def _expand_bev_pos(self, feat: torch.Tensor) -> torch.Tensor:
        """bev_pos を (B, HW, Dpos) に正規化"""
        B = int(feat.shape[0])
        bev_pos = self.h.bev_pos.to(feat.dtype).to(feat.device)
        if bev_pos.dim() == 2:
            bev_pos = bev_pos.unsqueeze(0).repeat(B, 1, 1)
        elif bev_pos.dim() == 3:
            if int(bev_pos.shape[0]) == 1:
                bev_pos = bev_pos.repeat(B, 1, 1)
        else:
            raise RuntimeError(f"Unexpected bev_pos dim: {bev_pos.dim()}")
        return bev_pos

    def forward(self, middle: torch.Tensor):
        B = int(middle.shape[0])

        # shared conv
        lidar_feat = self.h.shared_conv(middle)  # (B, C, H, W)
        C = int(lidar_feat.shape[1])
        H = int(lidar_feat.shape[2])
        W = int(lidar_feat.shape[3])
        HW = H * W

        # flatten
        lidar_feat_flatten = lidar_feat.view(B, C, HW)  # (B, C, HW)
        bev_pos = self._expand_bev_pos(lidar_feat)      # (B, HW, Dpos)

        # dense heatmap
        dense_heatmap = self.h.heatmap_head(lidar_feat)  # (B, Nc, H, W)
        heatmap = dense_heatmap.detach().sigmoid()

        # NMS (max_pool)
        padding = int(self.h.nms_kernel_size) // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=int(self.h.nms_kernel_size), stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner

        # dataset-specific exceptions (TransFusionの慣例)
        ds = ""
        try:
            ds = self.h.test_cfg.get("dataset", "")
        except Exception:
            pass
        if ds == "nuScenes":
            if int(self.h.num_classes) > 8:
                local_max[:, 8] = heatmap[:, 8]
            if int(self.h.num_classes) > 9:
                local_max[:, 9] = heatmap[:, 9]
        elif ds == "Waymo":
            if int(self.h.num_classes) > 1:
                local_max[:, 1] = heatmap[:, 1]
            if int(self.h.num_classes) > 2:
                local_max[:, 2] = heatmap[:, 2]

        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(B, int(self.h.num_classes), HW)  # (B, Nc, HW)

        # topK proposals
        K = int(self.h.num_proposals)
        top = heatmap.view(B, -1).topk(k=K, dim=-1, largest=True)[1]  # (B, K)
        top_class = top // HW
        top_index = top % HW

        # gather query_feat
        query_feat = lidar_feat_flatten.gather(
            dim=-1,
            index=top_index[:, None, :].expand(-1, C, -1),
        )  # (B, C, K)

        # one_hot + class encoding
        one_hot = self.classes_eye.index_select(0, top_class.reshape(-1))[None].permute(0, 2, 1)  # (B, Nc, K)
        query_cat = self.h.class_encoding(one_hot)  # expect (B, C, K)
        query_feat = query_feat + query_cat

        # query_pos
        query_pos = bev_pos.gather(
            dim=1,
            index=top_index[:, None, :].permute(0, 2, 1).expand(-1, -1, int(bev_pos.shape[-1])),
        )  # (B, K, Dpos)

        # query_heatmap_score
        query_heatmap_score = heatmap.gather(
            dim=-1,
            index=top_index[:, None, :].expand(-1, int(self.h.num_classes), -1),
        )  # (B, Nc, K)

        return query_feat, lidar_feat_flatten, query_pos, bev_pos, query_heatmap_score, one_hot


class HeadTransformerCore(nn.Module):
    """
    B側：Transformer decoder + prediction_heads + get_bboxes
    (query_feat, lidar_feat_flatten, query_pos, bev_pos, query_heatmap_score, one_hot)
      -> (score, rot, dim, reg, height, vel)
    """
    def __init__(self, object_head: nn.Module):
        super().__init__()
        self.h = object_head

    def forward(
        self,
        query_feat: torch.Tensor,         # (B, C, K)
        lidar_feat_flatten: torch.Tensor, # (B, C, HW)
        query_pos: torch.Tensor,          # (B, K, Dpos)
        bev_pos: torch.Tensor,            # (B, HW, Dpos)
        query_heatmap_score: torch.Tensor,# (B, Nc, K)
        one_hot: torch.Tensor,            # (B, Nc, K)
    ):
        # transformer decoder loop
        for i in range(int(self.h.num_decoder_layers)):
            query_feat = self.h.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)
            res = self.h.prediction_heads[i](query_feat)
            # center offset
            res["center"] = res["center"] + query_pos.permute(0, 2, 1)
            # ONNX向け：detachしない（グラフを切らない）
            query_pos = res["center"].permute(0, 2, 1)

        pred_heatmap = res["heatmap"]
        pred_rot = res["rot"]
        pred_dim = res["dim"]
        pred_center = res["center"]
        pred_height = res["height"]
        pred_vel = res.get("vel", torch.zeros_like(pred_center))

        # get_bboxes equivalent
        score = pred_heatmap.sigmoid()
        score = score * query_heatmap_score * one_hot
        reg = pred_center
        return score, pred_rot, pred_dim, reg, pred_height, pred_vel


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="entry yaml (e.g. convfuser.yaml)")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pth (FP32) e.g. bevfusion-det.pth")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    # load full config like ptq.py
    if hasattr(configs, "clear"):
        configs.clear()
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # disable pretrained download
    if "model" in cfg and isinstance(cfg.model, dict):
        disable_pretrained_init_cfg(cfg.model)
        cfg.model.setdefault("pretrained", None)

    # build model (NO dataset build)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg", None))
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    model.eval()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device).float()

    object_head = model.heads.object.to(device).float().eval()

    # infer middle shape
    H, W = infer_middle_hw_from_cfg(cfg, object_head=object_head)

    in_ch = 512
    try:
        # mmcv ConfigDict / dict どちらにも対応
        v = _get_by_path(cfg, ["model", "heads", "object", "in_channels"])
        if v is not None:
            in_ch = int(v)
    except Exception:
        pass

    print(f"[INFO] middle shape for export: (1, {in_ch}, {H}, {W})")
    print(f"[INFO] object head: num_classes={int(object_head.num_classes)}, num_proposals={int(object_head.num_proposals)}, "
          f"num_decoder_layers={int(object_head.num_decoder_layers)}")

    # modules
    mod_pre = HeadPreCore(object_head).to(device).float().eval()
    mod_tr = HeadTransformerCore(object_head).to(device).float().eval()

    # dummy middle
    middle = torch.randn(1, in_ch, H, W, device=device, dtype=torch.float32)

    # run once to get transformer inputs
    with torch.no_grad():
        (query_feat, lidar_feat_flatten, query_pos, bev_pos, query_heatmap_score, one_hot) = mod_pre(middle)

    os.makedirs(args.out_dir, exist_ok=True)
    out_pre = os.path.join(args.out_dir, "head.pre.fp32.onnx")
    out_tr = os.path.join(args.out_dir, "head.tr.fp32.onnx")

    # export A: head.pre
    torch.onnx.export(
        mod_pre,
        middle,
        out_pre,
        opset_version=args.opset,
        input_names=["middle"],
        output_names=[
            "query_feat",
            "lidar_feat_flatten",
            "query_pos",
            "bev_pos",
            "query_heatmap_score",
            "one_hot",
        ],
        do_constant_folding=False,
    )
    print(f"[OK] Exported -> {out_pre}")

    # export B: head.tr
    torch.onnx.export(
        mod_tr,
        (query_feat, lidar_feat_flatten, query_pos, bev_pos, query_heatmap_score, one_hot),
        out_tr,
        opset_version=args.opset,
        input_names=[
            "query_feat",
            "lidar_feat_flatten",
            "query_pos",
            "bev_pos",
            "query_heatmap_score",
            "one_hot",
        ],
        output_names=["score", "rot", "dim", "reg", "height", "vel"],
        do_constant_folding=False,
    )
    print(f"[OK] Exported -> {out_tr}")


if __name__ == "__main__":
    main()

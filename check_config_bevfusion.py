from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval
import pprint
import builtins

# =========================
# Settings
# =========================
config_path = "/Lidar_AI_Solution/CUDA-BEVFusion/bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml"

# =========================
# (A) builtins.open hook to log YAML files
# =========================
opened = []
_real_open = builtins.open

def logging_open(file, mode="r", *args, **kwargs):
    try:
        path = str(file)
        if ("r" in mode) and (path.endswith(".yaml") or path.endswith(".yml")):
            opened.append(path)
    except Exception:
        pass
    return _real_open(file, mode, *args, **kwargs)

# ここでフックを有効化（configs.loadより前に必要）
builtins.open = logging_open

try:
    # =========================
    # (1) mmcv.Config.fromfile: "convfuser.yaml を単体読み込み"
    # =========================
    print("\n" + "="*80)
    print("(1) mmcv.Config.fromfile(config_path) で単体読み込み")
    print("="*80)

    cfg_single = Config.fromfile(config_path)
    print(cfg_single)
    print("keys:", cfg_single.keys())
    print("has data?", "data" in cfg_single)

    # =========================
    # (2) torchpack configs.load(recursive=True): "再帰展開してフルconfig化"
    # =========================
    print("\n" + "="*80)
    print("(2) torchpack configs.load(config_path, recursive=True) → recursive_eval → mmcv.Config")
    print("="*80)

    # キャッシュがある環境だと再読み込みされないことがあるので、可能ならclear
    if hasattr(configs, "clear"):
        configs.clear()

    configs.load(config_path, recursive=True)
    cfg_full = Config(recursive_eval(configs), filename=config_path)

    # フルconfigの辞書をそのまま出す（削減しない）
    pprint.pprint(cfg_full._cfg_dict.to_dict())

    print("has data?", "data" in cfg_full)

    # ann_file の取り出し（両方出す：削減しない）
    print("train ann_file (direct):", cfg_full.data.train.get("ann_file", None))
    # CBGSDataset の場合は train.dataset.ann_file に入っていることが多い
    if isinstance(cfg_full.data.train, dict) and ("dataset" in cfg_full.data.train):
        try:
            print("train ann_file (nested dataset):", cfg_full.data.train["dataset"].get("ann_file", None))
        except Exception as e:
            print("train ann_file (nested dataset): ERROR", repr(e))

    # =========================
    # (3) torchpack configs object introspection
    # =========================
    print("\n" + "="*80)
    print("(3) configs オブジェクトの型と属性（中身の見え方確認）")
    print("="*80)

    print("configs type:", type(configs))
    print("attrs:", [a for a in dir(configs) if not a.startswith("__")])

finally:
    # フック解除（必ず元に戻す）
    builtins.open = _real_open

# =========================
# (4) YAML files that were actually opened during configs.load()
# =========================
print("\n" + "="*80)
print("(4) configs.load(recursive=True) 実行中に open() された YAML 一覧")
print("="*80)

for p in sorted(set(opened)):
    print(p)
print("count:", len(set(opened)))

"""
Dataset Tool
  Tab 1 – 两个数据集合并 → 类别筛选 → YOLO 转换
           输入: DatasetA/  DatasetB/  (各含 images/ 文件夹 + COCO JSON 文件)
           输出: output/
                  images/      合并后 + 筛选后的图像（保留原始文件名）
                  labels/      对应的 YOLO txt 标注
                  yolodata/    data.yaml  classes.txt
  Tab 2 – YOLO 数据集划分 (train / val / test)
"""

import os
import json
import math
import queue
import random
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# ──────────────────────────────────────────
# 配色 / 字体
# ──────────────────────────────────────────
BG       = "#1a1d27"
PANEL    = "#22263a"
CARD     = "#2b2f45"
ACCENT   = "#6c63ff"
ACCENT2  = "#a78bfa"
SUCCESS  = "#34d399"
DANGER   = "#f87171"
WARNING  = "#fbbf24"
TEXT     = "#e2e8f0"
TEXT_DIM = "#94a3b8"
BORDER   = "#3d4263"
ENTRY_BG = "#1e2235"

FONT_TITLE = ("Segoe UI", 17, "bold")
FONT_H2    = ("Segoe UI", 12, "bold")
FONT_H3    = ("Segoe UI", 11, "bold")
FONT_BODY  = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO  = ("Consolas", 9)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ══════════════════════════════════════════
# 纯逻辑层
# ══════════════════════════════════════════

def find_json_files(root: Path) -> list:
    """在数据集根目录下递归查找所有 .json 文件"""
    return [p for p in root.rglob("*.json")]


def find_images_dir(root: Path) -> Path:
    """优先返回 root/images/，否则返回 root 本身"""
    d = root / "images"
    return d if d.is_dir() else root


def build_img_lookup(img_dir: Path) -> dict:
    """返回 {文件名: 绝对路径} 和 {stem: 绝对路径}"""
    lookup = {}
    if img_dir and img_dir.is_dir():
        for root, _, files in os.walk(img_dir):
            for f in files:
                if Path(f).suffix.lower() in IMAGE_EXTS:
                    p = Path(root) / f
                    lookup[f] = p
                    lookup[Path(f).stem] = p
    return lookup


def _bbox_from_shape(shape: dict, img_w: int, img_h: int):
    """LabelMe shape → COCO bbox [x_min, y_min, w, h]（统一用外接矩形）"""
    pts = shape.get("points", [])
    if not pts:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(float(img_w), max(xs)) if img_w else max(xs)
    y_max = min(float(img_h), max(ys)) if img_h else max(ys)
    bw, bh = x_max - x_min, y_max - y_min
    return [x_min, y_min, bw, bh] if bw > 0 and bh > 0 else None


def parse_json_files(json_paths: list, log_fn=None, progress_fn=None) -> dict:
    """
    自动识别并解析 JSON 标注文件，支持：
      - LabelMe 格式：每张图一个 JSON，内含 'shapes' 字段
      - COCO 格式：整个数据集一个 JSON，内含 'categories'/'images'/'annotations'
    返回已合并的 COCO-like dict。
    """
    def _log(msg, tag=""):
        if log_fn: log_fn(msg, tag)
    def _prog(v):
        if progress_fn: progress_fn(v)

    cat_name2id: dict = {}
    merged_imgs = []
    merged_anns = []
    img_id_ctr = 0
    ann_id_ctr = 0
    # For COCO offset
    coco_img_off = 0
    coco_ann_off = 0

    n_labelme = 0
    n_coco    = 0
    n_skip    = 0
    total = max(len(json_paths), 1)

    for i, jp in enumerate(json_paths):
        _prog(int(i / total * 95))

        # 读取文件，尝试多种编码
        data = None
        for enc in ("utf-8", "gbk", "utf-8-sig"):
            try:
                with open(jp, encoding=enc) as f:
                    data = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            except Exception as e:
                _log(f"  读取失败 {jp.name}: {e}", "warn")
                break

        if not isinstance(data, dict):
            n_skip += 1
            continue

        # ── LabelMe 格式 ──
        if "shapes" in data:
            img_path = data.get("imagePath", "") or (jp.stem + ".jpg")
            img_fname = Path(img_path).name
            img_w = int(data.get("imageWidth",  0) or 0)
            img_h = int(data.get("imageHeight", 0) or 0)

            this_img_id = img_id_ctr
            img_id_ctr += 1
            merged_imgs.append({
                "id": this_img_id,
                "file_name": img_fname,
                "width":  img_w,
                "height": img_h,
            })

            for shape in data.get("shapes", []):
                label = (shape.get("label") or "").strip()
                if not label:
                    continue
                if label not in cat_name2id:
                    cat_name2id[label] = len(cat_name2id)
                bbox = _bbox_from_shape(shape, img_w, img_h)
                if bbox is None:
                    continue
                merged_anns.append({
                    "id": ann_id_ctr,
                    "image_id": this_img_id,
                    "category_id": cat_name2id[label],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                })
                ann_id_ctr += 1
            n_labelme += 1

        # ── COCO 格式 ──
        elif "categories" in data:
            cats = data.get("categories", [])
            imgs = data.get("images",     [])
            anns = data.get("annotations",[])
            _log(f"  [COCO] {jp.name}: {len(cats)} 类别, "
                 f"{len(imgs)} 图, {len(anns)} 标注", "ok")

            old2new_cat = {}
            for cat in cats:
                name = cat.get("name", f"class_{cat.get('id', 0)}")
                if name not in cat_name2id:
                    cat_name2id[name] = len(cat_name2id)
                old2new_cat[cat["id"]] = cat_name2id[name]

            old2new_img = {}
            for img in imgs:
                nid = img["id"] + coco_img_off
                old2new_img[img["id"]] = nid
                ni = dict(img); ni["id"] = nid
                merged_imgs.append(ni)

            for ann in anns:
                na = dict(ann)
                na["id"] = ann["id"] + coco_ann_off
                na["image_id"]   = old2new_img.get(ann["image_id"], ann["image_id"])
                na["category_id"]= old2new_cat.get(ann.get("category_id"), ann.get("category_id"))
                merged_anns.append(na)

            img_ids = [m["id"] for m in imgs]
            ann_ids = [a["id"] for a in anns]
            coco_img_off += (max(img_ids) + 1) if img_ids else 1
            coco_ann_off += (max(ann_ids) + 1) if ann_ids else 1
            n_coco += 1

        else:
            n_skip += 1

    _prog(100)
    parts = []
    if n_labelme: parts.append(f"LabelMe 格式 {n_labelme} 个")
    if n_coco:    parts.append(f"COCO 格式 {n_coco} 个")
    if n_skip:    parts.append(f"跳过 {n_skip} 个")
    _log("  格式识别: " + "，".join(parts), "info")

    categories = [{"id": v, "name": k}
                  for k, v in sorted(cat_name2id.items(), key=lambda x: x[1])]
    return {"categories": categories, "images": merged_imgs, "annotations": merged_anns}
    """多个 COCO JSON → 一个合并 dict（类别按名称去重，ID 重新分配）"""
    def _log(msg, tag=""):
        if log_fn:
            log_fn(msg, tag)

    cat_name2id: dict = {}
    merged_imgs = []
    merged_anns = []
    img_off = 0
    ann_off = 0

    for jp in json_paths:
        # 尝试 UTF-8，失败则尝试 GBK
        coco = None
        for enc in ("utf-8", "gbk", "utf-8-sig"):
            try:
                with open(jp, encoding=enc) as f:
                    coco = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            except Exception as e:
                _log(f"  读取失败 {jp.name}: {e}", "warn")
                break

        if coco is None:
            _log(f"  跳过（无法解析）: {jp.name}", "warn")
            continue

        if not isinstance(coco, dict):
            _log(f"  跳过（不是 COCO 格式）: {jp.name}", "warn")
            continue

        # 如果连 categories 字段都没有，说明不是标注文件
        if "categories" not in coco:
            _log(f"  跳过（无 categories 字段）: {jp.name}", "warn")
            continue

        cats = coco.get("categories", [])
        imgs = coco.get("images", [])
        anns = coco.get("annotations", [])
        _log(f"  ✓ {jp.name}: {len(cats)} 个类别, {len(imgs)} 张图像, {len(anns)} 个标注", "ok")

        # 类别映射
        old2new_cat = {}
        for cat in cats:
            name = cat.get("name", f"class_{cat.get('id', 0)}")
            if name not in cat_name2id:
                cat_name2id[name] = len(cat_name2id)
            old2new_cat[cat["id"]] = cat_name2id[name]

        # 图像映射
        old2new_img = {}
        img_list = imgs
        for img in img_list:
            nid = img["id"] + img_off
            old2new_img[img["id"]] = nid
            ni = dict(img)
            ni["id"] = nid
            merged_imgs.append(ni)

        # 标注
        ann_list = anns
        for ann in ann_list:
            na = dict(ann)
            na["id"] = ann["id"] + ann_off
            na["image_id"] = old2new_img.get(ann["image_id"], ann["image_id"])
            na["category_id"] = old2new_cat.get(ann.get("category_id"), ann.get("category_id"))
            merged_anns.append(na)

        img_ids = [i["id"] for i in img_list]
        ann_ids = [a["id"] for a in ann_list]
        img_off += (max(img_ids) + 1) if img_ids else 1
        ann_off += (max(ann_ids) + 1) if ann_ids else 1

    categories = [{"id": v, "name": k}
                  for k, v in sorted(cat_name2id.items(), key=lambda x: x[1])]
    return {"categories": categories, "images": merged_imgs, "annotations": merged_anns}


def convert_and_export(coco: dict, kept_names: set, name_mapping: dict,
                       lookup_a: dict, lookup_b: dict,
                       out_dir: Path, log_fn, prog_fn):
    """
    核心转换：
    - 只保留 kept_names 类别的标注
    - 将类别名称根据 name_mapping [原名 -> 新名] 进行重命名并重新分配 ID
    - 只处理有剩余标注的图像
    - 图像复制到 out_dir/images/ （原始文件名）
    - YOLO txt 写到 out_dir/labels/
    - COCO 格式对应单图的 JSON 写到 out_dir/labels/
    - data.yaml / classes.txt 写到 out_dir/yolodata/
    返回 (n_images, n_skipped, class_names)
    """
    # 过滤与重命名类别
    all_cats  = sorted(coco.get("categories", []), key=lambda c: c["id"])
    kept_cats = [c for c in all_cats if c["name"] in kept_names]
    if not kept_cats:
        raise ValueError("没有选中任何类别，请至少勾选一个。")

    # 构建合并后的新类别 (去重)
    new_cat_name2id = {}
    old_cat_to_yolo = {}  # old_id -> new_yolo_id
    for c in kept_cats:
        old_name = c["name"]
        new_name = name_mapping.get(old_name, old_name)
        if new_name not in new_cat_name2id:
            new_cat_name2id[new_name] = len(new_cat_name2id)
        old_cat_to_yolo[c["id"]] = new_cat_name2id[new_name]

    class_names = [k for k, v in sorted(new_cat_name2id.items(), key=lambda x: x[1])]
    log_fn(f"  合并后保留类别 ({len(class_names)}): {class_names}", "info")

    # 按 image_id 聚合有效标注
    kept_cat_ids = set(old_cat_to_yolo.keys())
    anns_by_img: dict = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") in kept_cat_ids:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

    images_meta = {img["id"]: img for img in coco.get("images", [])}
    # 只处理有剩余标注的图像
    valid_img_ids = {iid for iid in images_meta if iid in anns_by_img}
    n_skipped = len(images_meta) - len(valid_img_ids)
    log_fn(f"  有效图像: {len(valid_img_ids)}，跳过无标注图像: {n_skipped}", "ok")

    out_images   = out_dir / "images"
    out_labels   = out_dir / "labels"
    out_yolodata = out_dir / "yolodata"
    for d in [out_images, out_labels, out_yolodata]:
        d.mkdir(parents=True, exist_ok=True)

    total = max(len(valid_img_ids), 1)
    n_converted = 0
    n_img_copied = 0
    lookup = {**lookup_b, **lookup_a}   # A 优先覆盖 B

    for i, img_id in enumerate(sorted(valid_img_ids)):
        prog_fn(int(i / total * 88))
        img_info = images_meta[img_id]
        file_name = img_info.get("file_name", "")
        fname     = Path(file_name).name
        stem      = Path(file_name).stem
        img_w = img_info.get("width",  0)
        img_h = img_info.get("height", 0)

        # YOLO 标注行与 JSON 标注对象
        yolo_lines = []
        json_shapes = []

        for ann in anns_by_img.get(img_id, []):
            yolo_cls = old_cat_to_yolo.get(ann.get("category_id"))
            if yolo_cls is None:
                continue
            bbox = ann.get("bbox")
            if not bbox or img_w == 0 or img_h == 0:
                continue
            xmin, ymin, bw, bh = bbox
            xc = max(0.0, min(1.0, (xmin + bw / 2) / img_w))
            yc = max(0.0, min(1.0, (ymin + bh / 2) / img_h))
            wn = max(0.0, min(1.0, bw / img_w))
            hn = max(0.0, min(1.0, bh / img_h))
            yolo_lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            # 构造 LabelMe 风格的 shape
            json_shapes.append({
                "label": class_names[yolo_cls],
                "points": [
                    [xmin, ymin],
                    [xmin + bw, ymin + bh]
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            })

        # 写入 txt
        (out_labels / (stem + ".txt")).write_text("\n".join(yolo_lines), encoding="utf-8")

        # 写入对应 json
        json_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": json_shapes,
            "imagePath": fname,
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w
        }
        with open(out_labels / (stem + ".json"), "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2, ensure_ascii=False)

        n_converted += 1

        # 复制图像
        src = lookup.get(fname) or lookup.get(stem)
        if src and Path(src).exists():
            dst = out_images / fname
            if dst.exists():
                log_fn(f"  ⚠ 文件名冲突（已覆盖）: {fname}", "warn")
            shutil.copy2(src, dst)
            n_img_copied += 1
        else:
            log_fn(f"  ⚠ 图像未找到: {fname}", "warn")

    prog_fn(93)

    # classes.txt
    (out_yolodata / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")

    # data.yaml（路径指向上级的 images/ 和 labels/）
    yaml_lines = [
        f"path: {out_dir.resolve()}",
        "train: images",
        "val: images",
        f"nc: {len(class_names)}",
        f"names: {class_names}"
    ]
    (out_yolodata / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    prog_fn(100)
    return n_converted, n_skipped, class_names


# ──────────────────────────────────────────
# 数据集划分
# ──────────────────────────────────────────

def split_dataset(src_dir: Path, out_dir: Path, ratios: dict,
                  seed: int, log_fn, prog_fn):
    img_dir = src_dir / "images" if (src_dir / "images").is_dir() else src_dir
    lbl_dir = src_dir / "labels" if (src_dir / "labels").is_dir() else src_dir

    all_imgs = []
    for root, _, files in os.walk(img_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                all_imgs.append(Path(root) / f)
    if not all_imgs:
        raise ValueError(f"在 {img_dir} 中未找到图像文件")

    random.seed(seed)
    total = len(all_imgs)
    log_fn(f"共 {total} 张图像，开始按主导类别分层划分…", "info")

    # 1. 扫描所有标签，统计全局各类别出现频率
    class_global_counts = {}
    img_to_classes = {}
    
    prog_fn(5)
    log_fn("正在扫描标签以了解类别分布...", "info")
    for idx, img_path in enumerate(all_imgs):
        if idx % max(1, total // 20) == 0:
            prog_fn(5 + int(idx / total * 15))
        
        lbl_file = lbl_dir / (img_path.stem + ".txt")
        classes_in_img = set()
        if lbl_file.exists():
            try:
                for line in lbl_file.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        cls_id = int(parts[0])
                        classes_in_img.add(cls_id)
                        class_global_counts[cls_id] = class_global_counts.get(cls_id, 0) + 1
            except Exception:
                pass
        img_to_classes[img_path] = classes_in_img

    # 2. 为每张图片分配一个“最稀有”的主导类别
    groups = {}  # primary_class_id -> list of images
    fallback_group = [] # 对于完全没有标签的图像
    
    for img_path, classes in img_to_classes.items():
        if not classes:
            fallback_group.append(img_path)
        else:
            # 找到这张图里，全局出现总数最少的那个类别作为这张图的主要身份
            rarest_cls = min(classes, key=lambda c: class_global_counts.get(c, 0))
            if rarest_cls not in groups:
                groups[rarest_cls] = []
            groups[rarest_cls].append(img_path)
            
    group_keys = list(groups.keys())
    if fallback_group:
        group_keys.append("unlabeled")
        groups["unlabeled"] = fallback_group
    
    log_fn(f"图像已被分组到 {len(group_keys)} 个独立分层中以保证比例均衡。", "info")

    # 准备划分容器
    keys = [k for k, v in ratios.items() if v > 0]
    assigned = {k: [] for k in keys}
    
    # 3. 在每个隔离的组内独立洗牌并划分
    # 这样每个子组都会被按比例切开，总合起来就是绝对平衡的
    for grp_id, grp_imgs in groups.items():
        random.shuffle(grp_imgs)
        grp_total = len(grp_imgs)
        
        counts = {}
        used = 0
        for k in keys[:-1]:
            counts[k] = math.floor(ratios[k] * grp_total)
            used += counts[k]
        counts[keys[-1]] = grp_total - used
        
        idx = 0
        for split_name, cnt in counts.items():
            assigned[split_name].extend(grp_imgs[idx: idx + cnt])
            idx += cnt

    # 4. 执行文件复制与写入
    prog_fn(25)
    log_fn("正在将划分结果写入磁盘并复制文件...", "info")
    
    txt_files = {}   # split_name -> list of abs image paths
    
    copied = 0
    for split_name, imgs in assigned.items():
        if not imgs:
            continue
            
        si = out_dir / split_name / "images"
        sl = out_dir / split_name / "labels"
        si.mkdir(parents=True, exist_ok=True)
        sl.mkdir(parents=True, exist_ok=True)
        
        paths_in_split = []
        for img in imgs:
            prog_fn(25 + int(copied / total * 70))
            copied += 1
            
            dst_img = si / img.name
            shutil.copy2(img, dst_img)
            paths_in_split.append(str(dst_img.resolve()))
            
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, sl / lbl.name)
                
            json_lbl = lbl_dir / (img.stem + ".json")
            if json_lbl.exists():
                shutil.copy2(json_lbl, sl / json_lbl.name)
                
        txt_files[split_name] = paths_in_split
        log_fn(f"  {split_name}: {len(imgs)} 张", "ok")

    # 生成 train.txt / val.txt / test.txt
    for split_name, paths in txt_files.items():
        txt_path = out_dir / f"{split_name}.txt"
        txt_path.write_text("\n".join(paths), encoding="utf-8")
        log_fn(f"  → {txt_path.name}  ({len(paths)} 行)", "ok")

    # classes.txt / data.yaml
    class_names = []
    cf = src_dir / "classes.txt"
    if not cf.exists():
        cf = src_dir / "yolodata" / "classes.txt"
    if cf.exists():
        class_names = [l.strip() for l in cf.read_text(encoding="utf-8").splitlines() if l.strip()]
        shutil.copy2(cf, out_dir / "classes.txt")

    if class_names:
        yaml_lines = [
            f"path: {out_dir.resolve()}",
            f"nc: {len(class_names)}",
            f"names: {class_names}"
        ]
        for k in ["train", "val", "test"]:
            if k in txt_files:
                yaml_lines.insert(1, f"{k}: {k}.txt")
                
        (out_dir / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    prog_fn(100)
    return total


# ══════════════════════════════════════════
# GUI
# ══════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dataset Tool")
        self.geometry("1060x840")
        self.minsize(860, 680)
        self.configure(bg=BG)
        self.resizable(True, True)

        self._q: queue.Queue = queue.Queue()

        # Tab1 状态
        self.ds_a_var     = tk.StringVar()
        self.ds_b_var     = tk.StringVar()
        self.out_var      = tk.StringVar()
        self.merged_coco  = {}
        self.lookup_a     = {}
        self.lookup_b     = {}
        self.cat_vars: dict = {}   # name -> BooleanVar

        # Tab2 状态
        self.sp_src_var   = tk.StringVar()
        self.sp_out_var   = tk.StringVar()
        self.train_var    = tk.StringVar(value="70")
        self.val_var      = tk.StringVar(value="20")
        self.test_var     = tk.StringVar(value="10")
        self.seed_var     = tk.StringVar(value="42")

        self._build_ui()
        self._apply_styles()
        self._poll_queue()

    # ── 队列轮询 ────────────────────────────────────────────
    def _poll_queue(self):
        try:
            while True:
                item = self._q.get_nowait()
                action = item[0]
                args   = item[1:]
                if   action == "log1":   self._wlog(self.log1, *args)
                elif action == "log2":   self._wlog(self.log2, *args)
                elif action == "prog1":  self.prog1["value"] = args[0]
                elif action == "prog2":  self.prog2["value"] = args[0]
                elif action == "info":   messagebox.showinfo("完成", args[0])
                elif action == "err":    messagebox.showerror("错误", args[0])
                elif action == "cats":   self._build_cat_ui(args[0])
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    # 线程安全接口
    def _mlog1(self, msg, tag=""): self._q.put(("log1", msg, tag))
    def _mlog2(self, msg, tag=""): self._q.put(("log2", msg, tag))
    def _mprog1(self, v):          self._q.put(("prog1", v))
    def _mprog2(self, v):          self._q.put(("prog2", v))

    # ── 样式 ────────────────────────────────────────────────
    def _apply_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame",    background=BG)
        s.configure("TNotebook", background=PANEL, borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=TEXT_DIM,
                    font=("Segoe UI", 10, "bold"), padding=[20, 8])
        s.map("TNotebook.Tab",
              background=[("selected", CARD), ("active", BORDER)],
              foreground=[("selected", ACCENT2), ("active", TEXT)])
        s.configure("TLabel",    background=BG,    foreground=TEXT, font=FONT_BODY)
        s.configure("TEntry",    fieldbackground=ENTRY_BG, foreground=TEXT,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                    insertcolor=TEXT)
        s.configure("TButton",   background=CARD, foreground=TEXT,
                    font=FONT_BODY, borderwidth=1, bordercolor=BORDER)
        s.map("TButton", background=[("active", BORDER), ("pressed", BG)])
        s.configure("TCheckbutton", background=CARD, foreground=TEXT, font=FONT_BODY)
        s.map("TCheckbutton", background=[("active", CARD)])
        s.configure("TScrollbar", background=PANEL, troughcolor=BG,
                    gripcount=0, arrowcolor=TEXT_DIM)
        s.configure("TProgressbar", troughcolor=BG,
                    background=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT)

    # ── 顶栏 + 标签页 ───────────────────────────────────────
    def _build_ui(self):
        bar = tk.Frame(self, bg=PANEL, height=56)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="🗂  Dataset Tool",
                 font=FONT_TITLE, bg=PANEL, fg=TEXT).pack(side="left", padx=22, pady=8)
        tk.Label(bar, text="COCO JSON 合并 · 类别筛选 · YOLO 转换 · 数据集划分",
                 font=FONT_SMALL, bg=PANEL, fg=TEXT_DIM).pack(side="left", pady=18)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        t1 = tk.Frame(nb, bg=BG)
        nb.add(t1, text="  📋 合并 & 转换  ")
        t2 = tk.Frame(nb, bg=BG)
        nb.add(t2, text="  ✂  数据集划分  ")
        self._build_tab1(t1)
        self._build_tab2(t2)

    # ══════════════════════════════════════════
    # Tab 1
    # ══════════════════════════════════════════
    def _build_tab1(self, parent):
        _, sf = self._scrollable(parent)

        # ① 输入
        s1 = self._card(sf, "① 选择两个数据集文件夹")
        self._path_row(s1, "数据集 A 根目录", self.ds_a_var,
                       lambda: self._bdir(self.ds_a_var))
        self._path_row(s1, "数据集 B 根目录", self.ds_b_var,
                       lambda: self._bdir(self.ds_b_var))
        self._path_row(s1, "输出根目录",       self.out_var,
                       lambda: self._bdir(self.out_var))
        tip = tk.Frame(s1, bg=CARD)
        tip.pack(fill="x", padx=16, pady=(0, 6))
        tk.Label(tip,
                 text="程序将自动在根目录下查找 images/ 文件夹（图像）和 .json 文件（类别信息）",
                 font=FONT_SMALL, bg=CARD, fg=TEXT_DIM).pack(anchor="w")
        btn_row = tk.Frame(s1, bg=CARD)
        btn_row.pack(fill="x", padx=16, pady=(4, 14))
        tk.Button(btn_row, text="🔍  合并 JSON & 扫描类别",
                  font=("Segoe UI", 10, "bold"),
                  bg="#f59e0b", fg="#1a1d27",
                  activebackground="#fbbf24", activeforeground="#1a1d27",
                  relief="flat", bd=0, padx=16, pady=6, cursor="hand2",
                  command=self._do_scan).pack(side="left")
        tk.Label(btn_row, text="   扫描完成后可筛选类别，再点击「开始转换」",
                 font=FONT_SMALL, bg=CARD, fg=TEXT_DIM).pack(side="left")

        # ② 类别筛选（动态）
        self.cat_card = self._card(sf, "② 类别筛选  ——  取消勾选的类别将被剔除（对应图像也会移除）")
        self._cat_placeholder()

        # ③ 转换
        s3 = self._card(sf, "③ 开始转换")
        tk.Label(s3,
                 text="  转换后输出结构：output/images/  output/labels/  output/yolodata/",
                 font=FONT_SMALL, bg=CARD, fg=TEXT_DIM).pack(anchor="w", padx=16, pady=(8, 4))
        tk.Button(s3, text="🚀  开始转换",
                  font=("Segoe UI", 11, "bold"),
                  bg=SUCCESS, fg="#0d1117",
                  activebackground="#6ee7b7", activeforeground="#0d1117",
                  relief="flat", bd=0, padx=22, pady=8, cursor="hand2",
                  command=self._do_convert).pack(padx=16, pady=(2, 14), anchor="w")

        self._log_widget(sf, tab=1)

    # 类别 UI ──────────────────────────────────────────────
    def _cat_placeholder(self):
        for w in self.cat_card.winfo_children():
            w.destroy()
        tk.Label(self.cat_card,
                 text="请先选择两个数据集文件夹，然后点击「合并 JSON & 扫描类别」",
                 font=FONT_BODY, bg=CARD, fg=TEXT_DIM).pack(padx=20, pady=22)

    def _build_cat_ui(self, cat_names: list):
        """主线程中重建类别 checkbox 及重命名 UI"""
        for w in self.cat_card.winfo_children():
            w.destroy()
        self.cat_vars = {}
        self.cat_renames = {}

        tb = tk.Frame(self.cat_card, bg=CARD)
        tb.pack(fill="x", padx=16, pady=(10, 6))
        tk.Label(tb, text=f"共检测到 {len(cat_names)} 个类别",
                 font=FONT_H3, bg=CARD, fg=ACCENT2).pack(side="left")
        tk.Button(tb, text="全选", font=FONT_SMALL,
                  bg=SUCCESS, fg="#0d1117", relief="flat", bd=0,
                  padx=8, pady=2, cursor="hand2",
                  command=lambda: [v.set(True) for v in self.cat_vars.values()]
                  ).pack(side="right")
        tk.Button(tb, text="全不选", font=FONT_SMALL,
                  bg=DANGER, fg="white", relief="flat", bd=0,
                  padx=8, pady=2, cursor="hand2",
                  command=lambda: [v.set(False) for v in self.cat_vars.values()]
                  ).pack(side="right", padx=6)
        tk.Label(tb, text="勾选以保留，右侧填入新名称可修改类别（留空不改）",
                 font=FONT_SMALL, bg=CARD, fg=TEXT_DIM).pack(side="left", padx=(10, 0))

        COLS = 3
        rows_needed = math.ceil(len(cat_names) / COLS)
        c_height = min(300, rows_needed * 38 + 10)
        oc = tk.Canvas(self.cat_card, bg=CARD, highlightthickness=0, height=c_height)
        sb = ttk.Scrollbar(self.cat_card, orient="vertical", command=oc.yview)
        oc.configure(yscrollcommand=sb.set)
        if rows_needed > 6:
            sb.pack(side="right", fill="y")
        oc.pack(fill="x", padx=16, pady=(0, 12))
        grid = tk.Frame(oc, bg=CARD)
        win  = oc.create_window((0, 0), window=grid, anchor="nw")
        grid.bind("<Configure>", lambda e: (
            oc.configure(scrollregion=oc.bbox("all")),
            oc.itemconfig(win, width=oc.winfo_width())
        ))
        oc.bind("<Configure>", lambda e: oc.itemconfig(win, width=e.width))
        oc.bind("<MouseWheel>", lambda e: oc.yview_scroll(int(-1 * e.delta / 120), "units"))

        for i, name in enumerate(cat_names):
            var = tk.BooleanVar(value=True)
            self.cat_vars[name] = var
            cell = tk.Frame(grid, bg=PANEL, padx=8, pady=4)
            cell.grid(row=i // COLS, column=i % COLS, padx=4, pady=3, sticky="w")
            
            cb = tk.Checkbutton(cell, text=name[:15] + ("..." if len(name)>15 else ""),
                                variable=var, width=15, anchor="w",
                                bg=PANEL, fg=TEXT, selectcolor=BG,
                                activebackground=PANEL, font=FONT_MONO,
                                cursor="hand2")
            cb.pack(side="left")
            
            # 添加重命名输入框
            tk.Label(cell, text="->", bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL).pack(side="left")
            ren_var = tk.StringVar(value=name)
            self.cat_renames[name] = ren_var
            tk.Entry(cell, textvariable=ren_var, width=15, font=FONT_MONO,
                     bg=BG, fg=TEXT, bd=1, relief="solid").pack(side="left", padx=(4, 0))

    # 扫描 worker ───────────────────────────────────────────
    def _do_scan(self):
        a = self.ds_a_var.get().strip()
        b = self.ds_b_var.get().strip()
        if not a or not os.path.isdir(a):
            messagebox.showwarning("路径", "请先选择数据集 A 文件夹。")
            return
        if not b or not os.path.isdir(b):
            messagebox.showwarning("路径", "请先选择数据集 B 文件夹。")
            return

        self._wlog(self.log1, "─" * 48, "info")
        self._wlog(self.log1, "正在扫描 JSON 文件…", "info")
        threading.Thread(target=self._scan_worker,
                         args=(Path(a), Path(b)), daemon=True).start()

    def _scan_worker(self, pa: Path, pb: Path):
        try:
            # 查找 JSON
            jsons_a = find_json_files(pa)
            jsons_b = find_json_files(pb)
            all_jsons = jsons_a + jsons_b
            self._mlog1(f"  数据集 A: 找到 {len(jsons_a)} 个 JSON", "ok")
            self._mlog1(f"  数据集 B: 找到 {len(jsons_b)} 个 JSON", "ok")

            if not all_jsons:
                self._mlog1("❌ 未找到任何 JSON 文件，请确认数据集格式。", "err")
                return

            # 建立图像查找表
            img_dir_a = find_images_dir(pa)
            img_dir_b = find_images_dir(pb)
            self.lookup_a = build_img_lookup(img_dir_a)
            self.lookup_b = build_img_lookup(img_dir_b)
            cnt_a = len([k for k in self.lookup_a if "." in k])
            cnt_b = len([k for k in self.lookup_b if "." in k])
            self._mlog1(f"  数据集 A images/: {cnt_a} 张图像", "ok")
            self._mlog1(f"  数据集 B images/: {cnt_b} 张图像", "ok")

            # 解析所有 JSON（自动识别 LabelMe / COCO）
            n_all = len(all_jsons)
            self._mlog1(f"正在解析 {n_all} 个 JSON 文件，请稍候…", "info")
            merged = parse_json_files(
                all_jsons,
                log_fn=self._mlog1,
                progress_fn=self._mprog1,
            )
            self.merged_coco = merged
            cat_names = [c["name"] for c in sorted(merged["categories"], key=lambda x: x["id"])]
            n_img = len(merged["images"])
            n_ann = len(merged["annotations"])
            self._mlog1(f"✓ 完成：{n_img} 张图像，{n_ann} 个标注，{len(cat_names)} 个类别", "ok")
            self._q.put(("cats", cat_names))
        except Exception as ex:
            import traceback
            self._mlog1(f"❌ {ex}", "err")
            self._mlog1(traceback.format_exc(), "err")

    # 转换 worker ───────────────────────────────────────────
    def _do_convert(self):
        if not self.merged_coco:
            messagebox.showwarning("未扫描", "请先点击「合并 JSON & 扫描类别」。")
            return
        out = self.out_var.get().strip()
        if not out:
            messagebox.showwarning("输出路径", "请先选择输出根目录。")
            return
        
        kept = {name for name, var in self.cat_vars.items() if var.get()}
        if not kept:
            messagebox.showwarning("无类别", "请至少保留一个类别。")
            return
            
        name_mapping = {}
        for name in kept:
            new_name = self.cat_renames[name].get().strip()
            name_mapping[name] = new_name if new_name else name

        self.prog1["value"] = 0
        self._wlog(self.log1, "─" * 48, "info")
        self._wlog(self.log1, f"开始转换，保留 {len(kept)} 个原类别并映射为 {len(set(name_mapping.values()))} 个新类别…", "info")
        threading.Thread(target=self._convert_worker,
                         args=(dict(self.merged_coco), kept, name_mapping, Path(out)),
                         daemon=True).start()

    def _convert_worker(self, coco, kept, name_mapping, out_dir):
        try:
            n_img, n_skip, classes = convert_and_export(
                coco, kept, name_mapping,
                self.lookup_a, self.lookup_b,
                out_dir,
                log_fn=self._mlog1,
                prog_fn=self._mprog1,
            )
            self._mlog1(f"✅ 完成！{n_img} 张图像，{n_skip} 张被筛掉（无有效标注）", "ok")
            self._mlog1(f"📁 images/   → {out_dir / 'images'}", "ok")
            self._mlog1(f"📁 labels/   → {out_dir / 'labels'}", "ok")
            self._mlog1(f"📁 yolodata/ → {out_dir / 'yolodata'}", "ok")
            self._q.put(("info",
                         f"转换完成！\n{n_img} 张图像，跳过 {n_skip} 张\n"
                         f"类别: {classes}\n输出目录：{out_dir}"))
        except Exception as ex:
            import traceback
            self._mlog1(f"❌ {ex}", "err")
            self._mlog1(traceback.format_exc(), "err")

    # ══════════════════════════════════════════
    # Tab 2
    # ══════════════════════════════════════════
    def _build_tab2(self, parent):
        _, sf = self._scrollable(parent)

        s1 = self._card(sf, "① 数据集来源")
        self._path_row(s1, "YOLO 数据集目录", self.sp_src_var,
                       lambda: self._bdir(self.sp_src_var))
        tk.Label(s1,
                 text="  该目录下需有 images/ 和 labels/ 文件夹，以及 classes.txt 或 yolodata/classes.txt",
                 font=FONT_SMALL, bg=CARD, fg=TEXT_DIM).pack(anchor="w", padx=16, pady=(0, 10))

        s2 = self._card(sf, "② 划分比例（三项之和应为 100）")
        self._ratio_row(s2)

        s3 = self._card(sf, "③ 输出设置")
        self._path_row(s3, "输出根目录", self.sp_out_var,
                       lambda: self._bdir(self.sp_out_var))
        seed_row = tk.Frame(s3, bg=CARD)
        seed_row.pack(fill="x", padx=16, pady=(0, 10))
        tk.Label(seed_row, text="随机种子 (seed)", font=FONT_BODY,
                 bg=CARD, fg=TEXT, width=16, anchor="w").pack(side="left")
        tk.Entry(seed_row, textvariable=self.seed_var, font=FONT_MONO,
                 bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                 relief="flat", bd=4, highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=ACCENT,
                 width=8).pack(side="left", ipady=3)
        tk.Button(s3, text="✂  开始划分",
                  font=("Segoe UI", 11, "bold"),
                  bg=SUCCESS, fg="#0d1117",
                  activebackground="#6ee7b7", activeforeground="#0d1117",
                  relief="flat", bd=0, padx=22, pady=8, cursor="hand2",
                  command=self._do_split).pack(padx=16, pady=(4, 14), anchor="w")

        self._log_widget(sf, tab=2)

    def _ratio_row(self, parent):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=16, pady=10)
        total_lbl = tk.Label(row, text="∑ = 100", font=FONT_H2, bg=CARD, fg=SUCCESS)

        for label, var, color in [
            ("训练集 train %", self.train_var, ACCENT2),
            ("验证集 val   %", self.val_var,   SUCCESS),
            ("测试集 test  %", self.test_var,  WARNING),
        ]:
            col = tk.Frame(row, bg=PANEL, padx=14, pady=10)
            col.pack(side="left", padx=(0, 10))
            tk.Label(col, text=label, font=FONT_SMALL, bg=PANEL, fg=color).pack(anchor="w")
            tk.Entry(col, textvariable=var, font=("Segoe UI", 16, "bold"),
                     bg=ENTRY_BG, fg=color, insertbackground=color,
                     relief="flat", bd=2, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=color,
                     width=6, justify="center").pack(pady=(4, 0))

        total_lbl.pack(side="left", padx=20)

        def upd(*_):
            try:
                s = sum(int(v.get() or 0) for v in [self.train_var, self.val_var, self.test_var])
                total_lbl.config(text=f"∑ = {s}", fg=SUCCESS if s == 100 else DANGER)
            except ValueError:
                total_lbl.config(text="∑ = ?", fg=WARNING)

        for v in [self.train_var, self.val_var, self.test_var]:
            v.trace_add("write", upd)

    def _do_split(self):
        src = self.sp_src_var.get().strip()
        out = self.sp_out_var.get().strip()
        if not src or not os.path.isdir(src):
            messagebox.showwarning("来源", "请选择有效的数据集目录。")
            return
        if not out:
            messagebox.showwarning("输出", "请选择输出根目录。")
            return
        try:
            tr  = int(self.train_var.get())
            vl  = int(self.val_var.get())
            te  = int(self.test_var.get())
            sd  = int(self.seed_var.get())
            assert tr + vl + te == 100, "比例之和必须为 100"
            assert tr >= 0 and vl >= 0 and te >= 0
        except (ValueError, AssertionError) as e:
            messagebox.showerror("比例错误", str(e))
            return
        ratios = {"train": tr / 100, "val": vl / 100, "test": te / 100}
        self.prog2["value"] = 0
        self._wlog(self.log2, "─" * 48, "info")
        self._wlog(self.log2, f"train {tr}% / val {vl}% / test {te}%  seed={sd}", "info")
        threading.Thread(target=self._split_worker,
                         args=(Path(src), Path(out), ratios, sd),
                         daemon=True).start()

    def _split_worker(self, src, out, ratios, seed):
        try:
            total = split_dataset(src, out, ratios, seed,
                                  log_fn=self._mlog2,
                                  prog_fn=self._mprog2)
            self._mlog2(f"✅ 划分完成！共 {total} 张 → {out}", "ok")
            self._q.put(("info", f"划分完成！共 {total} 张图像\n输出目录：{out}"))
        except Exception as ex:
            import traceback
            self._mlog2(f"❌ {ex}", "err")
            self._mlog2(traceback.format_exc(), "err")

    # ── 共用 UI ─────────────────────────────────────────────
    def _scrollable(self, parent):
        c  = tk.Canvas(parent, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=c.yview)
        c.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        c.pack(side="left", fill="both", expand=True)
        sf  = tk.Frame(c, bg=BG)
        win = c.create_window((0, 0), window=sf, anchor="nw")
        sf.bind("<Configure>", lambda e: (
            c.configure(scrollregion=c.bbox("all")),
            c.itemconfig(win, width=c.winfo_width())
        ))
        c.bind("<Configure>",  lambda e: c.itemconfig(win, width=e.width))
        c.bind("<MouseWheel>", lambda e: c.yview_scroll(int(-1 * e.delta / 120), "units"))
        sf.bind("<MouseWheel>",lambda e: c.yview_scroll(int(-1 * e.delta / 120), "units"))
        return c, sf

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill="x", padx=20, pady=(10, 0))
        tk.Label(outer, text=title, font=FONT_H2, bg=BG, fg=ACCENT2).pack(anchor="w", pady=(6, 3))
        card = tk.Frame(outer, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="x")
        return card

    def _path_row(self, parent, label, var, cmd, lw=16):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=16, pady=6)
        tk.Label(row, text=label, font=FONT_BODY, bg=CARD, fg=TEXT,
                 width=lw, anchor="w").pack(side="left")
        e = tk.Entry(row, textvariable=var, font=FONT_MONO,
                     bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                     relief="flat", bd=4, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=ACCENT)
        e.pack(side="left", fill="x", expand=True, ipady=4)
        tk.Button(row, text="浏览…", font=FONT_BODY,
                  bg=BORDER, fg=TEXT, activebackground=ACCENT,
                  activeforeground="white", relief="flat", bd=0,
                  padx=10, cursor="hand2", command=cmd
                  ).pack(side="left", padx=(6, 0))
        return e

    def _log_widget(self, parent, tab=1):
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill="x", padx=20, pady=(10, 20))
        tk.Label(outer, text="日志 / Log", font=FONT_H2, bg=BG, fg=ACCENT2).pack(anchor="w", pady=(6, 3))
        card = tk.Frame(outer, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="x")
        if tab == 1:
            self.prog1 = ttk.Progressbar(card, mode="determinate")
            self.prog1.pack(fill="x", padx=12, pady=(8, 4))
            self.log1  = tk.Text(card, height=7, font=FONT_MONO,
                                 bg="#0d1117", fg=TEXT, insertbackground=TEXT,
                                 relief="flat", bd=0, wrap="word", state="disabled")
            self.log1.pack(fill="x", padx=12, pady=(0, 10))
            for tag, fg in [("ok", SUCCESS), ("warn", WARNING), ("err", DANGER), ("info", ACCENT2)]:
                self.log1.tag_configure(tag, foreground=fg)
        else:
            self.prog2 = ttk.Progressbar(card, mode="determinate")
            self.prog2.pack(fill="x", padx=12, pady=(8, 4))
            self.log2  = tk.Text(card, height=7, font=FONT_MONO,
                                 bg="#0d1117", fg=TEXT, insertbackground=TEXT,
                                 relief="flat", bd=0, wrap="word", state="disabled")
            self.log2.pack(fill="x", padx=12, pady=(0, 10))
            for tag, fg in [("ok", SUCCESS), ("warn", WARNING), ("err", DANGER), ("info", ACCENT2)]:
                self.log2.tag_configure(tag, foreground=fg)

    @staticmethod
    def _wlog(widget, msg, tag=""):
        widget.configure(state="normal")
        widget.insert("end", msg + "\n", tag)
        widget.see("end")
        widget.configure(state="disabled")

    def _bdir(self, var):
        d = filedialog.askdirectory(title="选择文件夹")
        if d:
            var.set(d)


# ──────────────────────────────────────────
# 入口
# ──────────────────────────────────────────
if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        App().mainloop()
    except KeyboardInterrupt:
        pass

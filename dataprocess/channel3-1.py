import argparse
from pathlib import Path
import cv2
import numpy as np


def is_pseudo_three_channel(img: np.ndarray, tol: float = 0.0) -> bool:
    if img is None:
        return False
    if img.ndim == 2:
        return True
    if img.ndim == 3:
        h, w, c = img.shape
        if c < 3:
            return True
        b, g, r = cv2.split(img[:, :, :3])
        if tol == 0.0:
            return np.array_equal(b, g) and np.array_equal(g, r)
        else:
            return np.allclose(b, g, atol=tol) and np.allclose(g, r, atol=tol)
    return False


def convert_to_single(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return img
    # 对于 3 或更多通道，取第一个颜色通道（B）作为灰度表示（伪三通道时任意通道相同）
    return img[:, :, 0]


def process_file(fp: Path, out_dir: Path, inplace: bool, tol: float) -> None:
    img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"{fp.name} -> 无法读取")
        return

    if not is_pseudo_three_channel(img, tol=tol):
        print(f"{fp.name} -> 非伪三通道，跳过")
        return

    single = convert_to_single(img)
    if single is None:
        print(f"{fp.name} -> 转换失败")
        return

    if inplace:
        out_path = fp
    else:
        out_path = out_dir / fp.name

    # 确保父目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(out_path), single)
    if ok:
        print(f"{fp.name} -> 已转换并保存到 {out_path}")
    else:
        print(f"{fp.name} -> 保存失败")


def main():
    parser = argparse.ArgumentParser(description="将伪三通道图片转换为单通道（灰度）图像。")
    parser.add_argument("path", nargs='?', help="图片文件或图片目录；若省略，将弹窗选择目录")
    parser.add_argument("--outdir", help="输出目录（默认: ./converted）", default="converted")
    parser.add_argument("--inplace", action='store_true', help="直接覆盖原文件（小心）")
    parser.add_argument("--tol", type=float, default=0.0, help="比较容差（默认 0）")
    parser.add_argument("--recursive", action='store_true', help="递归遍历子目录")
    args = parser.parse_args()

    selected = args.path
    if not selected:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            folder = filedialog.askdirectory(title="选择图片文件或文件夹")
            root.destroy()
        except Exception as e:
            print("无法打开选择窗口：", e)
            return

        if not folder:
            print("未选择任何路径，退出。")
            return
        p = Path(folder)
    else:
        p = Path(selected)

    if args.inplace and args.outdir:
        # 如果 inplace，忽略 outdir
        out_dir = p.parent
    else:
        out_dir = Path(args.outdir)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    if p.is_file():
        process_file(p, out_dir, args.inplace, args.tol)
        return

    if not p.exists() or not p.is_dir():
        print(f"无效路径: {p}")
        return

    if args.recursive:
        iterator = p.rglob("*")
    else:
        iterator = p.iterdir()

    for fp in sorted(iterator):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in exts:
            continue
        process_file(fp, out_dir, args.inplace, args.tol)


if __name__ == '__main__':
    main()

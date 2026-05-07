#!/usr/bin/env python3
"""
按比例随机划分可见光与红外数据集（图片+对应txt标签），并将结果复制到目标目录。
最终结构示例:
out_dir/visible/train/  (图片和对应的 .txt)
out_dir/visible/val/
out_dir/visible/test/
out_dir/infra/train/
...

用法: 运行脚本后通过 GUI 选择 `visible` 文件夹、`infra` 文件夹、以及输出根文件夹。
也可在脚本中设置随机种子以复现划分。
"""
import os
import shutil
import random
import tkinter as tk
from tkinter import filedialog, messagebox

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def find_pairs(folder):
    """在 folder 中查找图片与对应的 .txt 标注对，返回 (img_path, txt_path) 列表"""
    pairs = []
    for root, _, files in os.walk(folder):
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTS:
                img_path = os.path.join(root, f)
                txt_path = os.path.join(root, name + '.txt')
                if os.path.exists(txt_path):
                    pairs.append((img_path, txt_path))
                else:
                    # 如果没有同名 txt，则跳过
                    print(f'跳过（缺少 txt）: {img_path}')
    return sorted(pairs)


def build_basename_map(folder):
    """返回字典 basename -> (img_path, txt_path)"""
    d = {}
    for root, _, files in os.walk(folder):
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTS:
                img_path = os.path.join(root, f)
                txt_path = os.path.join(root, name + '.txt')
                if os.path.exists(txt_path):
                    d[name] = (img_path, txt_path)
    return d


def make_dirs(base_out, modality):
    out_modal = os.path.join(base_out, modality)
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(out_modal, split), exist_ok=True)
    return out_modal


def split_list(pairs, ratios=(0.7, 0.15, 0.15), seed=None):
    if seed is not None:
        random.seed(seed)
    pairs = pairs.copy()
    random.shuffle(pairs)
    n = len(pairs)
    r1 = int(n * ratios[0])
    r2 = int(n * ratios[1])
    train = pairs[:r1]
    val = pairs[r1:r1 + r2]
    test = pairs[r1 + r2:]
    return {'train': train, 'val': val, 'test': test}


def copy_pairs(splits, out_modal):
    counts = {'train': 0, 'val': 0, 'test': 0}
    for split, items in splits.items():
        dest_dir = os.path.join(out_modal, split)
        for img_path, txt_path in items:
            try:
                shutil.copy2(img_path, dest_dir)
                shutil.copy2(txt_path, dest_dir)
                counts[split] += 1
            except Exception as e:
                print('复制失败', img_path, e)
    return counts


class App:
    def __init__(self, master):
        self.master = master
        master.title('可见光/红外 数据集划分')

        self.visible_dir = tk.StringVar()
        self.infra_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.seed = tk.StringVar(value='42')

        tk.Label(master, text='Visible 文件夹:').grid(row=0, column=0, sticky='w')
        tk.Entry(master, textvariable=self.visible_dir, width=60).grid(row=0, column=1)
        tk.Button(master, text='选择', command=self.select_visible).grid(row=0, column=2)

        tk.Label(master, text='Infra 文件夹:').grid(row=1, column=0, sticky='w')
        tk.Entry(master, textvariable=self.infra_dir, width=60).grid(row=1, column=1)
        tk.Button(master, text='选择', command=self.select_infra).grid(row=1, column=2)

        tk.Label(master, text='输出根文件夹:').grid(row=2, column=0, sticky='w')
        tk.Entry(master, textvariable=self.output_dir, width=60).grid(row=2, column=1)
        tk.Button(master, text='选择', command=self.select_output).grid(row=2, column=2)

        tk.Label(master, text='随机种子 (空表示随机):').grid(row=3, column=0, sticky='w')
        tk.Entry(master, textvariable=self.seed, width=20).grid(row=3, column=1, sticky='w')

        tk.Button(master, text='开始划分', command=self.start, bg='#4CAF50', fg='white').grid(row=4, column=1, pady=10)

        self.status = tk.StringVar(value='等待操作')
        tk.Label(master, textvariable=self.status).grid(row=5, column=0, columnspan=3, sticky='w')

    def select_visible(self):
        d = filedialog.askdirectory()
        if d:
            self.visible_dir.set(d)

    def select_infra(self):
        d = filedialog.askdirectory()
        if d:
            self.infra_dir.set(d)

    def select_output(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir.set(d)

    def start(self):
        vis = self.visible_dir.get()
        infra = self.infra_dir.get()
        out = self.output_dir.get()
        if not vis or not infra or not out:
            messagebox.showwarning('提示', '请先选择可见光、红外和输出文件夹')
            return
        seed_val = None
        if self.seed.get().strip():
            try:
                seed_val = int(self.seed.get().strip())
            except ValueError:
                messagebox.showwarning('提示', '种子必须为整数或留空')
                return

        self.status.set('查找可见光文件对...')
        vis_pairs = find_pairs(vis)
        self.status.set(f'Found {len(vis_pairs)} visible pairs')

        if not vis_pairs:
            messagebox.showinfo('结果', '可见光文件夹中未找到图片与 txt 对，无法按可见光基准划分')
            return

        # 方案2: 以 visible 为基准划分，然后在 infra 中寻找相同 basename 的文件并复制
        self.status.set('按可见光基准划分...')
        # build maps
        vis_map = {os.path.splitext(os.path.basename(p[0]))[0]: p for p in vis_pairs}
        infra_map = build_basename_map(infra)

        basenames = list(vis_map.keys())
        if seed_val is not None:
            random.seed(seed_val)
        random.shuffle(basenames)
        n = len(basenames)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        train_names = basenames[:n_train]
        val_names = basenames[n_train:n_train + n_val]
        test_names = basenames[n_train + n_val:]

        # prepare output dirs
        vis_out_modal = make_dirs(out, 'visible')
        infra_out_modal = make_dirs(out, 'infra')

        vis_counts = {'train': 0, 'val': 0, 'test': 0}
        infra_counts = {'train': 0, 'val': 0, 'test': 0}

        def copy_name_list(name_list, split_name):
            for name in name_list:
                img_path, txt_path = vis_map[name]
                try:
                    shutil.copy2(img_path, os.path.join(vis_out_modal, split_name))
                    shutil.copy2(txt_path, os.path.join(vis_out_modal, split_name))
                    vis_counts[split_name] += 1
                except Exception as e:
                    print('复制 visible 失败', img_path, e)
                # copy infra counterpart if exists
                infra_item = infra_map.get(name)
                if infra_item:
                    try:
                        shutil.copy2(infra_item[0], os.path.join(infra_out_modal, split_name))
                        shutil.copy2(infra_item[1], os.path.join(infra_out_modal, split_name))
                        infra_counts[split_name] += 1
                    except Exception as e:
                        print('复制 infra 失败', infra_item[0], e)
                else:
                    print(f'警告: 在红外文件夹中未找到对应项: {name}')

        copy_name_list(train_names, 'train')
        copy_name_list(val_names, 'val')
        copy_name_list(test_names, 'test')

        msg = (
            f'完成划分:\n可见光: train={vis_counts["train"]}, val={vis_counts["val"]}, test={vis_counts["test"]}\n'
            f'红外: train={infra_counts["train"]}, val={infra_counts["val"]}, test={infra_counts["test"]}\n输出目录: {out}'
        )
        self.status.set('完成')
        messagebox.showinfo('完成', msg)


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()

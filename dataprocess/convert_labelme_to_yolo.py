#!/usr/bin/env python3
"""
LabelMe JSON -> YOLO txt 转换器（带简单可视化和 GUI）
运行后选择数据集文件夹和保存文件夹，可选预览每张图片的标注。
依赖: Pillow
"""
import os
import json
import threading
import base64
import io
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk


def collect_json_files(folder):
    exts = ['.json']
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    return sorted(files)


def load_labelme(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def infer_image_path(json_path, data):
    # 优先使用 json 内的 imagePath 或 imageData
    base = os.path.dirname(json_path)
    if 'imagePath' in data and data['imagePath']:
        candidate = os.path.join(base, data['imagePath'])
        if os.path.exists(candidate):
            return candidate
    # common image extensions
    stem = os.path.splitext(os.path.basename(json_path))[0]
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        p = os.path.join(base, stem + ext)
        if os.path.exists(p):
            return p
    return None


def load_image_from_data(data):
    img_b64 = data.get('imageData')
    if not img_b64:
        return None
    try:
        b = base64.b64decode(img_b64)
        im = Image.open(io.BytesIO(b))
        return im
    except Exception:
        return None


def shapes_to_bboxes(shapes):
    boxes = []
    for s in shapes:
        label = s.get('label', 'unknown')
        pts = s.get('points', [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        boxes.append((label, xmin, ymin, xmax, ymax))
    return boxes


def write_yolo_txt(txt_path, boxes, label2id, img_w, img_h):
    lines = []
    for label, xmin, ymin, xmax, ymax in boxes:
        cid = label2id[label]
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class ConverterApp:
    def __init__(self, master):
        self.master = master
        master.title('LabelMe -> YOLO 转换器')

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.preview = tk.BooleanVar(value=True)

        tk.Label(master, text='数据集文件夹:').grid(row=0, column=0, sticky='w')
        tk.Entry(master, textvariable=self.input_dir, width=60).grid(row=0, column=1)
        tk.Button(master, text='选择', command=self.select_input).grid(row=0, column=2)

        tk.Label(master, text='保存文件夹:').grid(row=1, column=0, sticky='w')
        tk.Entry(master, textvariable=self.output_dir, width=60).grid(row=1, column=1)
        tk.Button(master, text='选择', command=self.select_output).grid(row=1, column=2)

        tk.Checkbutton(master, text='预览每张图片', variable=self.preview).grid(row=2, column=1, sticky='w')

        tk.Button(master, text='开始转换', command=self.start_convert, bg='#4CAF50', fg='white').grid(row=3, column=1, pady=10)

        self.status = tk.StringVar(value='等待操作')
        tk.Label(master, textvariable=self.status).grid(row=4, column=0, columnspan=3, sticky='w')

    def select_input(self):
        d = filedialog.askdirectory()
        if d:
            self.input_dir.set(d)

    def select_output(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir.set(d)

    def start_convert(self):
        inp = self.input_dir.get()
        out = self.output_dir.get()
        if not inp or not out:
            messagebox.showwarning('提示', '请选择输入和输出文件夹')
            return
        t = threading.Thread(target=self.convert_all, args=(inp, out, self.preview.get()), daemon=True)
        t.start()

    def convert_all(self, inp, out, preview):
        self.status.set('收集 JSON 文件...')
        jsons = collect_json_files(inp)
        if not jsons:
            messagebox.showinfo('结果', '未找到 JSON 文件')
            self.status.set('未找到 JSON 文件')
            return
        # First pass: collect labels in encounter order
        labels = []
        label_set = set()
        samples = []  # list of tuples (json_path, image_path, boxes)
        self.status.set('解析标注，收集类别...')
        for j in jsons:
            try:
                data = load_labelme(j)
            except Exception as e:
                print('读取 JSON 失败', j, e)
                continue
            image = None
            image_path = infer_image_path(j, data)
            if image_path is None:
                # 尝试从 imageData 解码内嵌图片
                image = load_image_from_data(data)
                if image is None:
                    print('未找到对应图片或 imageData, 跳过', j)
                    continue
            shapes = data.get('shapes', [])
            boxes = shapes_to_bboxes(shapes)
            for lab, *_ in boxes:
                if lab not in label_set:
                    label_set.add(lab)
                    labels.append(lab)
            samples.append((j, image_path, boxes, image))

        # create label file
        ensure_dir(out)
        classes_path = os.path.join(out, 'classes.txt')
        with open(classes_path, 'w', encoding='utf-8') as f:
            for l in labels:
                f.write(l + '\n')

        label2id = {l: i for i, l in enumerate(labels)}

        # process each sample
        total = len(samples)
        self.status.set(f'开始转换 {total} 张图片...')
        idx = 0
        for jpath, imgpath, boxes, image in samples:
            idx += 1
            try:
                if image is not None:
                    w, h = image.size
                else:
                    with Image.open(imgpath) as im:
                        w, h = im.size
            except Exception as e:
                print('打开图片失败', imgpath, e)
                continue
            if imgpath:
                basename = os.path.splitext(os.path.basename(imgpath))[0]
            else:
                basename = os.path.splitext(os.path.basename(jpath))[0]
            txt_name = basename + '.txt'
            txt_path = os.path.join(out, txt_name)
            write_yolo_txt(txt_path, boxes, label2id, w, h)

            self.status.set(f'转换中 ({idx}/{total}): {os.path.basename(imgpath)}')

            if preview and boxes:
                # draw preview and wait for user
                try:
                    if image is not None:
                        im = image.copy()
                    else:
                        im = Image.open(imgpath)
                    draw = ImageDraw.Draw(im)
                    try:
                        font = ImageFont.truetype('arial.ttf', 14)
                    except Exception:
                        font = ImageFont.load_default()
                    for lab, xmin, ymin, xmax, ymax in boxes:
                        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
                        draw.text((xmin + 3, ymin + 3), f'{lab}', fill='red', font=font)
                    self.show_preview(im, os.path.basename(imgpath) if imgpath else os.path.basename(jpath))
                    if image is None:
                        im.close()
                except Exception as e:
                    print('预览失败', imgpath or jpath, e)

        self.status.set(f'转换完成，输出: {out}')
        messagebox.showinfo('完成', f'转换完成，YOLO txt 已保存到: {out}\n类别文件: classes.txt')

    def show_preview(self, pil_image, title='preview'):
        # modal preview window with Next button
        top = tk.Toplevel(self.master)
        top.title(title)
        # resize to fit screen if too large
        sw = top.winfo_screenwidth() - 200
        sh = top.winfo_screenheight() - 200
        w, h = pil_image.size
        scale = min(1.0, sw / w, sh / h)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            pil_image = pil_image.resize(new_size, Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(pil_image)
        lbl = tk.Label(top, image=imgtk)
        lbl.image = imgtk
        lbl.pack()
        btn = tk.Button(top, text='下一张', command=top.destroy)
        btn.pack(pady=5)
        top.transient(self.master)
        top.grab_set()
        self.master.wait_window(top)


if __name__ == '__main__':
    root = tk.Tk()
    app = ConverterApp(root)
    root.mainloop()

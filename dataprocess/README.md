LabelMe JSON -> YOLO 转换器（带可视化 GUI）

说明
- 脚本: convert_labelme_to_yolo.py
- 用途: 将 LabelMe 标注的 JSON 文件转换为 YOLO 格式的 txt（每张图片一个 txt），并生成 `classes.txt`。

依赖

安装依赖:

```bash
pip install -r requirements.txt
```

使用

```bash
python convert_labelme_to_yolo.py
```

操作步骤
- 点击“选择”选择数据集所在文件夹（包含 .json 和图片）
- 选择输出文件夹（脚本会把 .txt 和 classes.txt 保存到该目录）
- 可选“预览每张图片”以在转换时查看标注
- 点击“开始转换”开始处理

注意
- 脚本会尝试从 JSON 的 `imagePath` 字段或与 JSON 同名的常见图片扩展名找到对应图片
- shapes 中使用 `points` 字段的标注（多边形/矩形）会被转换为包围盒
- 类别按第一次遇到的顺序写入 `classes.txt`，YOLO 的 class id 从 0 开始

如需批量处理或自定义输出结构，可在脚本基础上修改保存路径逻辑。

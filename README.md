# dataset-transform
datasetprocess中的convert_labelme_to_yolo.py  实现LabelMe JSON -> YOLO 转换器（带可视化 GUI）。详情见dataprocess中的.md


split_visible_infra_dataset.py可以实现可见光与红外数据集划分，训练集，验证集，测试集比例为7：1.5：1.5


channel3-1.py 可将红外伪三通道批量转换为1通道


dataset_merge_tool.py
Dataset Tool
  Tab 1 – 两个数据集合并 → 类别筛选 → YOLO 转换
           输入: DatasetA/  DatasetB/  (各含 images/ 文件夹 + COCO JSON 文件)
           输出: output/
                  images/      合并后 + 筛选后的图像（保留原始文件名）
                  labels/      对应的 YOLO txt 标注
                  yolodata/    data.yaml  classes.txt
  Tab 2 – YOLO 数据集划分 (train / val / test)

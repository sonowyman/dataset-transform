# dataset-transform
datasetprocess中的convert_labelme_to_yolo.py  实现LabelMe JSON -> YOLO 转换器（带可视化 GUI）。详情见dataprocess中的.md
split_visible_infra_dataset.py可以实现可见光与红外数据集划分，训练集，验证集，测试集比例为7：1.5：1.5
channel3-1.py 可将红外伪三通道批量转换为1通道

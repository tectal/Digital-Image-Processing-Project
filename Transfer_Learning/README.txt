Transfer_Learning
********************************************************************************
Transfer_Learning/cub_util.py
DATA_DIR = os.path.expanduser(os.path.join("DATA_DIR", "CUB_200_2011"))，这里DATA_DIR为数据集地址。cub_util.py具体实现处理数据集，加载图片与label
--------------------------------------------------------------------------------
Transfer_Learning/transfer.py
DATA_DIR = os.path.expanduser(os.path.join("DATA_DIR", "CUB_200_2011"))，这里DATA_DIR为数据集地址。Transfer_Learning/transfer.py 读入label和图片，用数据集进行Transfer_Learning训练，用数据集测试准确率
********************************************************************************
使用说明：
运行transfer.py，先得到ResNet50 Model分类识别率；再得到Stacking Model的分类识别率；保存模型；利用模型进行预测。
运行transfer.py，得到加载CUB_200_2011数据集后200类鸟的示意图。

文件说明：
bcnn/create_h5_dataset.py 将图片添加label后保存成h5py文件供程序读取
bcnn/bcnn.py读入label和图片，用数据集进行bcnn训练，用测试集的前32张图片做验证，最后用测试集测试准确率
使用说明：
将原本数据集中的images文件夹放入bcnn/中，去掉测试集中图片，将文件夹重命名为train
将原本数据集中的images文件夹放入bcnn/中，去掉训练集中图片，将文件夹重命名为test
进入bcnn文件夹，运行python create_h5_dataset.py
下载vgg16_weights.npz放入bcnn/中，在该文件夹下运行python bcnn.py，进行最后全连接层的训练。
将load_weights中注释的语句恢复，并设置main中的isFineTune=True，再次运行python bcnn.py，进行卷积层和全连接层参数的训练。并可以得到最后的测试准确率。

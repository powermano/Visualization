# Visualization

## A tool for embedding visualization.

### 使用说明：

### 1. 通过./script/run_predict_parallel.py 生成带特征的预测结果：

```
1. cd script
2. python run_predict_parallel.py

```
可以通过save_feature = '1'， save_badcase = '1'， 来指定保存特征或者badcase。
只需要预测结果的时候不需要保存特征。

### 2. 修改config/example.yaml中input的值，具体你要可视化的结果：

```
1. vim config/example.yaml
2. python script/gen_tensorboard.py --cfg config/example.yaml

```
 其中show_image参数可以控制是否在可视化的时候显示resize的图片，如果不需要显示图片，则使用红色块代表live，黑色快代替spoof，方便区分。

### 3 download ./logs 文件到本地
```
1. cd ${path_to_your_logs_files}/logs/
2. tensorboard --log=./

```
根据控制台中的提示网址，可以查看embedding结果。

如果tensorboard不显示结果可以参考[tensorboard无法显示](https://blog.csdn.net/whitesilence/article/details/79261592)

# mmyolo deepsort 行人 车辆 跟踪 检测 计数

- 实现了 出/入 分别计数。
- 显示检测类别。
- 默认是 南/北 方向检测，若要检测不同位置和方向，可在 mmyolo_main.py 文件第13行和21行，修改2个polygon的点。
- 默认检测类别：小汽车。
- 检测类别可在 mmyolo_main.py 文件第87行修改。


## 运行环境

- python 3.6+，pip 20+
- pytorch
- mmyolo
- pip install -r requirements.txt


## 如何运行

1. 参考[mmyolo](https://github.com/open-mmlab/mmyolo) 安装对应环境（ v0.5.0 ）

   
2. 安装软件包

    ```
    $ pip install -r requirements.txt
    ```

3. 在 mmyolo_main.py 文件中第71行，设置要检测的视频文件路径

   
4. 运行程序

    ```
    python mmyolo_main.py
    ```


## 使用框架

- https://github.com/Sharpiless/Yolov5-deepsort-inference
- https://github.com/open-mmlab/mmyolo
- https://github.com/ZQPei/deep_sort_pytorch

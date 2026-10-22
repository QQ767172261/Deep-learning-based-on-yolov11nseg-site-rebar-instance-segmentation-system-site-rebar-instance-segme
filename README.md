深度学习基于yolov11n-seg工地钢筋实例分割系统 工地钢筋实例分割数据集 yolo分割模型 钢筋实例分割系统界面

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/28fc43da05844703a8b28d52aa9ffa7d.png)

工地钢筋实例分割数据集，1624张，yolo，voc，coco三种标注方式
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6b225c82132c414ba711619b861206b9.png)

数据集统计信息
====================
图像尺寸:640*640
类别数量: 1 类
训练集图像:1544; 验证集图像:60; 测试集图像:20
总图像数: 1624

类别分布:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ae2310e2a50f44d493fdce033353702a.png)
类别名称 | 包含该类别的图像数 | 总标注数
----------------------------------------
rebar | 1624 | 27679
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cc906a37159644c6b0cbf22fc4fd805d.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/796e2e9642ec4469a7020532db233f62.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8e9e5f4133704134a06e3519a0fc6aab.png)


模型代码：模型训练使用yolov11n-seg训练，80个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c453ecd260748a7b6c37d1dfb8e933c.png)

qt界面：运行界面采用pyqt5编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/689147eefcc443719d72a91d2276cbe1.png)


# 基于YOLOv11 钢筋实例分割系统完整简易代码
## 1、数据集yaml配置文件 `rebar_data.yaml`
```yaml
path: ./rebar_dataset
train: train/images
val: val/images
test: test/images

names:
  0: rebar
```

## 2、模型训练代码 train.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLOv11实例分割模型
    model = YOLO("yolo11n-seg.yaml").load("yolo11n-seg.pt")
    results = model.train(
        data="rebar_data.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        val=True,
        save=True
    )
```

## 3、单图分割推理代码 predict.py
```python
from ultralytics import YOLO

model = YOLO("runs/segment/train/weights/best.pt")

def seg_rebar(img_path):
    res = model(img_path)
    res[0].show()
    return res

if __name__ == "__main__":
    seg_rebar("rebar_test.jpg")
```

## 4、PyQt5可视化界面简易代码 main_gui.py（钢筋分割系统）
```python
import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QPushButton,QLabel,QFileDialog
from PyQt5.QtGui import QPixmap
from ultralytics import YOLO

class RebarSegWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLO11的钢筋实例分割系统")
        self.resize(1400,800)
        self.model = YOLO("runs/segment/train/weights/best.pt")
        #按钮
        self.btn_img = QPushButton("选择图片",self)
        self.btn_img.clicked.connect(self.open_img)
        self.label_raw = QLabel("原图",self)
        self.label_result = QLabel("分割结果",self)

    def open_img(self):
        path,_ = QFileDialog.getOpenFileName()
        if path:
            pix = QPixmap(path)
            self.label_raw.setPixmap(pix)
            #推理
            res = self.model(path,save=True)
            res_path = res[0].save_dir + "/image0.jpg"
            self.label_result.setPixmap(QPixmap(res_path))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = RebarSegWindow()
    win.show()
    sys.exit(app.exec_())
```

## 环境安装
```bash
pip install ultralytics pyqt5 opencv-python
```
功能：图片钢筋实例分割，识别rebar钢筋目标，输出分割掩码，支持GUI图片检测；可扩展视频、摄像头实时检测。



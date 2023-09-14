# 3d_ex\implicit_fem
## 使用方法
* 安装环境
```
pip install taichi==1.4.1
python3 -m pip install -U taichi meshtaichi_patcher
```
* 连接好力反馈设备Touch（使用默认名称）
* 直接运行main.py
## 基于python的显式/隐式有限元
* 参考taichi的ti example fem的算法
* 弹性体有限元仿真
* 加入了力反馈设备交互
* 加入了碰撞检测和碰撞响应

## 效果预览
![image](images/整体.mp4)

## 现存的问题
碰撞响应算法十分基础，有待改进

## 后续计划
* 1、加入场景
* 2、更新本构模型和力响应模型

## 注意
* 1、环境问题：python3.8+taichi1.4.1版本运行最佳，在新版本的taichi环境中运行会掉帧，原因不明
* 2、需要搭配力反馈设备touch使用，也可以在代码main中将力反馈相关部分注释掉，只运行仿真部分


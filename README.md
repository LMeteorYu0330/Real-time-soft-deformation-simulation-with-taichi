# 基于taichi和FEM的实时虚拟手术交互仿真
## 使用方法
* 安装环境
```
pip install taichi==1.4.1
python3 -m pip install -U taichi meshtaichi_patcher
```
* 连接好力反馈设备Touch（使用默认名称）
* 直接运行main.py
## 基于python-taichi的隐式有限元弹性器官仿真
* 参考taichi example中的implicit fem的算法
* 弹性器官有限元仿真
* 加入力反馈设备交互
* 使用虚位代理算法解决穿模问题

## 效果预览
![image]([https://github.com/LMeteorYu0330/Real-time-surgery-simulation-with-taichi/blob/master/images/2023-10-14.mp4](https://github.com/LMeteorYu0330/Real-time-surgery-simulation-with-taichi/raw/master/images/2023-10-14.mp4))

## 现存的问题
碰撞响应算法十分基础，有待改进

## 后续计划
* 1、加入手术场景
* 2、更新本构模型、数值求解器和力响应模型
* 3、完善readme

## 注意
* 1、环境问题：python3.8+taichi1.4.1版本运行最佳，在新版本的taichi环境中运行会掉帧，原因不明，其他python版本未测试
* 2、需要搭配力反馈设备touch使用，也可以在代码main中将力反馈相关部分注释掉，只运行仿真部分
* 3、pyhaptics.pyd文件是编译的力反馈函数库，仅适用于windows环境，放在anaconda对应虚拟环境的Lib\site-package文件夹下可以解析显示函数接口，直接放在脚本同一目录下仅能使用


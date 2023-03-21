# 3d_ex\implicit_fem

## 基于python的显式/隐式有限元
* 参考taichi的ti example fem的算法  
通过将mesh传入explicit类或者implicit类来选择显式或者隐式  
类的定义在fem_class.py中  

![img] (https://github.com/LMeteorYu0330/3d_implicit_fem/blob/master/images/1.gif)
## 现存的问题
无法修改本构模型，修改本构模型导致模型炸裂，原因还在查找中

## 后续计划
* 1、加入自碰撞和碰撞检测。
* 2、加入力反馈设备的交互。  
力反馈设备的python交互已经实现

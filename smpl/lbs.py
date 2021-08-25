import numpy as np

from .utils import *


def rigid_transformation(pose, joints, kintree_table):
    # 首先计算根结点 (0) 的世界坐标变换, 或者说是根结点相对世界坐标系的位姿
    T = np.zeros([4, 4])
    T[:3, :3] = rodrigues(pose[0])  # 轴角转换到旋转矩阵，相对世界坐标
    T[:3, 3] = joints[0]  # 根结点在世界坐标系下的位置
    T[3, 3] = 1  # 齐次矩阵，1

    Ts = np.zeros([24, 4, 4])
    Ts[0] = T

    # 计算子节点 (1~24) 的世界坐标变换
    for i in range(1, 24):
        # 首先计算子节点相对父节点坐标系的位姿 [R|t]
        T = np.zeros([4, 4])
        T[3, 3] = 1
        # 计算子节点相对父节点的旋转矩阵 R
        T[:3, :3] = rodrigues(pose[i])
        # 计算子节点相对父节点的偏移量 t
        T[:3, 3] = joints[i] - joints[kintree_table[i]]
        # 然后计算子节点相对世界坐标系的位姿
        Ts[i] = np.matmul(Ts[kintree_table[i]], T)  # 乘上其父节点的变换矩阵

    # 计算每个子节点相对 T-pose 时的位姿矩阵
    for i in range(24):
        R = Ts[i][:3, :3]
        t = Ts[i][:3, 3] - R.dot(joints[i])  # 子节点相对T-pose的偏移
        Ts[i][:3, 3] = t

    return Ts


def blend_skin(pose, v_posed, joints, weights, kintree_table):
    # 计算每个关节点相对于默认姿态的旋转向量和平移向量
    Ts = rigid_transformation(pose, joints, kintree_table)

    # 开始蒙皮操作，LBS 过程
    v_posed_homo = np.hstack([v_posed, np.ones([v_posed.shape[0], 1])])
    vertices_homo = np.matmul(weights.dot(Ts.reshape([24, 16])).reshape([-1, 4, 4]), v_posed_homo.reshape([-1, 4, 1]))
    vertices = vertices_homo[:, :3, 0]  # 由于是齐次矩阵，取前3列
    return vertices

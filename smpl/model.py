import numpy as np
import pickle

from .lbs import blend_skin
from .utils import rodrigues, batch_eye


class SMPL:
    def __init__(self, model_file):
        model = pickle.load(open(model_file, "rb"), encoding='iso-8859-1')
        self.v_template = model["v_template"]  # 平均body
        self.weights = model["weights"]  # joints对顶点的权重
        self.posedirs = model["posedirs"]  # pose对顶点的影响
        self.shapedirs = model["shapedirs"].x  # shape对顶点的影响
        self.kintree_table = model["kintree_table"]  # 运动学树
        self.J_regressor = model["J_regressor"]  # joints回归器，根据顶点坐标估算joints坐标
        self.faces = model["f"] + 1  # faces的索引，索引从1开始

    def __call__(self, shape=None, pose=None, trans=None):
        """
        SMPL算法
        :param shape: 10
        :param pose: 24x3
        :param trans: 3
        """
        shape = np.zeros(10) if shape is None else shape
        pose = np.zeros(24, 3) if pose is None else pose
        trans = np.zeros(3) if trans is None else trans

        v_shaped = self.v_template + self.shapedirs.dot(shape)  # shape->顶点
        v_posed = v_shaped + self.posedirs.dot((rodrigues(pose[1:]) - batch_eye(3)).ravel())  # pose->顶点

        joints = self.J_regressor.dot(v_shaped)  # 根据顶点估算的joints坐标
        vertices = blend_skin(pose, v_posed, joints, self.weights, self.kintree_table)  # 姿态影响的顶点变化
        vertices = vertices + trans.reshape((1, 3))

        joints = self.J_regressor.dot(vertices)  # 计算 pose 下 joints 位置
        return vertices, joints

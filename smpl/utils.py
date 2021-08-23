import numpy as np
import cv2


def rodrigues(x):
    if x.ndim == 1:
        return cv2.Rodrigues(x)[0]
    else:
        return np.stack([rodrigues(i) for i in x])


def with_zeros(x):
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))


def pack(x):
    return np.hstack([np.zeros((4, 3)), x.reshape((4, 1))])


def batch_eye(n):
    return np.eye(n)[np.newaxis, ...]


def write_obj(file, vertices, faces):
    with open(file, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

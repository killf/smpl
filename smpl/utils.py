import numpy as np


def rodrigues_cv(x):
    import cv2
    return cv2.Rodrigues(x)[0]


def rodrigues_np(r):
    theta = np.linalg.norm(r)
    x, y, z = r / theta
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[x * x * (1 - c) + c, x * y * (1 - c) + z * s, x * z * (1 - c) - y * s],
                     [x * y * (1 - c) - z * s, y * y * (1 - c) + c, y * z * (1 - c) + x * s],
                     [x * z * (1 - c) + y * s, y * z * (1 - c) - x * s, z * z * (1 - c) + c]]).T


def rodrigues(x):
    if x.ndim == 1:
        return rodrigues_np(x)
    else:
        return np.stack([rodrigues(i) for i in x])


def batch_eye(n):
    return np.eye(n)[np.newaxis, ...]


def write_obj(file, vertices, faces):
    with open(file, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

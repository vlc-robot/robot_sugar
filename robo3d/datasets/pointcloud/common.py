import numpy as np
import random

from robo3d.utils.img_randaug import (
    AutoContrast, Brightness, Color, Contrast, Equalize, Posterize, Sharpness
)


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = [
            (AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Equalize, None, None),
            (Posterize, 4, 4),
            (Sharpness, 1.8, 0.1)
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        return img


def normalize_pc(pc, return_params=False):
    # Normalize the point cloud to [-1, 1]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / m
    if return_params:
        return pc, (centroid, m)
    return pc

def random_scale_pc(pc, scale_low=0.8, scale_high=1.25):
    # Randomly scale the point cloud.
    scale = np.random.uniform(scale_low, scale_high)
    pc = pc * scale
    return pc

def shift_pc(pc, shift_range=0.1):
    # Randomly shift point cloud.
    shift = np.random.uniform(-shift_range, shift_range, size=[3])
    pc = pc + shift
    return pc

def rotate_perturbation_pc(pc, angle_sigma=0.06, angle_clip=0.18):
    # Randomly perturb the point cloud by small rotations (unit: radius)
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    cosval, sinval = np.cos(angles), np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cosval[0], -sinval[0]], [0, sinval[0], cosval[0]]])
    Ry = np.array([[cosval[1], 0, sinval[1]], [0, 1, 0], [-sinval[1], 0, cosval[1]]])
    Rz = np.array([[cosval[2], -sinval[2], 0], [sinval[2], cosval[2], 0], [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pc = np.dot(pc, np.transpose(R))
    return pc

def random_rotate_z(pc):
    # Randomly rotate around z-axis
    angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(angle), np.sin(angle)
    R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    return np.dot(pc, np.transpose(R))

def random_rotate_xyz(pc):
    # Randomly rotate around x, y, z axis
    angles = np.random.uniform(size=[3]) * 2 * np.pi
    cosval, sinval = np.cos(angles), np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cosval[0], -sinval[0]], [0, sinval[0], cosval[0]]])
    Ry = np.array([[cosval[1], 0, sinval[1]], [0, 1, 0], [-sinval[1], 0, cosval[1]]])
    Rz = np.array([[cosval[2], -sinval[2], 0], [sinval[2], cosval[2], 0], [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pc = np.dot(pc, np.transpose(R))
    return pc

def augment_pc(pc):
    pc = random_scale_pc(pc)
    pc = shift_pc(pc)
    pc = rotate_perturbation_pc(pc)
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

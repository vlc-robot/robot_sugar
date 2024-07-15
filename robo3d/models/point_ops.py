import torch
import torch.nn as nn

from pointnet2_ops import pointnet2_utils

from robo3d.utils.knn import knn_point


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


class PointGroup(nn.Module):  # FPS + KNN
    def __init__(self, num_groups, group_size, knn=True, radius=None):
        '''
        Whether use knn or radius to get the neighborhood
        '''
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.knn = knn
        self.radius = radius

    def forward(self, pc_fts):
        '''
            input: B N 3+x
            ---------------------------
            output: B G M 3+x
            center : B G 3
        '''
        batch_size, num_points, _ = pc_fts.shape
        xyz = pc_fts[..., :3].contiguous()

        # fps the centers out
        centers = fps(xyz, self.num_groups)  # B G 3

        if self.knn: # knn to get the neighborhood, shape=(batch, num_groups, group_size)
            idx = knn_point(self.group_size, xyz, centers)
        else:       # use radius to get the neighborhood (ball query), shape=(batch, num_groups, group_size)
            idx = pointnet2_utils.ball_query(self.radius, self.group_size, xyz, centers)
        assert idx.size(1) == self.num_groups
        assert idx.size(2) == self.group_size
        # shape=(batch, 1, 1)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhoods = pc_fts.view(batch_size * num_points, -1)[idx, :]
        neighborhoods = neighborhoods.view(batch_size, self.num_groups, self.group_size, -1).contiguous()
        # normalize
        neighborhoods[..., :3] = neighborhoods[..., :3] - centers.unsqueeze(2)
        return neighborhoods, centers


def three_interpolate_feature(unknown_xyz, known_xyz, known_feat):
    """
    input: unknown_xyz: (B, n, 3), known_xyz: (B, m, 3), known_feat: (B, m, c)
    output: (B, n, c)
    """
    # print(unknown_xyz.size(), unknown_xyz.dtype, unknown_xyz.device)
    # print(known_xyz.size(), known_xyz.dtype, known_xyz.device)
    # print(known_feat.size(), known_feat.dtype, known_feat.device)
    dist, idx = pointnet2_utils.three_nn(unknown_xyz, known_xyz)
    # shape=(B, n, 3)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet2_utils.three_interpolate(
        known_feat.transpose(1, 2).contiguous(), idx, weight
    )
    interpolated_feats = interpolated_feats.transpose(1, 2)
    return interpolated_feats
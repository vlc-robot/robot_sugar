import numpy as np


def get_intrinsic_matrix(im_width, im_height, fov_rad=0.691111015275598):
    res = np.array([im_width, im_height])
    pp_offsets = res / 2
    ratio = res[0] / res[1]
    # pa_x = pa_y = math.radians(self.get_perspective_angle())
    # pa_x, pa_y = 0.691111015275598, 0.691111015275598
    pa_x = pa_y = fov_rad
    if ratio > 1:
        pa_y = 2 * np.arctan(np.tan(pa_y / 2) / ratio)
    elif ratio < 1:
        pa_x = 2 * np.arctan(np.tan(pa_x / 2) * ratio)
    persp_angles = np.array([pa_x, pa_y])
    focal_lengths = -res / (2 * np.tan(persp_angles / 2))
    return np.array(
        [[focal_lengths[0], 0.,               pp_offsets[0]],
            [0.,               focal_lengths[1], pp_offsets[1]],
            [0.,               0.,               1.]])

def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
    return uniform_pixel_coords

def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector,
                      (h, w, -1))

def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo

def pointcloud_from_depth_and_camera_params(
    depth: np.ndarray, cam2world_matrix: np.ndarray,
    intrinsics: np.ndarray
) -> np.ndarray:
    """Converts depth (in meters) to point cloud in world frame.
    :return: A numpy array of size (width, height, 3)
    """
    HEIGHT, WIDTH = depth.shape[:2]
    upc = _create_uniform_pixel_coords_image(depth.shape)
    pc = upc * np.expand_dims(depth, -1)    # (HEIGHT, WIDTH, 3)

    pc = pc.reshape(-1, 3).transpose()
    pc = np.matmul(np.linalg.inv(intrinsics), pc) # (3, npoints)
    pc = np.concatenate([pc, np.ones((1, pc.shape[1]))], axis=0)
    world_coords = np.matmul(cam2world_matrix, pc)[:3].transpose()
    
    world_coords = world_coords.reshape(HEIGHT, WIDTH, 3)
    return world_coords


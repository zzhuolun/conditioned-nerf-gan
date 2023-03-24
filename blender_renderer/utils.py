# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
import numpy as np
import math
import bpy
import bpy_extras
from mathutils import Matrix
from mathutils import Vector
from typing import Union

# import matplotlib.pyplot as plt
# ---------------------------------------------------------------
# 3x4 P matrix from Blender camera
# ---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camd.sensor_fit == "VERTICAL":
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # goes here
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew, u_0, 0), (0, alpha_v, v_0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    )

    normalized_K = np.array(
        [
            [alpha_u / resolution_x_in_px * 2, skew, 0, 0],
            [0, alpha_v / resolution_y_in_px * 2, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return normalized_K


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location
    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],),
            (0, 0, 0, 1),
        )
    )
    return np.asarray(RT)


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K @ RT, K, RT


def sample_cam(
    num_views: int, cam_r_start: float = 0.7, cam_r_end: float = 1.5
) -> np.ndarray:
    """Spherical uniform sampling of camera origins

    Args:
        num_views (int): number of camera orgins to sample
        cam_r_start (float, optional): Closest camera distance to origin. Defaults to 0.7.
        cam_r_end (float, optional): Farest camera distance to origin. Defaults to 1.5.
    """
    # sample theta by inverse transform sampling
    theta = np.arccos(1 - np.random.rand(num_views))
    theta = np.clip(theta, 1e-5, np.pi - 1e-5)
    phi = np.random.rand(num_views) * np.pi * 2
    r = np.random.rand(num_views) * (cam_r_end - cam_r_start) + cam_r_start
    cam_origin = np.zeros((num_views, 3))
    cam_origin[:, 0] = r * np.sin(theta) * np.cos(phi)
    cam_origin[:, 1] = r * np.sin(theta) * np.sin(phi)
    cam_origin[:, 2] = r * np.cos(theta)
    return cam_origin


# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vector lengths."""
    return vectors / (np.linalg.norm(vectors))


def cam2world_from_cam_origin(origin: np.ndarray) -> np.ndarray:
    """Takes in the camera origin and returns a cam2world matrix."""
    forward_vector = normalize(-origin)
    up_vector = np.array([0, 0, 1], dtype=float)

    left_vector = normalize(np.cross(up_vector, forward_vector))

    up_vector = normalize(np.cross(forward_vector, left_vector))

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.stack(
        (-left_vector, -up_vector, forward_vector), axis=-1
    )

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world

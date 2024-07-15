from typing import List, Dict, Optional, Sequence, Tuple, TypedDict, Union, Any
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import Observation
from rlbench.demo import Demo 
from rlbench.backend.utils import task_file_to_task_class
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.backend.utils import rgb_handles_to_mask

from .actioner import BaseActioner
from .coord_transforms import convert_gripper_pose_world_to_image, quat_to_euler, euler_to_quat
# from .visualize import plot_attention
from .recorder import TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion


CAMERA_ATTR = {
    "front": "_cam_front",
    "wrist": "_cam_wrist",
    "left_shoulder": "_cam_over_shoulder_left",
    "right_shoulder": "_cam_over_shoulder_right"
}

class Mover:
    def __init__(self, task: TaskEnvironment, disabled: bool = False, max_tries: int = 1):
        self._task = task
        self._last_action: Optional[np.ndarray] = None
        self._step_id = 0
        self._max_tries = max_tries
        self._disabled = disabled

    def __call__(self, action: np.ndarray):
        action = action.copy()

        if self._disabled:
            return self._task.step(action)

        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()

        try_id = 0
        obs = None
        terminate = None
        reward = 0

        for try_id in range(self._max_tries):
            obs, reward, terminate = self._task.step(action)

            pos = obs.gripper_pose[:3]
            rot = obs.gripper_pose[3:7]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())  # type: ignore
            dist_rot = np.sqrt(np.square(target[3:7] - rot).sum())  # type: ignore
            # criteria = (dist_pos < 5e-2, dist_rot < 1e-1, (gripper > 0.5) == (target_gripper > 0.5))
            criteria = (dist_pos < 5e-2,)

            if all(criteria) or reward == 1:
                break

            print(
                f"Too far away (pos: {dist_pos:.3f}, rot: {dist_rot:.3f}, step: {self._step_id})... Retrying..."
            )

        # we execute the gripper action after re-tries
        action = target
        if (
            not reward
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            obs, reward, terminate = self._task.step(action)

        if try_id == self._max_tries - 1:
            print(f"Failure after {self._max_tries} tries")

        self._step_id += 1
        self._last_action = action.copy()

        other_obs = []

        return obs, reward, terminate, other_obs


class RLBenchEnv(object):
    def __init__(
        self,
        data_path='',
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        apply_mask=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        gripper_pose=None,
        image_size=[128, 128],
        cam_rand_factor=0.0,
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.apply_mask = apply_mask
        self.gripper_pose = gripper_pose
        self.image_size = image_size
        self.cam_rand_factor=cam_rand_factor

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            apply_rgb, apply_depth, apply_pc, apply_mask, apply_cameras, image_size,
        )
        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
            gripper_action_mode=Discrete(),
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config, headless=headless
        )

        self.cam_info = None

    def get_observation(self, obs: Observation):
        """Fetch the desired state based on the provided demo.
            :param obs: incoming obs
            :return: required observation (rgb, pc, gripper state)
        """

        # fetch state: (#cameras, H, W, C)
        state_dict = {"rgb": [], "depth": [], "pc": []}
        if self.apply_mask:
            state_dict["gt_mask"] = []
            
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

            if self.apply_mask:
                mask = getattr(obs, "{}_mask".format(cam))
                # state_dict["gt_mask"] += [rgb_handles_to_mask((mask * 255).astype(np.uint8))]
                if mask.ndim == 2:
                    state_dict["gt_mask"] += [mask]
                else:
                    state_dict["gt_mask"] += [rgb_handles_to_mask(mask).astype(np.uint8)]
        
        # fetch gripper state (3+4+1, )
        gripper = np.concatenate([obs.gripper_pose, [obs.gripper_open]]).astype(np.float32)
        state_dict['gripper'] = gripper

        if self.gripper_pose:
            gripper_imgs = np.zeros(
                (len(self.apply_cameras), 1, 128, 128), dtype=np.float32
            )
            for i, cam in enumerate(self.apply_cameras):
                u, v = convert_gripper_pose_world_to_image(obs, cam)
                if u > 0 and u < 128 and v > 0 and v < 128:
                    gripper_imgs[i, 0, v, u] = 1
            state_dict["gripper_imgs"] = gripper_imgs

        return state_dict

    def get_demo(self, task_name, variation, episode_index, load_images=True):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
            random_selection=False,
            load_images=load_images
        )
        return demos[0]

    def evaluate(
        self, taskvar_id, task_str, max_episodes, variation, num_demos, log_dir, 
        actioner: BaseActioner, max_tries: int = 1, 
        demos: Optional[List[Demo]] = None, demo_keys: List = None,
        save_attn: bool = False, save_image: bool = False,
        record_video: bool = False, include_robot_cameras: bool = True, video_rotate_cam: bool = False,
        video_resolution: int = 480,
        return_detail_results: bool = False,
        skip_demos: int = 0,
    ):
        """
        Evaluate the policy network on the desired demo or test environments
            :param task_type: type of task to evaluate
            :param max_episodes: maximum episodes to finish a task
            :param num_demos: number of test demos for evaluation
            :param model: the policy network
            :param demos: whether to use the saved demos
            :return: success rate
        """

        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task.set_variation(variation)  # type: ignore

        if skip_demos > 0:
            for k in range(skip_demos):
                task.reset()

        if record_video:
            # Add a global camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_resolution = [video_resolution, video_resolution]
            cam = VisionSensor.create(cam_resolution)
            cam.set_pose(cam_placeholder.get_pose())
            cam.set_parent(cam_placeholder)

            if video_rotate_cam:
                global_cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)
            else:
                global_cam_motion = StaticCameraMotion(cam)

            cams_motion = {"global": global_cam_motion}

            if include_robot_cameras:
                # Env cameras
                cam_left = VisionSensor.create(cam_resolution)
                cam_right = VisionSensor.create(cam_resolution)
                cam_wrist = VisionSensor.create(cam_resolution)

                left_cam_motion = AttachedCameraMotion(cam_left, task._scene._cam_over_shoulder_left)
                right_cam_motion = AttachedCameraMotion(cam_right, task._scene._cam_over_shoulder_right)
                wrist_cam_motion = AttachedCameraMotion(cam_wrist, task._scene._cam_wrist)

                cams_motion["left"] = left_cam_motion
                cams_motion["right"] = right_cam_motion
                cams_motion["wrist"] = wrist_cam_motion
            tr = TaskRecorder(cams_motion, fps=30)
            task._scene.register_step_callback(tr.take_snap)

            video_log_dir = log_dir / 'videos' / f'{task_str}_{taskvar_id}'
            os.makedirs(str(video_log_dir), exist_ok=True)

        success_rate = 0.0

        if demos is None:
            fetch_list = [i for i in range(num_demos)]
        else:
            fetch_list = demos

        if demo_keys is None:
            demo_keys = [f'episode{i}' for i in range(num_demos)]

        if return_detail_results:
            detail_results = {}

        with torch.no_grad():
            cur_demo_id = 0
            for demo_id, demo in tqdm(zip(demo_keys, fetch_list)):
                # reset a new demo or a defined demo in the demo list
                if isinstance(demo, int):
                    instructions, obs = task.reset()
                else:
                    print("Resetting to demo", demo_id)
                    instructions, obs = task.reset_to_demo(demo)  # type: ignore

                if self.cam_rand_factor:
                    cams = {}
                    for cam_name in self.apply_cameras:
                        if cam_name != "wrist":
                            cams[cam_name] = getattr(task._scene, CAMERA_ATTR[cam_name])
                    
                    if self.cam_info is None:
                        self.cam_info = {}
                        for cam_name, cam in cams.items():
                            self.cam_info[cam_name] = cam.get_pose()
                        
                    for cam_name, cam in cams.items():
                        # pos +/- 1 cm
                        cam_pos_range = self.cam_rand_factor * 0.01
                        # euler angles +/- 0.05 rad = 2.87 deg
                        cam_rot_range = self.cam_rand_factor * 0.05

                        delta_pos = np.random.uniform(low=-cam_pos_range, high=cam_pos_range, size=3)
                        delta_rot = np.random.uniform(low=-cam_rot_range, high=cam_rot_range, size=3)
                        orig_pose = self.cam_info[cam_name]

                        orig_pos = orig_pose[:3]
                        orig_quat = orig_pose[3:]
                        orig_rot = quat_to_euler(orig_quat, False)
                        
                        new_pos = orig_pos + delta_pos
                        new_rot = orig_rot + delta_rot
                        new_quat = euler_to_quat(new_rot, False)

                        new_pose = np.concatenate([new_pos, new_quat])

                        cam.set_pose(new_pose)

                actioner.reset(task_str, variation, instructions, demo_id)

                move = Mover(task, max_tries=max_tries)
                reward = None

                if log_dir is not None and (save_attn or save_image):
                    ep_dir = log_dir / task_str / demo_id
                    ep_dir.mkdir(exist_ok=True, parents=True)

                for step_id in range(max_episodes):
                    # fetch the current observation, and predict one action
                    obs_state_dict = self.get_observation(obs)  # type: ignore

                    if log_dir is not None and save_image:
                        for cam_id, img_by_cam in enumerate(obs_state_dict['rgb']):
                            cam_dir = ep_dir / f'camera_{cam_id}'
                            cam_dir.mkdir(exist_ok=True, parents=True)
                            Image.fromarray(img_by_cam).save(cam_dir / f"{step_id}.png")

                    output = actioner.predict(taskvar_id, step_id, obs_state_dict, episode_id=demo_id)
                    action = output["action"]

                    if action is None:
                        break

                    # TODO
                    if log_dir is not None and save_attn and output["action"] is not None:
                        ep_dir = log_dir / f"episode{demo_id}"
                        # fig = plot_attention(
                        #     output["attention"],
                        #     obs_state_dict['rgb'],
                        #     obs_state_dict['pc'],
                        #     ep_dir / f"attn_{step_id}.png",
                        # )

                    # update the observation based on the predicted action
                    try:
                        obs, reward, terminate, _ = move(action)

                        if reward == 1:
                            success_rate += 1 / num_demos
                            break
                        if terminate:
                            print("The episode has terminated!")
                    except (IKError, ConfigurationPathError, InvalidActionError) as e:
                        print(task_str, demo_id, step_id, e)
                        reward = 0
                        break
                
                cur_demo_id += 1
                print(
                    task_str, "Variation", variation, "Demo", demo_id, 'Step', step_id+1,
                    "Reward", reward, "Accumulated SR: %.2f" % (success_rate * 100), 
                    'Estimated SR: %.2f' % (success_rate * num_demos / cur_demo_id * 100)
                )

                if return_detail_results:
                    detail_results[demo_id] = reward

                if record_video: # and reward < 1:
                    tr.save(str(video_log_dir / f"{demo_id}_SR{reward}"))

        self.env.shutdown()

        if return_detail_results:
            return success_rate, detail_results
        return success_rate

    def create_obs_config(
        self, apply_rgb, apply_depth, apply_pc, apply_mask, apply_cameras, image_size, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        unused_cams = CameraConfig()
        unused_cams.set_all(False)
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=apply_mask,
            render_mode=RenderMode.OPENGL,
            image_size=image_size,
            **kwargs,
        )

        camera_names = apply_cameras
        kwargs = {}
        for n in camera_names:
            kwargs[n] = used_cams

        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,
        )
        obs_config.left_shoulder_camera.masks_as_one_channel = False
        obs_config.right_shoulder_camera.masks_as_one_channel = False
        obs_config.overhead_camera.masks_as_one_channel = False
        obs_config.wrist_camera.masks_as_one_channel = False
        obs_config.front_camera.masks_as_one_channel = False

        return obs_config


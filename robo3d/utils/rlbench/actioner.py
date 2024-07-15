from typing import List, Dict, Optional, Sequence, Tuple, TypedDict, Union, Any

class BaseActioner:

    def reset(self, task_str, variation, instructions, demo_id):
        self.task_str = task_str
        self.variation = variation
        self.instructions = instructions
        self.demo_id = demo_id

        self.step_id = 0
        self.state_dict = {}
        self.history_obs = {}

    def predict(self, *args, **kwargs):
        raise NotImplementedError('implete predict function')

# class Actioner:
#     def __init__(
#         self,
#         record_actions: bool = False,
#         replay_actions: Optional[Path] = None,
#         ground_truth_rotation: bool = False,
#         ground_truth_position: bool = False,
#         ground_truth_gripper: bool = False,
#         model = None,  # model includes t and z
#         model_rotation: Optional[nn.Module] = None,
#         model_position: Optional[nn.Module] = None,
#         model_gripper: Optional[nn.Module] = None,
#         apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
#         instructions: Optional[Dict] = None,
#         taskvar_token: bool = False,
#     ):
#         self._record_actions = record_actions
#         self._replay_actions = replay_actions
#         self._ground_truth_rotation = ground_truth_rotation
#         self._ground_truth_position = ground_truth_position
#         self._ground_truth_gripper = ground_truth_gripper
#         assert (model is not None) ^ (
#             model_rotation is not None
#             and model_position is not None
#             and model_gripper is not None
#         )
#         self._model = model
#         self._model_rotation = model_rotation
#         self._model_position = model_position
#         self._model_gripper = model_gripper
#         self._apply_cameras = apply_cameras
#         self._instructions = instructions
#         self._taskvar_token = taskvar_token

#         if self._taskvar_token:
#             with open(Path(__file__).parent / "tasks.csv", "r") as fid:
#                 self._tasks = [l.strip() for l in fid.readlines()]

#         self._actions: Dict = {}
#         self._instr: Optional[torch.Tensor] = None
#         self._taskvar: Optional[torch.Tensor] = None
#         self._task: Optional[str] = None

#     def load_episode(
#         self, task_str: str, variation: int, demo_id: int, demo: Union[Demo, int]
#     ):
#         self._task = task_str

#         if self._instructions is None:
#             self._instr = None
#         else:
#             instructions = list(self._instructions[task_str][variation])
#             self._instr = random.choice(instructions).unsqueeze(0)

#         if self._taskvar_token:
#             task_id = self._tasks.index(task_str)
#             self._taskvar = torch.Tensor([[task_id, variation]]).unsqueeze(0)
#             print(self._taskvar)

#         if self._replay_actions is not None:
#             self._actions = torch.load(
#                 self._replay_actions / f"episode{demo_id}" / "actions.pth"
#             )
#         elif (
#             self._ground_truth_rotation
#             or self._ground_truth_position
#             or self._ground_truth_gripper
#         ):
#             if isinstance(demo, int):
#                 raise NotImplementedError()
#             action_ls = self.get_action_from_demo(demo)
#             self._actions = dict(enumerate(action_ls))
#         else:
#             self._actions = {}

#     def get_action_from_demo(self, demo: Demo):
#         """
#         Fetch the desired state and action based on the provided demo.
#             :param demo: fetch each demo and save key-point observations
#             :param normalise_rgb: normalise rgb to (-1, 1)
#             :return: a list of obs and action
#         """
#         key_frame = keypoint_discovery(demo)
#         action_ls = []
#         for f in key_frame:
#             obs = demo[f]
#             action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])  # type: ignore
#             action = torch.from_numpy(action_np)
#             action_ls.append(action.unsqueeze(0))
#         return action_ls

#     def predict(
#         self, step_id: int, rgbs: torch.Tensor, pcds: torch.Tensor, gripper: torch.Tensor
#     ) -> Dict[str, Any]:
#         padding_mask = torch.ones_like(rgbs[:, :, 0, 0, 0, 0]).bool()
#         output: Dict[str, Any] = {"action": None, "attention": {}}

#         if self._instr is not None:
#             self._instr = self._instr.to(rgbs.device)

#         if self._taskvar is not None:
#             self._taskvar = self._taskvar.to(rgbs.device)

#         if self._replay_actions:
#             if step_id not in self._actions:
#                 print(f"Step {step_id} is not prerecorded!")
#                 return output
#             action = self._actions[step_id]
#         elif self._model is None:
#             action = torch.Tensor([]).to(self.device)
#             keys = ("position", "rotation", "gripper")
#             slices = (slice(0, 3), slice(3, 7), slice(7, 8))
#             for key, slice_ in zip(keys, slices):
#                 model = getattr(self, f"_model_{key}")
#                 t = model["t"][self._task][: step_id + 1].unsqueeze(0)
#                 z = model["z"][self._task][: step_id + 1].unsqueeze(0)
#                 pred = model["model"](
#                     rgbs, pcds, padding_mask, t, z, self._instr, gripper, self._taskvar
#                 )
#                 action_key = model["model"].compute_action(pred)
#                 action = torch.cat([action, action_key[slice_]])
#             output["action"] = action
#         else:
#             if self._task is None:
#                 raise ValueError()
#             t = self._model["t"][self._task][: step_id + 1].unsqueeze(0)
#             z = self._model["z"][self._task][: step_id + 1].unsqueeze(0)
#             pred = self._model["model"](
#                 rgbs, pcds, padding_mask, t, z, self._instr, gripper, self._taskvar
#             )
#             output["action"] = self._model["model"].compute_action(pred)  # type: ignore
#             output["attention"] = pred["attention"]

#         if self._ground_truth_rotation:
#             if step_id not in self._actions:
#                 print(f"No ground truth available for step {step_id}!")
#                 return output
#             output["action"][:, 3:7] = self._actions[step_id][:, 3:7]
#         if self._ground_truth_position:
#             if step_id not in self._actions:
#                 print(f"No ground truth available for step {step_id}!")
#                 return output
#             output["action"][:, :3] = self._actions[step_id][:, :3]
#         if self._ground_truth_gripper:
#             if step_id not in self._actions:
#                 print(f"No ground truth available for step {step_id}!")
#                 return output
#             output["action"][:, 7] = self._actions[step_id][:, 7]

#         if self._record_actions:
#             self._actions[step_id] = output["action"]

#         return output

#     def save(self, ep_dir):
#         if self._record_actions:
#             torch.save(self._actions, ep_dir / "actions.pth")

#     @property
#     def device(self):
#         if self._model is not None:
#             return next(self._model["model"].parameters()).device
#         return next(self._model_position["model"].parameters()).device  # type: ignore

#     def eval(self):
#         if self._model is not None:
#             self._model["model"].eval()
#         else:
#             self._model_position["model"].eval()  # type: ignore
#             self._model_rotation["model"].eval()  # type: ignore
#             self._model_gripper["model"].eval()  # type: ignore


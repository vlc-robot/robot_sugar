from typing import List, Optional, Union
import os

from ..default import get_config as get_default_config
from ..default import CN

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    config = get_default_config(config_paths, opts)

    config.defrost()

    if isinstance(config.DATASET.taskvars, str):
        if not os.path.exists(config.DATASET.taskvars):
            config.DATASET.taskvars = config.DATASET.taskvars.split(',')

    if isinstance(config.DATASET.camera_ids, str):
         config.DATASET.camera_ids = [int(v) for v in config.DATASET.camera_ids.split(',')]

    if isinstance(config.MODEL.action_head_config.multiscale_pc_layers, str):
        config.MODEL.action_head_config.multiscale_pc_layers = [int(v) for v in config.MODEL.action_head_config.multiscale_pc_layers.split(',')]
    elif isinstance(config.MODEL.action_head_config.multiscale_pc_layers, int):
        config.MODEL.action_head_config.multiscale_pc_layers = [config.MODEL.action_head_config.multiscale_pc_layers]
        
    config.CMD_TRAILING_OPTS = []
    config.freeze()

    return config
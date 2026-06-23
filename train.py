import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    if "task" in cfg and hasattr(cfg.task, "action_repr"):
        action_repr = cfg.task.action_repr
        if action_repr == "angle":
            action_dim = 1 + int(cfg.task.num_segments)
        else:
            action_dim = 1 + 2 * int(cfg.task.num_imus)
        cfg.task.action_dim = action_dim
        if "shape_meta" in cfg.task and "action" in cfg.task.shape_meta:
            cfg.task.shape_meta.action.shape = [action_dim]
        # keep encoder vs joint_angle obs mutually exclusive if requested
        use_joint_angle_obs = bool(getattr(cfg.task, "use_joint_angle_obs", False))
        if "shape_meta" in cfg.task and "obs" in cfg.task.shape_meta:
            obs_meta = cfg.task.shape_meta.obs
            if "encoder" in obs_meta:
                obs_meta.encoder.ignore_by_policy = use_joint_angle_obs
            if "joint_angle" in obs_meta:
                obs_meta.joint_angle.ignore_by_policy = not use_joint_angle_obs

    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

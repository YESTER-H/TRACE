# SPDX-License-Identifier: Apache-2.0
"""Unchanged U1.5 feature/reward families on logical state; only Recovery learns."""
import torch
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_apply_inverse
from cyclo_lab.manager_based.mimic.mdp import rewards as mimic_rewards
from cyclo_lab.manager_based.mimic.mdp import events as mimic_events
from .body_adapter import BODY_NAMES, tensor, logical_contact_history


def command(env, name="reference_trajectory"):
    return env.command_manager.get_term(name)


def live_tracking_command(env, command_name):
    c = command(env,command_name)
    return torch.cat((c.reference.joint_position[c.tracking_phase],c.reference.joint_velocity[c.tracking_phase]),-1)


def live_anchor_orientation(env, command_name):
    c = command(env,command_name)
    q = c.reference.body_orientation_w[c.tracking_phase,c.logical_anchor_id]
    _,relative = subtract_frame_transforms(c.robot_anchor_pos_w,c.robot_anchor_quat_w,
                                            c.robot_anchor_pos_w,q)
    return matrix_from_quat(relative)[...,:2].reshape(env.num_envs,6)


def oracle_weight(env):
    return torch.zeros((env.num_envs,1),device=env.device)


def success_bonus(env):
    """Terminal Recovery completion (final frame + upright); no event counting."""
    return command(env).success_mask().float()


def oracle_recovery_identity(env):
    """Training-only [Supine, Prone] one-hot router; never an actor network input."""
    c = command(env)
    return torch.nn.functional.one_hot(c.recovery_reference_ids,len(c.recovery_references)).float()


def base_linear_velocity(env):
    c = command(env)
    return quat_apply_inverse(c.logical_body_quat_w[:,0],c.logical_body_lin_vel_w[:,0])


def joint_position(env):
    c = command(env)
    return (tensor(c.robot.data.joint_pos)-tensor(c.robot.data.default_joint_pos))[:,c.joint_ids]


def joint_velocity(env):
    c = command(env)
    return (tensor(c.robot.data.joint_vel)-tensor(c.robot.data.default_joint_vel))[:,c.joint_ids]


def canonical_home_joint_offset_noise(env, env_ids, asset_cfg, pos_distribution_params=None,
                                     operation="abs", distribution="uniform"):
    """Preserve native startup noise, then synchronize offset by action names.

    The upstream event writes physical-index q into action-index offset. Those
    indices coincide on old PhysX but not on Newton's DFS joint layout.
    The synchronization makes no additional random draws or state writes.
    """
    mimic_events.apply_home_joint_offset_noise(
        env,env_ids,asset_cfg,pos_distribution_params,operation,distribution)
    robot = env.scene[asset_cfg.name]
    action = env.action_manager.get_term("joint_pos")
    ordered = tensor(robot.data.default_joint_pos)[:,action._joint_ids]
    if env_ids is None:
        action._offset.copy_(ordered)
    else:
        action._offset[env_ids] = ordered[env_ids]


def root_height_reward(env, command_name, std):
    c = command(env,command_name)
    error = (c.body_pos_w[:,0,2]-c.robot_body_pos_w[:,0,2]).square()
    return torch.exp(-error/std**2)*2.5


def imitation_reward(env, command_name, std, kernel_name):
    # Always Recovery: no Tracking reward leakage, same 2.5 acquisition scale.
    return getattr(mimic_rewards,kernel_name)(env,command_name,std)*2.5


def undesired_contacts(env, threshold=1.0):
    c = command(env)
    history = logical_contact_history(env,c.adapter)
    allowed = {"left_ankle_roll_link","right_ankle_roll_link",
               "left_wrist_roll_rubber_hand","right_wrist_roll_rubber_hand",
               "left_knee_link","right_knee_link"}
    ids = [i for i,n in enumerate(BODY_NAMES) if n not in allowed]
    return (history[:,:,ids].norm(dim=-1).amax(1)>threshold).sum(-1)


def persistent_failure(env):
    c = command(env)
    c.step_error_snapshot = {k:v.detach().clone() for k,v in c.target_error_components().items()}
    c.step_error_snapshot["joint_position"] = (c.joint_pos-tensor(c.robot.data.joint_pos)[:,c.joint_ids]).norm(dim=-1)
    c.step_error_snapshot["joint_velocity"] = (c.joint_vel-tensor(c.robot.data.joint_vel)[:,c.joint_ids]).norm(dim=-1)
    return c.update_persistent_target_error()


def nonfinite_state(env):
    c = command(env)
    state = torch.cat((tensor(c.robot.data.root_link_pose_w),tensor(c.robot.data.root_com_vel_w),
                       tensor(c.robot.data.joint_pos),tensor(c.robot.data.joint_vel)),-1)
    return ~torch.isfinite(state).all(-1)


def success_completion(env):
    c = command(env)
    result = c.success_mask()
    c._event("success_completions",result.sum())
    return result


def grace_failure(env):
    c = command(env)
    result = c.final_mask()&(c.hold_steps>=c.grace_steps-1)&~c.upright_mask()
    c._event("grace_failures",result.sum())
    return result

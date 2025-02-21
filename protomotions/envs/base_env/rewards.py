from typing import Dict, Tuple
from omegaconf import DictConfig
import torch
from torch import Tensor

from protomotions.envs.mimic.mimic_utils import mul_exp_mean


    ###############################################################
    # Rewards
    ###############################################################
    
    
    
    
@torch.jit.script
def reward_tracking_lin_vel(commands: Tensor, base_lin_vel: Tensor, tracking_sigma: float) -> Tensor:
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)

@torch.jit.script
def reward_tracking_ang_vel(commands: Tensor, base_ang_vel: Tensor, tracking_sigma: float) -> Tensor:
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)

@torch.jit.script 
def reward_lin_vel_z(base_lin_vel: Tensor) -> Tensor:
    return torch.square(base_lin_vel[:, 2])

@torch.jit.script
def reward_action_rate(last_actions: Tensor, actions: Tensor) -> Tensor:
    return torch.sum(torch.square(last_actions - actions), dim=1)

@torch.jit.script
def reward_similar_to_default(dof_pos: Tensor, default_dof_pos: Tensor) -> Tensor:
    return torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1)

@torch.jit.script
def reward_ang_vel_xy(base_ang_vel: Tensor) -> Tensor:
    return torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)

@torch.jit.script
def reward_orientation(projected_gravity: Tensor) -> Tensor:
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

@torch.jit.script
def reward_dof_acc(last_dof_vel: Tensor, dof_vel: Tensor, dt: float) -> Tensor:
    return torch.sum(torch.square((last_dof_vel - dof_vel) / dt), dim=1)

@torch.jit.script
def reward_joint_power(dof_vel: Tensor, control_force: Tensor) -> Tensor:
    return torch.sum(torch.abs(dof_vel) * torch.abs(control_force), dim=1)

@torch.jit.script
def reward_control_force(control_force: Tensor) -> Tensor:
    return torch.sum(torch.square(control_force), dim=1)

@torch.jit.script
def reward_dof_vel(dof_vel: Tensor) -> Tensor:
    return torch.sum(torch.square(dof_vel), dim=1)

@torch.jit.script
def reward_collision(contact_forces: Tensor, penalised_contact_indices: Tensor) -> Tensor:
    contact_norms = torch.norm(contact_forces[:, penalised_contact_indices, :], dim=-1)
    return torch.sum(1.0 * (contact_norms > 0.1), dim=1)



# 需要参数重构的函数 ====================================
@torch.jit.script
def reward_base_height(base_height: Tensor, target_height: float) -> Tensor:
    return torch.square(base_height - target_height)

@torch.jit.script
def reward_smoothness(
    actions: Tensor, 
    last_actions: Tensor, 
    last_last_actions: Tensor
) -> Tensor:
    return torch.sum(
        torch.square(actions - 2 * last_actions + last_last_actions),  # 简化二阶差分计算
        dim=1
    )

# 需要状态维护的困难函数 ===================================
@torch.jit.script
def reward_feet_air_time(
    contact_forces: Tensor,
    feet_indices: Tensor,
    feet_air_time: Tensor,
    last_contacts: Tensor,
    dt: float,
    commands: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    # 需要维护状态变量 last_contacts 和 feet_air_time
    # 返回奖励值及更新后的状态
    contact = contact_forces[:, feet_indices, 2] > 1.
    contact_filt = torch.logical_or(contact, last_contacts)
    first_contact = (feet_air_time > 0.) * contact_filt
    new_air_time = feet_air_time + dt
    rew_airTime = torch.sum((new_air_time - 0.5) * first_contact, dim=1)
    rew_airTime *= torch.norm(commands[:, :2], dim=1) > 0.1
    reset_air_time = new_air_time * ~contact_filt
    return rew_airTime, contact, reset_air_time

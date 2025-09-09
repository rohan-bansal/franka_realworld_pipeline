
from typing import Union, Sequence, Dict, Optional, Tuple

from copy import deepcopy
from collections import OrderedDict
import functools
from scipy.spatial.transform import Rotation
import numpy as np
from scipy.interpolate import CubicSpline
from torchgen.utils import split_name_params

def action_dict_to_vector(
        action_dict: Dict[str, np.ndarray], 
        action_keys: Optional[Sequence[str]]=None) -> np.ndarray:
    if action_keys is None:
        action_keys = list(action_dict.keys())
    actions = [action_dict[k] for k in action_keys]

    action_vec = np.concatenate(actions, axis=-1)
    return action_vec


def vector_to_action_dict(
        action: np.ndarray, 
        action_shapes: Dict[str, Tuple[int]],
        action_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    action_dict = dict()
    start_idx = 0
    for key in action_keys:
        this_act_shape = action_shapes[key]
        this_act_dim = np.prod(this_act_shape)
        end_idx = start_idx + this_act_dim
        action_dict[key] = action[...,start_idx:end_idx].reshape(
            action.shape[:-1]+this_act_shape)
        start_idx = end_idx
    return action_dict

def compute_omega_base_frame(axis_angles: np.ndarray, dt: float=1.0, max_omega: float=0.5):
    """
    Approximate angular omega from axis_angle array in base frame
    Input: axis_angle is Tx3
    """
    num_points, coordinate = axis_angles.shape

    # print(axis_angles)

    if coordinate!=3:
        raise ValueError("Input shape should be 3 with x,y,z")
    if dt <= 1e-4:
        print("WARNING: dt too small for compute velocity")
        return

    rotation_mats = Rotation.from_rotvec(axis_angles)

    twists = []
    for i in range(num_points):

        if i==(num_points-1):
            omega = np.zeros(3)
        else:
            R_i = rotation_mats[i].as_matrix()
            R_ip1 = rotation_mats[i+1].as_matrix()
            # TODO: how accurate it needs to be? To satisfy the R_dot@R_T + R@R_dot_T=0\
            # TODO: what if using R_{T+1}@R_T
            dR = (R_ip1 - R_i) / dt
            omega_mat = dR @ R_i.T
            omega = np.array([
                omega_mat[2,1],
                omega_mat[0,2],
                omega_mat[1,0]
            ])

        if np.any(np.abs(omega) > max_omega):
            print(f"Warning: Omega {omega} is larger than {max_omega}. Clipped ot 0")
            # for tt in range(num_points):
            #     print(f"axis angle {axis_angles[tt]} at {i}")
            # print(f" omega {omega}")
            # omega = np.clip(omega, -max_omega, max_omega) # should not clip
            omega = np.zeros(3)
        twists.append(omega)

    # print("max omega: ", np.max(np.abs(twists)))
    # if np.any(np.abs(twists) > max_omega):
    #     print(f"Omega is larger than {max_omega}")
    return np.squeeze(np.array(twists))


def compute_velocity_base_frame(traj: np.ndarray, dt: float=1.0, max_vel: float=2.0):
    """
    A simple linear velocity approximation in base_frame. Forward euler
    v(t) = [s(t+1)-s(t)] / dt

    Input x,y,z position = np.array Tx3
    max_vel: float in m/s. To limit the max vel of end pos.
    Output x,y,z velocity = np.array Tx3
    """
    vel = np.zeros_like(traj, dtype=np.float64)

    time_len, coordinate = traj.shape
    if coordinate!=3:
        raise ValueError("Input shape should be 3 with x,y,z")
    if dt <= 1e-4:
        print("WARNING: dt too small")
        return np.squeeze(vel)

    # vel[1:-1, :] = (traj[2:,:]-traj[:-2, :]) * 0.5 / dt
    vel[:-1, :] = (traj[1:,:]-traj[:-1, :]) / dt

    if np.any(np.abs(vel) > max_vel):
        # raise ValueError(f"Velocity is larger than {max_vel}")
        print(f"Warning: linear vel is larger than {np.max(np.abs(vel))}, clipped")
        vel = np.clip(vel, -max_vel, max_vel)
    return np.squeeze(vel)

def fit_spline_to_action_sequence(act, num_knots=None):
    if num_knots is None:
        num_knots = act.shape[0]

    knots = np.linspace(0, act.shape[0] - 1, num_knots, dtype=int)
    splines = []

    for i in range(act.shape[1]):
        y = act[knots, i]
        c = CubicSpline(knots, y, axis=3)
        splines.append(c)

    return splines

def sample_position_and_velocity_from_splines(splines, t):
    p = []
    v = []
    for i in range(splines):
        p.append(splines[i](t))
        v.append(splines[i].derivative(1)(t))

    return p, v

def test_omega():
    # angular_array = np.array([[1,1,1], [10,10,10]])
    # angular_array = np.array([[-3.090345,  0.152242, -0.073512], [-3.066816,  0.145094, -0.07335]])
    angular_array = np.array([[ 3.107387, -0.039294,  0.068526], [3.037761,-0.045868, 0.059463], [3.011958, -0.044979, 0.0624]])

    data = np.array([
        [3.107387, -0.039294, 0.068526],
        [3.037761, -0.045868, 0.059463],
        [3.011958, -0.044979, 0.0624],
        [2.978058, -0.014871, 0.063966],
        [2.980623, -0.027228, 0.062491],
        [2.983492, -0.022446, 0.063193],
        [2.954604, -0.029651, 0.044111],
        [2.940293, -0.044574, 0.028019],
        [2.879453, -0.061043, 0.027404],
        [2.876623, -0.059886, 0.027363],
        [2.879825, -0.061396, 0.017221],
        [2.824517, -0.056212, 0.043559]
    ])
    omega = compute_omega_base_frame(data, dt=0.1)
    lin = compute_velocity_base_frame(angular_array)

    print(f"omega {omega}")

if __name__=="__main__":
    test_omega()

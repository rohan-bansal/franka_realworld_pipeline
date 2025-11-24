import numpy as np
import robosuite.utils.transform_utils as TUtils
from easydict import EasyDict as edict

def scale_action(controller, action):
    if controller.action_scale is None:
        controller.action_scale = abs(controller.output_max - controller.output_min) / abs(controller.input_max - controller.input_min)
        controller.action_output_transform = (controller.output_max + controller.output_min) / 2.0
        controller.action_input_transform = (controller.input_max + controller.input_min) / 2.0
    # action = np.clip(action, controller.input_min, controller.input_max)
    transformed_action = (action - controller.action_input_transform) * controller.action_scale + controller.action_output_transform

    return transformed_action

def reverse_scale_action(controller, transformed_action):
    """
    Reverses the scaling process performed by the scale_action method.

        if self.action_scale is None:
            self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
            self.action_output_transform = (self.output_max + self.output_min) / 2.0
            self.action_input_transform = (self.input_max + self.input_min) / 2.0
        action = np.clip(action, self.input_min, self.input_max)
        transformed_action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform

        return transformed_action
    
    Returns:
        np.array: Original action before scaling.
    """
    # Calculate scaling and transformation parameters
    if controller.action_scale is None:
        action_scale = abs(controller.output_max - controller.output_min) / abs(controller.input_max - controller.input_min)
        action_output_transform = (controller.output_max + controller.output_min) / 2.0
        action_input_transform = (controller.input_max + controller.input_min) / 2.0
    else:
        action_scale = controller.action_scale
        action_output_transform = controller.action_output_transform
        action_input_transform = controller.action_input_transform

    # Reverse the transformation
    action = (transformed_action - action_output_transform) / action_scale + action_input_transform

    # Clip the action to be within the input range
    # action = np.clip(action, controller.input_min, controller.input_max)

    return action

def pose2mat(pos, ori):
    if ori.shape[0] == 4:
        ori = TUtils.quat2mat(ori)
    pose_mat = np.zeros((4, 4), dtype=np.float32)
    pose_mat[:3, :3] = ori
    pose_mat[:3, 3] = pos
    pose_mat[3, 3] = 1.0
    return pose_mat

def set_goal_position(delta, current_position, position_limit=None, set_pos=None):
    """
    Calculates and returns the desired goal position, clipping the result accordingly to @position_limits.
    @delta and @current_position must be specified if a relative goal is requested, else @set_pos must be
    specified to define a global goal position

    Args:
        delta (np.array): Desired relative change in position
        current_position (np.array): Current position
        position_limit (None or np.array): 2d array defining the (min, max) limits of permissible position goal commands
        set_pos (None or np.array): If set, will ignore @delta and set the goal position to this value

    Returns:
        np.array: calculated goal position in absolute coordinates

    Raises:
        ValueError: [Invalid position_limit shape]
    """
    n = len(current_position)
    if set_pos is not None:
        goal_position = set_pos
    else:
        goal_position = current_position + delta

    if position_limit is not None:
        if position_limit.shape != (2, n):
            raise ValueError(
                "Position limit should be shaped (2,{}) " "but is instead: {}".format(n, position_limit.shape)
            )

        # Clip goal position
        goal_position = np.clip(goal_position, position_limit[0], position_limit[1])

    return goal_position


def set_goal_orientation(delta, current_orientation, orientation_limit=None, set_ori=None):
    """
    Calculates and returns the desired goal orientation, clipping the result accordingly to @orientation_limits.
    @delta and @current_orientation must be specified if a relative goal is requested, else @set_ori must be
    an orientation matrix specified to define a global orientation

    Args:
        delta (np.array): Desired relative change in orientation, in axis-angle form [ax, ay, az]
        current_orientation (np.array): Current orientation, in rotation matrix form
        orientation_limit (None or np.array): 2d array defining the (min, max) limits of permissible orientation goal commands
        set_ori (None or np.array): If set, will ignore @delta and set the goal orientation to this value

    Returns:
        np.array: calculated goal orientation in absolute coordinates

    Raises:
        ValueError: [Invalid orientation_limit shape]
    """
    # directly set orientation
    if set_ori is not None:
        goal_orientation = set_ori

    # otherwise use delta to set goal orientation
    else:
        # convert axis-angle value to rotation matrix
        quat_error = TUtils.axisangle2quat(delta)
        rotation_mat_error = TUtils.quat2mat(quat_error)
        goal_orientation = np.dot(rotation_mat_error, current_orientation)

    # check for orientation limits
    if np.array(orientation_limit).any():
        if orientation_limit.shape != (2, 3):
            raise ValueError(
                "Orientation limit should be shaped (2,3) " "but is instead: {}".format(orientation_limit.shape)
            )

        # Convert to euler angles for clipping
        euler = TUtils.mat2euler(goal_orientation)

        # Clip euler angles according to specified limits
        limited = False
        for idx in range(3):
            if orientation_limit[0][idx] < orientation_limit[1][idx]:  # Normal angle sector meaning
                if orientation_limit[0][idx] < euler[idx] < orientation_limit[1][idx]:
                    continue
                else:
                    limited = True
                    dist_to_lower = euler[idx] - orientation_limit[0][idx]
                    if dist_to_lower > np.pi:
                        dist_to_lower -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_lower += 2 * np.pi

                    dist_to_higher = euler[idx] - orientation_limit[1][idx]
                    if dist_to_lower > np.pi:
                        dist_to_higher -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_higher += 2 * np.pi

                    if dist_to_lower < dist_to_higher:
                        euler[idx] = orientation_limit[0][idx]
                    else:
                        euler[idx] = orientation_limit[1][idx]
            else:  # Inverted angle sector meaning
                if orientation_limit[0][idx] < euler[idx] or euler[idx] < orientation_limit[1][idx]:
                    continue
                else:
                    limited = True
                    dist_to_lower = euler[idx] - orientation_limit[0][idx]
                    if dist_to_lower > np.pi:
                        dist_to_lower -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_lower += 2 * np.pi

                    dist_to_higher = euler[idx] - orientation_limit[1][idx]
                    if dist_to_lower > np.pi:
                        dist_to_higher -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_higher += 2 * np.pi

                    if dist_to_lower < dist_to_higher:
                        euler[idx] = orientation_limit[0][idx]
                    else:
                        euler[idx] = orientation_limit[1][idx]
        if limited:
            goal_orientation = TUtils.euler2mat(np.array([euler[0], euler[1], euler[2]]))
    return goal_orientation

def action2eef_target_pose(current_pose, action,  controller):
    '''
    current_pose: 4x4 np array
    action: np array with shape (7,)
    '''
    
    current_pos = current_pose[:3, 3]
    current_ori_mat = current_pose[:3, :3]


    
    delta = action[:-1]
    if hasattr(controller, "scale_action"):
        scaled_delta = controller.scale_action(delta)
    else:
        scaled_delta = scale_action(controller, delta)

    goal_ori = set_goal_orientation(
                scaled_delta[3:], current_ori_mat, orientation_limit=controller.orientation_limits, set_ori=None
            )
    goal_pos = set_goal_position(
            scaled_delta[:3], current_pos, position_limit=controller.position_limits, set_pos=None
        )
    
    gripper_action = np.array(action[-1:])

    goal_pose = pose2mat(goal_pos, goal_ori)

    return goal_pose, gripper_action

def eef_target_pose2action(current_pose, goal_pose, controller, gripper_action=None):
    '''
    current_pose: 4x4 np array
    goal_pose: 4x4 np array
    gripper_action: np array with shape (1,) or None
    '''
    
    current_pos = current_pose[:3, 3]
    current_ori_mat = current_pose[:3, :3]

    goal_pos = goal_pose[:3, 3]
    goal_ori_mat = goal_pose[:3, :3]

    delta_pos = goal_pos - current_pos

    # goal_ori_mat = TUtils.quat2mat(goal_quat)
    R_error = np.dot(goal_ori_mat, current_ori_mat.T)
    error_quat = TUtils.mat2quat(R_error)
    delta_ori = TUtils.quat2axisangle(error_quat)

    scaled_delta = np.concatenate([delta_pos, delta_ori])
    delta = reverse_scale_action(controller, scaled_delta)
    action = np.concatenate([delta, gripper_action]) if gripper_action is not None else delta
    return action


controller = edict(
        {
            'action_scale': np.array([0.125, 0.125, 0.125, 0.25, 0.25, 0.25]),
            # 'action_scale': np.array([0.018, 0.045, 0.045, 0.2 , 0.055 , 0.3 ]),
            # 'action_scale': np.array([0.01390294, 0.03738892, 0.03563335,  0.10477141, 0.05334425, 0.19542998 ]),
            'output_max': np.array([0.05, 0.05, 0.05, 0.5 , 0.5 , 0.5 ]),
            'output_min': np.array([-0.05, -0.05, -0.05, -0.5 , -0.5 , -0.5 ]),
            'input_max': np.array([1., 1., 1., 1., 1., 1.]),
            'input_min': np.array([-1., -1., -1., -1., -1., -1.]),
            'action_output_transform': np.array([0., 0., 0., 0., 0., 0.]),
            'action_input_transform': np.array([0., 0., 0., 0., 0., 0.]),
            'orientation_limits': None,
            'position_limits': None
        }
)
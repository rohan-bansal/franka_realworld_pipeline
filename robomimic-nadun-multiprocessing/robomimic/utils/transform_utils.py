import math
import numpy as np
import torch
# from triton.language import tensor

EPS = np.finfo(float).eps * 4.

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    Reimplement math.isclose()
    """
    if hasattr(math, "isclose"):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def quat2axisangle(quat):
    """
    Converts (x, y, z, w) quaternion to axis-angle format.
    Returns a unit vector direction and an angle.
    """

    # conversion from axis-angle to quaternion:
    #   qw = cos(theta / 2); qx, qy, qz = u * sin(theta / 2)

    # normalize qx, qy, qz by sqrt(qx^2 + qy^2 + qz^2) = sqrt(1 - qw^2)
    # to extract the unit vector

    # clipping for scalar with if-else is orders of magnitude faster than numpy
    if quat[3] > 1.:
        quat[3] = 1.
    elif quat[3] < -1.:
        quat[3] = -1.

    den = np.sqrt(1. - quat[3] * quat[3])
    if isclose(den, 0.):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3), 0.

    # convert qw to theta
    theta = 2. * math.acos(quat[3])

    return quat[:3] / den, 2. * math.acos(quat[3])



def vec2axisangle(vec):
    """
    Converts Euler vector (exponential coordinates) to axis-angle.
    """
    angle = np.linalg.norm(vec)
    if isclose(angle, 0.):
        # treat as a zero rotation
        return np.array([1., 0., 0.]), 0.
    axis = vec / angle
    return axis, angle


def axisangle2vec(axis, angle):
    """
    Converts axis-angle to Euler vector (exponential coordinates).
    """
    return axis * angle

def axisangle2mat(axis, angle):
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    R = np.array([
        [cos_theta + ux ** 2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta,
         ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy ** 2 * (1 - cos_theta),
         uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta,
         cos_theta + uz ** 2 * (1 - cos_theta)]
    ])

    return R

def axis_angle_to_rotation_matrix(axis_angle):
    """
    Convert an axis-angle representation (3D vector) to a rotation matrix.
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:  # Avoid division by zero
        return np.eye(3)
    axis = axis_angle / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def geodesic_distance(axis_angle1, axis_angle2):
    """
    Calculate the geodesic distance between two axis-angle representations.
    """
    R1 = axis_angle_to_rotation_matrix(axis_angle1)
    R2 = axis_angle_to_rotation_matrix(axis_angle2)
    R_relative = np.dot(R1.T, R2)  # Relative rotation matrix
    trace = np.trace(R_relative)
    trace = np.clip(trace, -1.0, 3.0)  # Ensure numerical stability
    angle = np.arccos((trace - 1) / 2)
    return angle

def mat2axisangle(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if np.isclose(angle, 0):
        # Angle is close to 0
        return np.array([1, 0, 0]), 0  # Default axis
    elif np.isclose(angle, np.pi):
        # Angle is close to pi
        # Special case: The rotation matrix may represent a 180 degree rotation
        R_plus_I = R + np.eye(3)
        axis = np.sqrt((R_plus_I.diagonal() + 1) / 2)
        return axis / np.linalg.norm(axis), angle

    # General case
    ux = (R[2, 1] - R[1, 2]) / (2 * np.sin(angle))
    uy = (R[0, 2] - R[2, 0]) / (2 * np.sin(angle))
    uz = (R[1, 0] - R[0, 1]) / (2 * np.sin(angle))

    return np.array([ux, uy, uz]), angle

def mat2quat(rmat, precise=False):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat: 3x3 rotation matrix
        precise: If isprecise is True, the input matrix is assumed to be a precise
             rotation matrix and a faster algorithm is used.

    Returns:
        vec4 float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]
    if precise:
        # This code uses a modification of the algorithm described in:
        # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
        # which is itself based on the method described here:
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        # Altered to work with the column vector convention instead of row vectors
        m = M.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
            else:
                t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
        else:
            if m[0, 0] < -m[1, 1]:
                t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
            else:
                t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]
        q = np.array(q)
        q *= 0.5 / np.sqrt(t)
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]



def axisangle2quat(axis, angle):
    """
    Converts axis-angle to (x, y, z, w) quat.
    """

    # handle zero-rotation case
    if isclose(angle, 0.):
        return np.array([0., 0., 0., 1.])

    # make sure that axis is a unit vector
    assert isclose(np.linalg.norm(axis), 1., rel_tol=1e-3)

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.)
    q[:3] = axis * np.sin(angle / 2.)
    return q

def quat2mat(quaternion):
    """
    Converts given quaternion (x, y, z, w) to matrix.

    Args:
        quaternion: vec4 float angles

    Returns:
        3x3 rotation matrix
    """

    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )

def normalize_euler_vector(matrix):
    print("entering function ***************")
    matrix = np.array(matrix)
    for i in range(1, len(matrix)):
        euler_vec_prev = matrix[i - 1, 3:6]
        euler_vec_curr = matrix[i, 3:6]

        # Get axis, angle from euler vectors
        axis_curr, angle_curr = vec2axisangle(euler_vec_curr)
        print(axis_curr, angle_curr)

        # Check if axes are nearly opposite
        dot_product = np.dot(euler_vec_prev, euler_vec_curr)
        print(f"dot product {dot_product}")
        if dot_product < -0.6:
            # Flip the current axis and adjust the angle
            new_axis_curr = -axis_curr
            angle_curr = 2 * np.pi - angle_curr

            euler_vec_curr = axisangle2vec(new_axis_curr, angle_curr)
            matrix[i, 3:6] = euler_vec_curr

    return matrix

def convert_sign_axis_angle_tensor(rot_vec: torch.tensor) -> torch.tensor:
    """
    ABORTED
    To avoid the jumping of axis angle around the +-pi angle.
    Make sure axis has the same sign (i.e. x axis positive)

    Args:
        rot_vec:
    Returns:

    """
    magnitude = torch.norm(rot_vec)
    if magnitude == 0:
        return rot_vec
    unit_axis = rot_vec / magnitude

    if unit_axis[0] == 0.0:
        assert "x axis of the axis angle is 0.0, need to be handled"

    if unit_axis[0] < 0:
        # reverse the axis, keep the axis angle the same
        unit_axis = -unit_axis
        magnitude = -magnitude
        # avoid the -pi +pi jump
        # add 2pi won't change rotation, but can make sure the the continuity in value
        # assumption is, the axis inversion happens when near +-pi, -> -axis -> 2pi-mag
        # TODO: this is wrong
        if magnitude < 0 : #-torch.pi:
            magnitude += 2*torch.pi
        else:
            assert "magnitude should be > 0"
        # if magnitude > np.pi:
        #     magnitude -= 2*np.pi

    return magnitude*unit_axis

def convert_sign_axis_angle(rot_vec: np.ndarray) -> np.ndarray:
    """
    ABORTED
    To avoid the jumping of axis angle around the +-pi angle.
    Make sure axis has the same sign (i.e. x axis positive)

    Args:
        rot_vec:
    Returns:

    """
    magnitude = np.linalg.norm(rot_vec)
    if magnitude == 0:
        return rot_vec
    unit_axis = rot_vec / magnitude

    if unit_axis[0] == 0.0:
        assert "x axis of the axis angle is 0.0, need to be handled"

    if unit_axis[0] < 0:
        # reverse the axis, keep the axis angle the same
        unit_axis = -unit_axis
        magnitude = -magnitude
        # avoid the -pi +pi jump
        # add 2pi won't change rotation, but can make sure the the continuity in value
        # assumption is, the axis inversion happens when near +-pi, -> -axis -> 2pi-mag
        if magnitude < 0: # -np.pi:
            magnitude += 2*np.pi
        else:
            assert False, "magnitude should be > 0"
        # if magnitude > np.pi:
        #     magnitude -= 2*np.pi

    return magnitude*unit_axis

def absolute_action_to_delta(abs_action, start_pos, start_quat):
    # convert action from absolute to relative for compatibility with rest of code
    action = np.array(abs_action)

    # absolute pose target
    target_pos = action[:3]
    axis, angle = vec2axisangle(action[3:6])
    target_rot = quat2mat(axisangle2quat(axis, angle))

    delta_position = target_pos - start_pos

    start_rot = quat2mat(start_quat)

    delta_rot_mat = target_rot.dot(start_rot.T)
    delta_rot_quat = mat2quat(delta_rot_mat)
    delta_rot_aa = quat2axisangle(delta_rot_quat)
    delta_rotation = delta_rot_aa[0] * delta_rot_aa[1]

    action[:3] = delta_position
    action[3:6] = delta_rotation

    return action

def interpolate_rotations(R1, R2, num_steps, axis_angle=True):
    """
    From mimicGen
    Interpolate between 2 rotation matrices. If @axis_angle, interpolate the axis-angle representation
    of the delta rotation, else, use slerp.

    NOTE: I have verified empirically that both methods are essentially equivalent, so pick your favorite.
    """
    if axis_angle:
        # delta rotation expressed as axis-angle
        delta_rot_mat = R2.dot(R1.T)
        delta_quat = mat2quat(delta_rot_mat)
        delta_axis, delta_angle = quat2axisangle(delta_quat)

        # fix the axis, and chunk the angle up into steps
        rot_step_size = delta_angle / num_steps

        # convert into delta rotation matrices, and then convert to absolute rotations
        if delta_angle < 0.05:
            # small angle - don't bother with interpolation
            rot_steps = np.array([R2 for _ in range(num_steps)])
        else:
            delta_rot_steps = [quat2mat(axisangle2quat(delta_axis, i * rot_step_size)) for i in range(num_steps)]
            rot_steps = np.array([delta_rot_steps[i].dot(R1) for i in range(num_steps)])
    else:
        q1 = T.mat2quat(R1)
        q2 = T.mat2quat(R2)
        rot_steps = np.array([T.quat2mat(quat_slerp(q1, q2, tau=(float(i) / num_steps))) for i in range(num_steps)])

    # add in endpoint
    rot_steps = np.concatenate([rot_steps, R2[None]], axis=0)

    return rot_steps

if __name__ == "__main__":
    import torch
    test1 = torch.tensor([[3.01172060e+00, -8.24450809e-02, 9.14058581e-02], [-3.26645469, 0.0909564, -0.18281081]])
    res = convert_sign_axis_angle_tensor(test1[1,:])

    # test = np.array([[3.01172060e+00, -8.24450809e-02, 9.14058581e-02], [-3.26645469, 0.0909564, -0.18281081]])
    # res = normalize_euler_vector(test)
    # res = convert_sign_axis_angle(test[1,:])
    print(f"result {res}")

import numpy as np

# for FK
import spatialmath as sm
from general_robotics_toolbox.urdf import robot_from_xml_file, robot_from_xacro_file
from general_robotics_toolbox import fwdkin


xacro_fn = "/home/robot-aiml/acl_ws/src/cartesian_controllers_universal_robots/urdf/setup.urdf.xacro"

class FK_Solver():
    def __init__(self):
        root_link = "base_link"
        tip_link = "tool0"
        self.ur5 = robot_from_xacro_file(xacro_fn, root_link=root_link, tip_link=tip_link)
    def joints_to_ee_pose(self, q):
        remapped_q = np.copy(q)
        remapped_q[0] = q[2]
        remapped_q[2] = q[0]
        T = fwdkin(self.ur5, remapped_q)
        quat = sm.UnitQuaternion(T.R)
        calc_ee_pose = T.p.tolist() + quat.data[0].tolist()
        return calc_ee_pose
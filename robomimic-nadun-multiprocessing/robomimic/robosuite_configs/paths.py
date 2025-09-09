from pathlib import Path

script_dir = Path(__file__).resolve().parent

### Robot-specific controllers
def baxter():
    return script_dir / 'default_baxter.json'

def iiwa():
    return script_dir / 'default_iiwa.json'

def jaco():
    return script_dir / 'default_jaco.json'

def kinova3():
    return script_dir / 'default_kinova3.json'

def panda():
    return script_dir / 'default_panda.json'

def sawyer():
    return script_dir / 'default_sawyer.json'

def ur5e():
    return script_dir / 'default_ur5e.json'


### General controllers
def ik_pose():
    return script_dir / 'ik_pose.json'

def joint_position_nadun():
    return script_dir / 'joint_position_nadun.json'

def joint_position_original():
    return script_dir / 'joint_position.json'

def joint_torque():
    return script_dir / 'joint_torque.json'

def joint_velocity():
    return script_dir / 'joint_velocity_nadun.json'

def joint_velocity_original():
    return script_dir / 'joint_velocity.nadun'

def osc_pose():
    return script_dir / 'osc_pose.json'

def osc_position():
    return script_dir / 'osc_position.json'

import pandas as pd
import matplotlib.pyplot as plt
import pickle

from docutils.nodes import label
from matplotlib.pyplot import ylabel
import numpy as np
import pandas as pd
from sympy.utilities.mathml import apply_xsl


def plot_replay(data_fp=None, label=None, axs=None):
    # data_fp = '/media/nadun/Data/phd_project/experiment_logs/control_freq_eval/can_eval.pkl'

    with open(data_fp, 'rb') as f:
        data = pickle.load(f)

    print()

    X = []
    pos_y_1 = []
    pos_y_2 = []
    rot_y_1 = []
    rot_y_2 = []
    for freq in data:
        X.append(freq)
        errors = data[freq]
        pos_errors = errors['pos_errors']
        rot_errors = errors['rot_errors']

        mean_pos_error = sum(pos_errors)/len(pos_errors)
        mean_rot_error = sum(rot_errors)/len(rot_errors)

        max_pos_errors = errors['max_pos_errors']
        max_rot_errors = errors['max_rot_errors']

        max_pos_error = max(max_pos_errors)
        max_rot_error = max(max_rot_errors)

        pos_y_1.append(mean_pos_error)
        pos_y_2.append(max_pos_error)

        rot_y_1.append(mean_rot_error)
        rot_y_2.append(max_rot_error)


    axs[0].plot(X, pos_y_1, label=label)
    # axs[0].plot(X, pos_y_2, ls='None', color='red', markersize=8, marker='X', label='max')
    axs[0].set_title("Position Error")
    axs[0].set(ylabel='Mean Error (m)')

    axs[1].plot(X, rot_y_1, label=label)
    # axs[1].plot(X, rot_y_2, ls='None', color='red', markersize=8, marker='X', label='max')
    axs[1].set_title("Rotation Error")
    axs[1].set(ylabel="Mean Error (Quaternion magnitude)")

    fig.supxlabel('Control Freq')

def get_statistics_from_model_rollout(data_fp):
    df = pd.read_pickle(data_fp)

    X = []
    success_rates = []
    sim_steps = []

    for index, row in df.iterrows():
        cf = row['control_freq']
        X.append(cf)

        succ_rate = np.array(row['Success_Rate'])
        succ = np.mean(succ_rate)
        success_rates.append(succ)
        horizon = np.array(row['Horizon'])

        mask = succ_rate == 1.0
        horizon = horizon[mask]
        sim_steps.append(np.mean(horizon) * (500 / cf))

    return X, success_rates, sim_steps

def plot_model(data_fp):

    df = pd.read_pickle(data_fp)

    X = []
    success_rates = []
    sim_steps = []

    for index, row in df.iterrows():
        cf = row['control_freq']
        X.append(cf)

        succ_rate = np.array(row['Success_Rate'])
        succ = np.mean(succ_rate)
        success_rates.append(succ)
        horizon = np.array(row['Horizon'])

        mask = succ_rate == 1.0
        horizon = horizon[mask]
        sim_steps.append(np.mean(horizon) * (500/cf))

    fig, axs = plt.subplots(2)
    axs[0].plot(X, success_rates, label="Success Rate")
    axs[1].plot(X, sim_steps, color='red', label="Sim steps per success")

    fig.suptitle("Eval of OSC models over different control freqs")
    fig.supxlabel('Control Freq')
    plt.legend()

    plt.show()

    print()


def plot_sampling_freq_eval(data_fp, label):
    df = pd.read_excel(data_fp)
    control_freqs = df['control_freq']
    success_rates = df['Success_Rate']
    plt.title("Effect of changing observation sampling frequency on Lift Task")
    plt.plot(control_freqs, success_rates, label=label)

    print()



### Plot OSC Pose eval
data_fp = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/absolute_osc/square_image/20240919151432/logs/control_freq_eval_2024-10-20.pkl"
control_freqs, success_rates, sim_steps = get_statistics_from_model_rollout(data_fp)


fig, axs = plt.subplots(2)
# plt.plot(control_freqs, success_rates, color="maroon", label="OSC Pose Model")
axs[0].plot(control_freqs, success_rates, color="maroon", label="Commanded Pose Model")
axs[1].plot(control_freqs, sim_steps, color='maroon', label="Commanded Pose Model")

### Plot Joint Position Eval

data_fp = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/joint_position/square_image/20240921132122/logs/control_freq_eval_2024-10-20.pkl"
_, success_rates, sim_steps = get_statistics_from_model_rollout(data_fp)

axs[0].plot(control_freqs, success_rates, color="green", label="Joint Position Model")
axs[1].plot(control_freqs, sim_steps, color="green", label="Joint Position Model")
# plt.plot(control_freqs, sim_steps, color="green", label="Joint Position Model")
# plt.plot(control_freqs, success_rates, color="green", label="Joint Position Model")


### Plot reached osc eval

data_fp = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/reached_osc/square_image/20241010114545/logs/control_freq_eval_2024-10-20.pkl"
_, success_rates, sim_steps = get_statistics_from_model_rollout(data_fp)

axs[0].plot(control_freqs, success_rates, color="darkorange", label="Reached Pose Model")
axs[1].plot(control_freqs, sim_steps, color="darkorange", label="Reached Pose Model")


### Plotting stuff
axs[0].set_ylim(0,1)
axs[0].set_title("success rate over different speeds (Square)")
axs[0].set_ylabel("Success Rate")

axs[1].set_title("# sim steps at different speeds (Square)")
plt.xlabel('Control Freq (Hz)')
axs[1].set_ylabel("Sim Steps per success")
plt.legend()

plt.show()


### Plot tracking error eval
# fig, axs = plt.subplots(2)
# plot_replay(data_fp="/media/nadun/Data/phd_project/experiment_logs/tracking_error_eval/can_eval_osc.pkl",
#             label="OSC Controller", axs=axs)
# plot_replay(data_fp="/media/nadun/Data/phd_project/experiment_logs/tracking_error_eval/can_eval_jp.pkl",
#             label="Joint Position Controller", axs=axs)
#
# plt.legend()
# plt.show()


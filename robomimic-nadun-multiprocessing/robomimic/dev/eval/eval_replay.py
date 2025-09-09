import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from collections import OrderedDict


# DATA

lift_replay_data = {20: 0.0, 30: 11.0, 40: 31.0, 50: 68.0, 60: 96.0,
                    70: 118.0, 80: 128.0, 90: 141.0, 100: 150.0, 110: 156.0,
                    120: 156.0, 130: 159.0, 140: 162.0, 150: 167.0, 160: 172.0,
                    170: 175.0, 180: 177.0, 190: 177.0, 200: 180.0, 210: 183.0,
                    220: 185.0, 230: 187.0, 240: 188.0, 250: 190.0, 260: 189.0,
                    270: 189.0, 280: 190.0, 290: 191.0}

can_replay_data = {20: 0.0, 30: 2.0, 40: 32.0, 50: 68.0, 60: 92.0, 70: 92.0,
                   80: 100.0, 90: 109.0, 100: 117.0, 110: 121.0, 120: 129.0,
                   130: 134.0, 140: 139.0, 150: 141.0, 160: 142.0, 170: 146.0,
                   180: 150.0, 190: 150.0, 200: 149.0, 210: 150.0, 220: 153.0,
                   230: 153.0, 240: 152.0, 250: 158.0, 260: 157.0, 270: 159.0, 280: 156.0, 290: 154.0}


square_replay_data = {20: 0.0, 30: 2.0, 40: 4.0, 50: 9.0, 60: 9.0, 70: 12.0, 80: 17.0,
                      90: 26.0, 100: 32.0, 110: 33.0, 120: 37.0, 130: 43.0, 140: 43.0,
                      150: 49.0, 160: 55.0, 170: 55.0, 180: 63.0, 190: 60.0, 200: 64.0,
                      210: 65.0, 220: 67.0, 230: 73.0, 240: 73.0, 250: 71.0, 260: 72.0, 270: 77.0, 280: 79.0, 290: 83.0}



can_replay_speed_kp_commanded_data = {20: {50: 190.0, 100: 200.0, 150: 200.0, 200: 196.0, 250: 189.0, 300: 189.0},
                            30: {50: 146.0, 100: 180.0, 150: 185.0, 200: 179.0, 250: 174.0, 300: 165.0},
                            40: {50: 97.0, 100: 123.0, 150: 141.0, 200: 143.0, 250: 131.0, 300: 114.0},
                            50: {50: 47.0, 100: 78.0, 150: 86.0, 200: 83.0, 250: 67.0, 300: 55.0},
                            60: {50: 4.0, 100: 25.0, 150: 24.0, 200: 21.0, 250: 15.0, 300: 15.0}}


can_replay_speed_kp_reached_data = {20: {50: 144.0, 100: 184.0, 150: 192.0, 200: 193.0, 250: 196.0, 300: 197.0},
                                    30: {50: 98.0, 100: 161.0, 150: 179.0, 200: 188.0, 250: 191.0, 300: 191.0},
                                    40: {50: 75.0, 100: 120.0, 150: 145.0, 200: 165.0, 250: 173.0, 300: 176.0},
                                    50: {50: 38.0, 100: 94.0, 150: 118.0, 200: 132.0, 250: 146.0, 300: 153.0},
                                    60: {50: 4.0, 100: 57.0, 150: 68.0, 200: 74.0, 250: 81.0, 300: 88.0}}

dummy_tracking_error = [6,5,4,3,2,1]
dummy_tracking_error_commanded = [i*3 + 4 for i in dummy_tracking_error]
dummy_kp = [50, 100, 150, 200, 250, 300]

original_speed = 20
LINEWIDTH = 3




# Set general plot settings
# sns.set(context='notebook', style='darkgrid', font_scale=1.5)  # font_scale increases font size

def get_success_over_speed_and_kp(data=None, log_fn=None):
    if data is None and log_fn is None:
        raise Exception("Must pass data or log_fn")

    if log_fn is not None:
        with open(log_fn, "rb") as f:
            data = pickle.load(f)

    for speed in data:
        for kp in data[speed]:
            data[speed][kp] /= 200

    return data


def get_success_over_kp(data=None, log_fn=None):

    if data is None and log_fn is None:
        raise Exception("Must pass data or log_fn")

    if log_fn is not None:
        with open(log_fn, "rb") as f:
            data = pickle.load(f)

    for kp in data:
        data[kp] = data[kp] / 200

    return data

def main():

    sns.set_palette('colorblind')

    can_commanded_20_hz = "/media/nadun/Data/phd_project/experiment_logs/sim/replay_demos/can/replay_over_kp_commanded/replay_20_hz.pkl"
    can_commanded_40_hz = "/media/nadun/Data/phd_project/experiment_logs/sim/replay_demos/can/replay_over_kp_commanded/replay_40_hz.pkl"
    can_reached_20_hz = "/media/nadun/Data/phd_project/experiment_logs/sim/replay_demos/can/replay_over_kp_reached/replay_fast_20_hz.pkl"
    can_reached_40_hz = "/media/nadun/Data/phd_project/experiment_logs/sim/replay_demos/can/replay_over_kp_reached/replay_fast_40_hz.pkl"




    with plt.style.context("seaborn-v0_8-whitegrid"):
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"]  = 1.25
        plt.rcParams["font.size"] = 15
        plt.rcParams['grid.color'] = 'black'
        plt.rcParams['grid.alpha'] = 0.5

        # params = plt.rcParams
        fig, ax = plt.subplots()
        # plot_success_over_kp(lift_replay_data, ax, label="Lift")
        # plot_success_over_kp(can_replay_data, ax, label="Can")
        # plot_success_over_kp(square_replay_data, ax, label="Square")

        # Plotting success over kp
        can_commanded_20_hz_data = get_success_over_kp(log_fn=can_commanded_20_hz)
        can_reached_20_hz_data = get_success_over_kp(log_fn=can_reached_20_hz)
        can_commanded_40_hz_data = get_success_over_kp(log_fn=can_commanded_40_hz)
        can_reached_40_hz_data = get_success_over_kp(log_fn=can_reached_40_hz)

        sns.lineplot(can_commanded_20_hz_data, ax=ax, color="orange", label='Commanded Poses (c=1)', linestyle='--', linewidth=LINEWIDTH)
        sns.lineplot(can_reached_20_hz_data, ax=ax, color="deepskyblue", label='Reached Poses (c=1)', linewidth=LINEWIDTH)
        sns.lineplot(can_commanded_40_hz_data, ax=ax, color="darkorange", label='Commanded Poses (c=0.5)', linestyle='--', linewidth=LINEWIDTH)
        sns.lineplot(can_reached_40_hz_data, ax=ax, color="darkblue", label='Reached Poses (c=0.5)', linewidth=LINEWIDTH)
        plt.axvline(150, color="red", label="demo collection kp")

        sns.lineplot(x=dummy_kp, y=dummy_tracking_error_commanded, ax=ax, color="darkorange", label='Commanded Poses (c=0.5)', linestyle='--', linewidth=LINEWIDTH)
        sns.lineplot(x=dummy_kp, y=dummy_tracking_error,  ax=ax, color="darkblue", label='Reached Poses (c=0.5)', linewidth=LINEWIDTH)

        # ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
        ax.set_xlabel("Gain (Kp)")

        # Plotting success over speed

        can_replay_speed_kp_commanded_success = get_success_over_speed_and_kp(can_replay_speed_kp_commanded_data)
        can_replay_speed_kp_reached_success = get_success_over_speed_and_kp(can_replay_speed_kp_reached_data)

        # Iterate over speeds, pick succes for low and high gain
        can_commanded_low_gain_success = OrderedDict()
        can_commanded_high_gain_success = OrderedDict()

        can_reached_low_gain_success = OrderedDict()
        can_reached_high_gain_success = OrderedDict()

        # for speed in can_replay_speed_kp_commanded_success:
        #     c = original_speed/speed
        #     s_low = can_replay_speed_kp_commanded_success[speed][100]
        #     s_high = can_replay_speed_kp_commanded_success[speed][300]
        #     can_commanded_low_gain_success[c] = s_low
        #     can_commanded_high_gain_success[c] = s_high
        #
        #     s2_low = can_replay_speed_kp_reached_success[speed][100]
        #     s2_high = can_replay_speed_kp_reached_success[speed][300]
        #     can_reached_low_gain_success[c] = s2_low
        #     can_reached_high_gain_success[c] = s2_high
        #
        # sns.lineplot(can_commanded_low_gain_success, ax=ax[1], color="orange", label='Commanded Poses (Kp=100)', linestyle='--', linewidth=LINEWIDTH)
        # sns.lineplot(can_reached_low_gain_success, ax=ax[1], color="deepskyblue", label='Reached Poses (Kp=100)', linewidth=LINEWIDTH)
        # sns.lineplot(can_commanded_high_gain_success, ax=ax[1], color="darkorange", label='Commanded Poses (Kp=300)', linestyle='--', linewidth=LINEWIDTH)
        # sns.lineplot(can_reached_high_gain_success, ax=ax[1], color="darkblue", label='Reached Poses (Kp=300)', linewidth=LINEWIDTH)
        #
        # ax[1].text(0.01, 0.95, "(b)", transform=ax[1].transAxes, fontsize=12, fontweight='bold', va='top')
        # ax[1].set_xlabel("Speedup Factor (c)")
        # ax[1].invert_xaxis()







        # plot_success_over_kp(log_fn=can_commanded_20_hz, ax=ax, label="Can Commanded Poses (c=1)")
        # plot_success_over_kp(log_fn=can_reached_20_hz, ax=ax, label="Can Reached Poses (c=1)")
        # plot_success_over_kp(log_fn=can_commanded_40_hz, ax=ax, label="Can Commanded Poses (c=0.5)")
        # plot_success_over_kp(log_fn=can_reached_40_hz, ax=ax, label="Can Reached Poses (c=0.5)")

        plt.ylabel(" Tracking Error")
        # plt.xlabel("Gain (Kp)")
        ax.legend(frameon=True, edgecolor='black', facecolor='white',)
        # ax[1].legend(frameon=True, edgecolor='black', facecolor='white', )

        # plt.title("Effect of Controller Gain on Demo Replay At Different Speeds")
        # plt.tight_layout()
        plt.show()


main()
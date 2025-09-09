import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle



def get_success_rate_over_kwargs(data, ax=None, label=None):
    """

    Args:
        data: a dictionary of {val_kwarg : {rollouts, stats}}
        label:

    Returns:

    """

    success_data = {}
    for val in data:
        kwarg_data = data[val]
        avg_stats = kwarg_data['avg_stats']
        # rollouts = kwarg_data['rollouts']
        success_rate = avg_stats['Success_Rate']
        # X.append(val*10)
        success_data[val*10] = success_rate
        # y.append(success_rate)

    # sns.lineplot(x=X, y=y, label=label, linewidth=3)

    return success_data


log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/time_capped/noisy_eval_kp_250/noisy_eval_2025-01-16 18:54:39.610914.pkl"

with open(log_fn, "rb") as f:
    data = pickle.load(f)

high_kp_succcess =  get_success_rate_over_kwargs(data)

log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/time_capped/noisy_eval_kp_100/noisy_eval_2025-01-16 18:54:19.591218.pkl"

with open(log_fn, "rb") as f:
    data = pickle.load(f)


low_kp_success = get_success_rate_over_kwargs(data)



sns.set_palette('colorblind')

with plt.style.context("seaborn-v0_8-whitegrid"):
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"] = 1.25
    plt.rcParams["font.size"] = 15
    plt.rcParams['grid.color'] = 'black'
    plt.rcParams['grid.alpha'] = 0.5

    fig, ax = plt.subplots()

    sns.lineplot(high_kp_succcess, ax=ax, label="High Gain Controller")
    sns.lineplot(low_kp_success, ax=ax, label="Low Gain Controller")

    plt.ylabel("Success Rate")
    plt.xlabel("Noise Scale")
    # plt.title("Effect of increasing noise scale with high-gain vs low-gain controller")

    plt.legend()
    plt.show()
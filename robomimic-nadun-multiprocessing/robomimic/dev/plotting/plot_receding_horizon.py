import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from robomimic.dev.eval.eval_inconsistency import calculate_smoothness_from_vel

### Some constants

PLOT_FEATURE = 2 # which part of the pred to plot

def load_data(log_fn):
    with open(log_fn, 'rb') as f:
        data = pickle.load(f)

    return data

def plot_predictions(demo, label=None):
    kwargs = demo['kwargs']
    execute_n_actions = kwargs['execute_n_actions']
    inf_delay = kwargs['inf_delay']
    preds = demo['preds']

    # sns.set(context='notebook', style='whitegrid', font_scale=1.5)  # font_scale increases font size



    x_lists = [] # timesteps
    y_lists = [] # predicted values

    for i, pred in enumerate(preds):
        X = []
        y = []
        if i < 2:
            start_timestep = i * execute_n_actions
        else:
            start_timestep = (i - 1) * (execute_n_actions + inf_delay) + execute_n_actions

        for j in range(pred.shape[0]):
            X.append(start_timestep + j)
            y.append(pred[j, PLOT_FEATURE])
        x_lists.append(X)
        y_lists.append(y)

    for idx, X in enumerate(x_lists):
        # plt.plot(X, y_lists[idx], ls='None',  marker=f'${str(idx)}$', label=label)
        # plt.plot(X, y_lists[idx], label=label)
        sns.lineplot(x=X, y=y_lists[idx], label=label, linewidth=1)


def plot_executed_actions(demo, label=None):

    executed_actions = demo['executed_actions']

    X = []
    y = []

    for i, a in enumerate(executed_actions):
        X.append(i)
        y.append(a[PLOT_FEATURE])

    plt.plot(X, y, label=label)

def plot_velocity(demo, label=None):
    positions = []
    for obs_list in demo['obs']:
        positions.append(obs_list['robot0_eef_pos'][-1])
    positions = np.array(positions)

    data = calculate_smoothness_from_vel(positions=positions, return_all_data=True)

    outliers = data['outliers']
    X = [] # timesteps
    y = [] # velocities

    outlier_y = [] # velocities for outliers

    for i in range(data['vel'].shape[0]):
        X.append(i)
        y.append(data['vel'][i,PLOT_FEATURE])
        if i in outliers:
            outlier_y.append(data['vel'][i, PLOT_FEATURE])

    plt.plot(X, y, label=label)
    plt.plot(outliers, outlier_y, ls='None', marker='x', color='red')

    print()



def main():
    '''
    Do the actual plotting here


    '''

    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/square_image_action_horizon_16/20241211173329/logs/inpainting_2024-12-12 14:34:29.959133.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/noisy_eval_inpainting_2024-12-14 22:28:23.684674.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 17:13:10.727684.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 17:17:14.271470.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 17:22:35.669258.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 17:36:03.413872.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 17:41:16.438681.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 17:46:03.989480.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 18:02:22.827706.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 18:05:43.911655.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 18:10:10.053724.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 18:12:51.130993.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 18:46:47.201080.pkl"
    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/inpainting_eval/inpaint_eval_slow_2025-01-17 18:48:11.662051.pkl"
    # log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/blended_inpainting/inpaint_4_parabolic_2025-01-19 20:02:14.632119.pkl"

    log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/time_capped/inpainting_eval/inpaint_eval_slow_2025-01-17 18:48:11.662051.pkl"

    data = load_data(log_fn)
    demo = data['rollouts']['demo_2']

    # plot_velocity(demo)
    plot_predictions(demo)
    # plot_executed_actions(demo=demo, label="Executed actions")

    plt.title("Blended Inpainting (Without inpainting last step with previous prediction)")
    plt.xlabel("Time step")
    plt.legend()
    plt.show()

main()
import pickle
import matplotlib.pyplot as plt

### Some constants

PLOT_FEATURE = 2 # which part of the pred to plot

log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/absolute_osc/can_image/20240918173345/logs/receding_horizon_eval_2024-11-17 18:21:35.442668.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/absolute_osc/can_image/20240918173345/logs/eval_inpainting2024-11-24 00:23:22.812032.pkl"
log_fn  ="/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/absolute_osc/can_all_obs/20240918173401/logs/eval_inpainting2024-11-24 23:40:42.453576.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/absolute_osc/can_all_obs/20240918173401/logs/eval_inpainting2024-11-25 00:04:34.625537.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/normal_rollout_2024-11-03 17:48:14.826698.pkl"

log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/normal_rollout_2024-12-08 18:05:10.348727.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/guide_x0_3_steps_mse_loss_2024-12-08 16:05:18.769013.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/guide_x0_3_steps_exponential_loss_2024-12-08 16:11:57.031083.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/normal_rollout_2024-12-08 18:17:40.210602.pkl"
log_fn  ="/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/regular_inpainting_2024-12-08 18:03:52.240471.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/normal_rollout_2024-12-10 16:07:58.488338.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/action_horizon_experiments/absolute_osc/can_image_150_demos_action_horizon_16/20241101205408/logs/inpainting_2024-12-10 16:07:01.640655.pkl"

with open(log_fn, 'rb') as f:
    data = pickle.load(f)

demo = data['rollouts']['demo_0']
preds = demo['preds']
# execute_n_actions = demo["execute_n_actions"]
execute_n_actions = 4
inf_delay = 4

### Plot the predictions over time
x_lists = []
y_lists = []

for i, pred in enumerate(preds):
    # if (i % 2) == 1:
    #     continue
    X = []
    y = []
    if i < 2:
        start_timestep = i * execute_n_actions
    else:
        start_timestep = (i - 1) * (execute_n_actions + inf_delay) + execute_n_actions
    # start_timestep = i * execute_n_actions
    for j in range(pred.shape[0]):
        X.append(start_timestep+j)
        y.append(pred[j, PLOT_FEATURE])
    x_lists.append(X)
    y_lists.append(y)

for idx, X in enumerate(x_lists):
    # plt.plot(X, y_lists[idx], ls='None',  marker=f'${str(idx)}$')
    plt.plot(X, y_lists[idx])

plt.title("Receding Horizon predictions inpainting")
plt.ylabel("Predicted z position")
plt.xlabel("Timestep")
plt.ylim(0.85, 1.15)
plt.show()

print()
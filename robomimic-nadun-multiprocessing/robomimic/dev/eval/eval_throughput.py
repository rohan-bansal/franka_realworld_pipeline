import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# Some constants


def compute_throughput_for_rollouts(rollouts, unit_time=60, clip_failure_time=False, failure_time=15, time_cap_limit=50):
    """
    Computes throughput (number of successes per unit time) and time per succesful rollout
    Args:
        rollouts:
        unit_time: what unit of time to use
        clip_failure_time: If True, if a rollout is not successful, give it a fixed time penalty
        failure_time: what to clip the time of a failure to

    Returns:

    """

    total_sim_time = 0
    num_successes = 0
    num_rollouts = len(rollouts)

    for trial in rollouts:
        trial = rollouts[trial]
        if clip_failure_time:
            if not trial['success']:
                total_sim_time += failure_time
                continue

        # check if we cap the max time for a rollout, if so a trial is a failure if it isn't completed in this time
        if time_cap_limit > 0:
            if trial['total_sim_time_elapsed'] > time_cap_limit:
                total_sim_time += time_cap_limit
                continue

        if trial['success']:
            num_successes += 1
        total_sim_time += trial['total_sim_time_elapsed']

    time_per_success = total_sim_time/num_successes
    success_rate = num_successes/num_rollouts

    return unit_time/time_per_success, time_per_success, success_rate

def get_max_time_for_success(data):
    max_time = 0
    for kp in data:
        results = data[kp]
        rollouts = results['rollouts']
        for trial in rollouts:
            trial = rollouts[trial]
            if trial['success']:
                max_time = max(max_time, trial['total_sim_time_elapsed'])

    print(f"Max success time for task:{max_time}")



def eval_throughput_over_kp(log_fn, ax=None, label=None, filter_sucess=False):
    """

    Args:
        data: dictionary of {kp: results}

    Returns:

    """
    with open(log_fn, "rb") as f:
        data = pickle.load(f)


    X = []
    y = []
    for kp in data:
        results = data[kp]
        rollouts = results['rollouts']
        # sim_time_per_rollout = []
        # for trial in rollouts:
        #     trial = rollouts[trial]
        #     if FILTER_SUCCESS:
        #         if not trial['success']:
        #             continue
        #     else:
        #         if not trial['success']:
        #             sim_time_per_rollout.append(200)
        #             continue
        #     sim_time_per_rollout.append(trial['total_sim_time_elapsed'])

        throughput = compute_throughput_for_rollouts(rollouts, filter_success=filter_sucess)
        X.append(kp)
        y.append(throughput)

    plt.title("Effect of increasing controller gain on task throughput (Sim)")
    plt.xlabel("Kp")
    plt.ylabel("Average time per successful rollout (Simulated Seconds)")
    sns.lineplot(x=X, y=y,ax=ax, linewidth=2, label=label)

def get_success_rate_over_kp(log_fn):
    with open(log_fn, "rb") as f:
        data = pickle.load(f)

    kp_to_success_rate = {}
    for kp in data:
        results = data[kp]
        rollouts = results['rollouts']
        success = []
        for trial in rollouts:
            trial = rollouts[trial]
            success.append(trial['success'])


        kp_to_success_rate[kp] = sum(success) / len(success)


    return kp_to_success_rate

    # plt.title("Effect of increasing controller gain on task success rate (Sim)")
    # plt.xlabel("Kp")
    # plt.ylabel("Success Rate")
    # sns.lineplot(x=X, y=y,ax=ax, linewidth=2, label=label)



log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/speed_adaptive_models/absolute_osc/can_image_action_horizon_16_speed_adaptive/20250108003152/logs/eval_adaptive_speed_2025-01-12 18:41:49.895058.pkl"

log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/speed_adaptive_models/absolute_osc/square_image_action_horizon_16_speed_adaptive/20250108125500/logs/eval_adaptive_speed_fixed_2025-01-13 08:35:53.359289.pkl"
# log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/sim/speed_adaptive_models/absolute_osc/square_image_action_horizon_16_speed_adaptive/20250108125500/logs/eval_only_slow_2025-01-12 22:16:29.999622.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/normal_eval/normal_eval_slow_2025-01-16 03:55:22.298872.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/window_slowdown/window_slowdown_eval_2025-01-16 01:06:18.195522.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/delta_action_models/can_image_action_horizon_16_delta_action/20250115160733/logs/aggregated_actions/aggregate_10_cm2025-01-16 22:10:29.619525.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/delta_action_models/can_image_action_horizon_16_delta_action/20250115160733/logs/aggregated_actions/aggregate_10_cm2025-01-16 22:10:29.619525.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/rss/sim/delta_action_models/can_image_action_horizon_16_delta_action/20250115160733/logs/aggregated_actions/aggregate_10_cm_2025-01-16 22:28:08.986237.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/lift_image_action_horizon_16_speed_adaptive/20250114170107/logs/ablations/without_slowdown_2025-01-21 21:21:34.827384.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/lift_image_action_horizon_16_speed_adaptive/20250114170107/logs/ablations/without_consistency_guiding_2025-01-21 22:47:38.610671.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/ablations/without_slowdown_2025-01-21 20:27:33.808861.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/ablations/without_consistency_guiding_2025-01-21 22:47:38.441447.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/square_image_action_horizon_16_speed_adaptive/20250115015113/logs/ablations/without_slowdown_2025-01-21 20:44:13.838813.pkl"
log_fn = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/square_image_action_horizon_16_speed_adaptive/20250115015113/logs/ablations/without_consistency_guiding_2025-01-21 22:47:38.441348.pkl"


dict_of_logs = {
    # "lift_baseline_dp": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/absolute_osc/lift_image_action_horizon_16_baseline_absolute_osc/20250117225014/logs/normal_rollout_time_uncapped_2025-01-21 20:48:31.464407.pkl",
    # "can_baseline_dp" : "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/absolute_osc/can_image_action_horizon_16_speed_baseline_absolute_osc/20250117225006/logs/normal_rollout_time_uncapped_2025-01-21 20:37:30.923929.pkl",
    # "square_baseline_dp": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/absolute_osc/square_image_action_horizon_16_baseline_absolute_osc/20250117225113/logs/normal_rollout_time_uncapped_2025-01-21 20:53:31.490871.pkl",
    #
    # "lift_scaled_delta": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/delta_action_models/lift_image_action_horizon_16_delta_action/20250115160740/logs/scaled_delta/scale_15_uncapped_2025-01-21 21:17:34.886435.pkl",
    # "can_scaled_delta": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/delta_action_models/can_image_action_horizon_16_delta_action/20250115160733/logs/scaled_delta/scale_15_uncapped_2025-01-21 21:17:33.539923.pkl",
    # "square_scaled_delta": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/delta_action_models/square_image_action_horizon_16_delta_actions/20250115160749/logs/scaled_delta/scale_15_uncapped_2025-01-21 21:17:34.886535.pkl",
    #
    # "lift_aggregated_actions": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/delta_action_models/lift_image_action_horizon_16_delta_action/20250115160740/logs/aggregated_actions/aggregate_5cm_uncapped_2025-01-21 21:08:31.416046.pkl",
    # "can_aggregated_actions": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/delta_action_models/can_image_action_horizon_16_delta_action/20250115160733/logs/aggregated_actions/aggregate_5cm_uncapped_2025-01-21 20:56:31.734895.pkl",
    # "square_aggregated_actions": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/delta_action_models/square_image_action_horizon_16_delta_actions/20250115160749/logs/aggregated_actions/aggregate_5cm_uncapped_2025-01-21 21:03:34.065695.pkl",


    # "lift_without_slowdown": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/lift_image_action_horizon_16_speed_adaptive/20250114170107/logs/ablations/without_slowdown_2025-01-21 21:21:34.827384.pkl",
    # "can_without_slowdown": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/ablations/without_slowdown_2025-01-21 20:27:33.808861.pkl",
    # "square_without_slowdown": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/square_image_action_horizon_16_speed_adaptive/20250115015113/logs/ablations/without_slowdown_2025-01-21 20:44:13.838813.pkl",
    #
    #
    # "lift_without_consistency": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/lift_image_action_horizon_16_speed_adaptive/20250114170107/logs/ablations/without_consistency_guiding_2025-01-21 22:47:38.610671.pkl",
    # "can_without_consistency": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/ablations/without_consistency_guiding_2025-01-21 22:47:38.441447.pkl",
    # "square_without_consistency": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/square_image_action_horizon_16_speed_adaptive/20250115015113/logs/ablations/without_consistency_guiding_2025-01-21 22:47:38.441348.pkl"

    # "can_without_consistency": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/can_image_action_horizon_16_speed_adaptive/20250114173246/logs/quick_test_2025-01-25 01:43:07.415693.pkl"

    # "stack_dp_slow" : "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/baseline_dp/stack_image_action_horizon_16_baseline_dp_commanded_pose/20250126041316/logs/eval_normal_test_2025-01-27 00:01:33.686171.pkl",
    # "stack_dp_fast" : "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/baseline_dp/stack_image_action_horizon_16_baseline_dp_commanded_pose/20250126041316/logs/eval_fast_test_2025-01-27 00:35:58.780006.pkl",
    # "stack_adaptive": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/speed_adaptive_models/stack_image_speed_adaptive/20250126145935/logs/quick_test_2025-01-26 23:47:05.105756.pkl",
    
    "coffee_adaptive": "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/cfg/speed_adaptive_models/coffee_image_action_horizon_16_cfg_speed_adaptive/20250127013055/logs/quick_test_2025-01-27 17:58:05.476479.pkl",
    "coffee_baseline_dp" : "/media/nadun/Data/phd_project/robomimic/bc_trained_models/rss/sim/baseline_dp/coffee_image_action_horizon_16_baseline_dp_commanded_pose/20250127005553/logs/eval_normal_test_2025-01-27 11:52:29.853268.pkl"

}


for model in dict_of_logs:
    log_fn = dict_of_logs[model]
    with open(log_fn, "rb") as f:
        data = pickle.load(f)

    rollouts = data['rollouts']


    throughput, time_per_successful_rollout, success_rate = compute_throughput_for_rollouts(rollouts, clip_failure_time=True)

    print(f"Model {model} has success_rate: {success_rate} with throughput: {throughput} and time per success: {time_per_successful_rollout}")




# with plt.style.context("seaborn-v0_8-whitegrid"):
#     plt.rcParams["axes.edgecolor"] = "0.15"
#     plt.rcParams["axes.linewidth"]  = 1.25
#     plt.rcParams["font.size"] = 15
#     # params = plt.rcParams
#     fig, ax = plt.subplots()
#     plot_success_rate_over_kp(log_fn_slow, ax=ax, label="Can Slow Execution")
#     plot_success_rate_over_kp(log_fn_fast, ax=ax, label="Can Fast Execution")
#     plot_success_rate_over_kp(log_fn_commanded, ax=ax, label="Can Commanded Fast")
#     plt.show()


print()
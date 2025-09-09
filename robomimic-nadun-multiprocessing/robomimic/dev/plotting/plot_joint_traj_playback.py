import pandas as pd
import matplotlib.pyplot as plt
import nexusformat.nexus as nx
import h5py
from collections import defaultdict

fp_gripper_heuristic = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/can_image_diffusion_policy/20240620123559/logs/multi_eval_gripper_check.xlsx"
fp_no_gripper_heuristic = "/media/nadun/Data/phd_project/robomimic/bc_trained_models/diffusion_policy/can_image_diffusion_policy/20240620123559/logs/multi_eval.xlsx"

def get_limit_and_success_rates(fp):
    df = pd.read_excel(fp)

    act_limits = df["action_magnitude_limit"].unique().tolist()

    x = []
    y = []
    h = []

    for limit in act_limits:
        success_rates = df[df["action_magnitude_limit"] == limit]['Success_Rate']
        mean_success_rate = success_rates.mean()
        x.append(max(0, limit*0.05))
        y.append(mean_success_rate)

        # next filter horizons
        horizons = df[(df["action_magnitude_limit"] == limit) & (df["Success_Rate"] == 1)]['Horizon']
        h.append(horizons.mean())
    return x, y, h



# kp_list = []
# limit_to_stats = defaultdict(list)
# for limit in act_limits:
#     df_act_limit = df[df["action_magnitude_limit"] == limit]
#     x.append(max(0, limit*0.05))
#     for kp in df_act_limit["kp"].unique().tolist():
#         kp_list.append(kp)
#         success_rates = df_act_limit[df_act_limit["kp"] == kp]["Success_Rate"]
#         mean_success_rate = success_rates.mean()
#         limit_to_stats[limit].append((kp, mean_success_rate))

    # next filter horizons
    # horizons = df[(df["action_magnitude_limit"] == limit) & (df["Success_Rate"] == 1)]['Horizon']
    # h.append(horizons.mean())


# Tot = len(limit_to_stats.keys()) - 1
# Cols = 2
# Rows = Tot // Cols
# if Tot % Cols != 0:
#     Rows += 1


# Position = range(1,Tot + 1)
#
# fig = plt.figure(1)
# fig.supylabel("Success Rate")
# fig.supxlabel("kp")
# fig.suptitle("Effect of kp on success rate")
# k = 0
# for key in limit_to_stats.keys():
#     if key == -1:
#         continue
#     stats = limit_to_stats[key]
#     x = [stat[0] for stat in stats]
#     y = [stat[1] for stat in stats]
#     ax = fig.add_subplot(Rows, Cols, Position[k])
#     ax.set_title(f"Maximum action magnitude: {key * 5} cm")
#     ax.plot(x, y)
#
#     k += 1
# plt.show()
f, (ax1, ax2) = plt.subplots(1, 2)

x, y, h = get_limit_and_success_rates(fp_gripper_heuristic)

# first plot action magnitude vs success rate
ax1.plot(x, y, label="with gripper heuristic")
ax1.set_ylim(bottom=0)
ax1.set_ylabel("Success Rate")
f.supxlabel("Maximum Action Magnitude (cm)")
ax1.set_title("Effect of action magnitude on success rate")

# next plot action magnitude vs horizon
ax2.plot(x, h, label="with gripper heuristic")
ax2.set_ylabel("# Env Steps (actions)")
ax2.set_title("Effect of action magnitude on number env steps")

x, y, h = get_limit_and_success_rates(fp_no_gripper_heuristic)
ax1.plot(x, y, label="without gripper heuristic")
ax2.plot(x, h, label="without gripper heuristic")



plt.legend()
plt.show()
print()
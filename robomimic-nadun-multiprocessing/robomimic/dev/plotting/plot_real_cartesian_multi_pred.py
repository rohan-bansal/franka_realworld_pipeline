import pickle
import matplotlib.pyplot as plt

### Some constants

PLOT_FEATURE = 2 # which part of the pred to plot
Y_LIM = (0.20, 0.70)
fn = "/home/robot-aiml/ac_learning_repos/experiment_logs/pick_cube_10_24_filtered/cartesian_actions_framestack_2_image_only_run_8_1x.pkl"

with open(fn, 'rb') as f:
    data = pickle.load(f)

demo = data['demo_0']
preds = demo['actions']
execute_n_actions = demo["run_actions"]
obs_ee_pose = demo["obs_ee_pose"]


### Plot the predictions over time
x_lists = []
y_lists = []
obs_x_list = []
obs_y_list = []

for i, pred in enumerate(preds):
    X = []
    y = []
    start_timestep = i*execute_n_actions

    obs_x_list.append(start_timestep)
    obs_y_list.append(obs_ee_pose[i][PLOT_FEATURE])
    for j in range(pred.shape[0]):
        X.append(start_timestep+j)
        y.append(pred[j, PLOT_FEATURE])
    x_lists.append(X)
    y_lists.append(y)

for idx, X in enumerate(x_lists):
    # plt.plot(X, y_lists[idx], ls='None', marker=f'${str(idx)}$',
    #          color='orange', markersize=12, label="prediction")

    plt.plot(X, y_lists[idx], label="prediction")

    plt.plot(obs_x_list[idx], obs_y_list[idx], ls='None', marker=f'${str(idx)}$',
             color='red', markersize=16, label="obs z position of robot")


plt.ylabel("Predicted z position")
plt.xlabel("Timestep")

#Plotting stuff
handles, labels = plt.gca().get_legend_handles_labels()
# Remove duplicates by converting to a dictionary (which removes duplicates by key)
by_label = dict(zip(labels, handles))
# Create the legend with unique labels
plt.legend(by_label.values(), by_label.keys())
plt.ylim(Y_LIM)


plt.show()

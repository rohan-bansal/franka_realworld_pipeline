import h5py
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from numpy import polyfit


ORIGINAL_ACTION_KEY = "consistent_absolute_actions"
SPLINE_ACTION_KEY = "absolute_actions_cubic_spline_32"
SPLINE_LENGTH = 32
SEGMENT_LENGTH = 4


def add_polyfit_spline(demo_fn):
    demo_file = h5py.File(demo_fn, 'a')
    demos = demo_file['data']
    counter = 0
    for demo_name in demos:

        if (counter % 10) == 0:
            print(f"Processing demo: {counter}")
        counter += 1

        demo = demos[demo_name]

        actions = demo[ORIGINAL_ACTION_KEY][:]
        spline_actions = []

        for i in range(actions.shape[0] - SPLINE_LENGTH):

            act_segment = actions[i:i+SPLINE_LENGTH]
            x = [x for x in range(SPLINE_LENGTH)]

            coeffs = [] # per act dim coeffs of the spline
            # Fit a spline per dimension of action
            for j in range(act_segment.shape[1]):
                y = act_segment[:, j]

                c = polyfit(x, y, 3)
                coeffs.append(c)
            coeffs = np.array(coeffs)
            spline_actions.append(coeffs)

        for i in range(actions.shape[0] - len(spline_actions)):
            spline_actions.append(coeffs)
        if SPLINE_ACTION_KEY in demo:
            del demo[SPLINE_ACTION_KEY]
        demo.create_dataset(SPLINE_ACTION_KEY, data=np.array(spline_actions))

def add_cubic_spline(demo_fn):
    demo_file = h5py.File(demo_fn, 'a')
    demos = demo_file['data']
    counter = 0
    for demo_name in demos:

        if (counter % 10) == 0:
            print(f"Processing demo: {counter}")
        counter += 1

        demo = demos[demo_name]

        actions = demo[ORIGINAL_ACTION_KEY][:]
        # We have to repeat the last action to ensure that we have splines for the entire trajectory
        last_act = actions[-1, :]
        repeated_act = np.tile(last_act, (SPLINE_LENGTH, 1))
        actions = np.vstack([actions, repeated_act])
        spline_actions = []

        for i in range(actions.shape[0] - SPLINE_LENGTH):

            act_chunk = actions[i:i + SPLINE_LENGTH]
            x = [x for x in range(0, SPLINE_LENGTH, SEGMENT_LENGTH)]
            x.append(SPLINE_LENGTH - 1)

            coeffs = []  # per act dim coeffs of the spline
            # Fit a spline per dimension of action
            for j in range(act_chunk.shape[1]):
                y = [act_chunk[n, j] for n in x]

                c = CubicSpline(x, y, 3)
                coeffs.append(c.c)
            coeffs = np.array(coeffs).reshape(-1, order='C')
            spline_actions.append(coeffs)
        if SPLINE_ACTION_KEY in demo:
            del demo[SPLINE_ACTION_KEY]
        demo.create_dataset(SPLINE_ACTION_KEY, data=np.array(spline_actions))


demo_fn = "/nethome/nkra3/flash7/phd_project/robomimic-nadun/datasets/lift/ph/all_obs_v141.hdf5"

# demo_fn = "/media/nadun/Data/phd_project/robomimic/datasets/can/ph/all_obs_v141.hdf5"
add_cubic_spline(demo_fn)








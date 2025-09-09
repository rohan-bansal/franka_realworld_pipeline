import pickle
import numpy as np

def get_success_over_200_trials(data=None, log_fn=None):
    """
    Data is a dict of {val: reward for 200 trials}
    """

    if data is None and log_fn is None:
        raise Exception("Must pass data or log_fn")

    if log_fn is not None:
        with open(log_fn, "rb") as f:
            data = pickle.load(f)

    for val in data:
        data[val] = data[val] / 200

    return data

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

def get_success_rate_over_kwargs(data=None, log_fn=None, return_array=False):
     
    if data is None and log_fn is None:
        raise Exception("Must pass data or log_fn")

    if log_fn is not None:
        with open(log_fn, "rb") as f:
            data = pickle.load(f)

    # TODO delete multiplier from val
    success_data = {}
    for val in data:
        kwarg_data = data[val]
        
        if not return_array:
            avg_stats = kwarg_data['avg_stats']
            success_rate = avg_stats['Success_Rate']
            success_data[val*10] = success_rate
        else:
            rollouts = kwarg_data['rollouts']
            success_array = []
            for demo in rollouts:
                success_array.append(rollouts[demo]['success'])
            success_data[val*10] = np.array(success_array)



    return success_data
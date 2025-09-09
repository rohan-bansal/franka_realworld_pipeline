import numpy as np


def calculate_inconsistency(pred1, pred2, action_dim=None, weighted=False, beta=0.5):
    """
    Calculate the inconsistency (error) between two overlapping segments of actions.
    Similar to https://github.com/YuejiangLIU/bid_diffusion/blob/main/diffusion_policy/sampler/single.py
    Args:
        pred1: action sequence of shape (T, D) where T is the timesteps, and D is the action dimension
        pred2:
        weighted: whether to weight the inconsistency, initial errors are weighted more
        beta:

    Returns:

    """

    if action_dim is None:
        action_dim= pred1.shape[-1]

    pred1 = pred1[..., :action_dim]
    pred2 = pred2[..., :action_dim]

    diff = pred1 - pred2
    THIS IS WRONG
    dist = np.linalg.norm(diff)

    if weighted:
        # Calculate weights
        weights = np.array([beta ** i for i in range(pred1.shape)])
        weights = weights / weights.sum()

        # apply weights to Euclidean Distance
        dist = dist*weights

    return np.sum(dist)



def compute_euclidean_distance(demo, disregard_inpainted_actions = True):
    """
    Compute the euclidean distance
    """
    # Parse settings
    kwargs = demo['kwargs']
    inf_delay = kwargs['inf_delay']
    execute_n_actions = kwargs['execute_n_actions']
    
    preds = demo['preds']

    curr_prediction_index = execute_n_actions

    for i in range(len(preds) - 1):
        if not disregard_inpainted_actions:
            if i > 0:
                curr_prediction_index = execute_n_actions + inf_delay
            # This is the part of the current prediction that will overlap with the beginning of the next
            curr_pred_overlap = preds[i][curr_prediction_index:]
            # This is part of the next prediction that overlaps with the current
            next_pred_overlap = preds[i + 1][:curr_pred_overlap.shape[0]]


        elif disregard_inpainted_actions: # we only consider the overlapping sections that are not inpainted
            if i == 0:
                curr_prediction_index = execute_n_actions + inf_delay
            else:
                curr_prediction_index = execute_n_actions + inf_delay*2

            curr_pred_overlap = preds[i][curr_prediction_index:]
            # Disregard the section at the beginning of the next prediction, which is inpainted
            next_pred_overlap = preds[i+1][inf_delay:inf_delay+curr_pred_overlap.shape[0]]

        # Select only the translation parts
        curr_pred_overlap = curr_pred_overlap[:, :3]
        next_pred_overlap = next_pred_overlap[:, :3]

        # Calculate euclidean distance for overlapping section
        euclidean_distance = np.sum(np.sqrt(np.sum(np.square(curr_pred_overlap - next_pred_overlap), axis=1)))
    
    return euclidean_distance
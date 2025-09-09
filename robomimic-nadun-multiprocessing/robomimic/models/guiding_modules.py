import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyClassifier(nn.Module):
    def __init__(self, classifier_scale=10.0):

        # default classifier weight to apply to classifier guidance if user does not pass in classifier guidance to
        # get_classifier_guidance()
        # TODO: complete self.classifier
        self.classifier = nn.Sequential()
        self.default_classifier_scale = classifier_scale

    def forward(self, x, timestep, cond, device):
        return torch.tensor(1, device=device)

    def get_guidance(self, x, classifier_scale = None, classifier_kwargs=None):
        '''
        Modeled after cond_fn in classfier_sample.py from this repo: https://github.com/openai/guided-diffusion.git
        Args:
            x:
            classifier_scale:
            classifier_kwargs:

        Returns:

        '''
        assert classifier_kwargs is not None, "need to pass in classifier kwargs before calling this fn"

        if classifier_scale is None:
            classifier_scale = self.default_classifier_scale

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            y = classifier_kwargs['y']
            device = x_in.device
            logits = self.forward(x_in, classifier_kwargs['timestep'], classifier_kwargs["cond"], device)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            sum_log_probs = selected.sum()
            return torch.autograd.grad(sum_log_probs, x_in)[0] * classifier_scale


class ConsistencyGuider(nn.Module):
    def __init__(self, classifier_scale=1.0):

        self.loss = nn.MSELoss()
        self.default_classifier_scale = classifier_scale

    def get_guidance(self, x, classifier_scale = None, classifier_kwargs=None):
        assert classifier_kwargs is not None, "need to pass in classifier kwargs before calling this fn"

        if classifier_scale is None:
            classifier_scale = self.default_classifier_scale

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True) # this is the current prediction we want to guide
            y = classifier_kwargs['y'] # this is what we want to make the prediction consistent with
            x_selected = x_in[:y.shape[0]]
            loss = self.loss(x_selected, y)
            loss.backward()

            return x_in.grad * classifier_scale

# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         byzantine/attacks.py
# Description:  Byzantine Attack classes
# ===========================================================================
from collections import OrderedDict

import torch

from public_config import public_datasetAssignmentDict
from utilities import Utilities as Utils
import os

#### Attack Base Class
class NoAttack:
    """Attack base class"""

    def __init__(self, **kwargs):
        self.config = kwargs['config']
        self.clients = kwargs['clients']
        self.runner = kwargs['runner_instance']

    def get_perturbed_client_models(self, **kwargs):
        """Called before the model is communicated. Defaults to returning the individual client models (unchanged)."""
        return Utils.get_client_models(self.clients)

    def get_perturbed_client_predictions(self, **kwargs):
        """Called before the predictions are communicated. Defaults to returning the individual client predictions (unchanged)."""
        return self.runner.get_client_predictions(mode='train')


class ParameterRandomVector(NoAttack):
    """Random vector attack"""

    @torch.no_grad()
    def perturb_byzantine_model(self, client_state_dict: OrderedDict) -> OrderedDict:
        """Takes the state_dict of a client and perturbs it."""
        client_state_dict = client_state_dict.copy()
        for key in client_state_dict:
            client_state_dict[key] = torch.randn_like(client_state_dict[key].float())
        return client_state_dict

    def get_perturbed_client_models(self, **kwargs):
        client_model_list = []
        for client in self.clients:
            client_state_dict = client.model.state_dict()
            if client.is_byzantine:
                client_state_dict = self.perturb_byzantine_model(client_state_dict)
            client_model_list.append(client_state_dict)

        return client_model_list

class ParameterRandomVectorScaled(ParameterRandomVector):
    """Random vector attack but scale to have the same L2 norm as the original model."""

    @torch.no_grad()
    def perturb_byzantine_model(self, client_state_dict: OrderedDict) -> OrderedDict:
        """Takes the state_dict of a client and perturbs it."""
        client_state_dict = client_state_dict.copy()
        for key in client_state_dict:
            p_old = client_state_dict[key].float()
            old_norm = torch.norm(p_old)
            client_state_dict[key] = torch.randn_like(p_old)
            new_norm = torch.norm(p_old)
            assert new_norm > 0
            client_state_dict[key] = p_old * (old_norm / new_norm)

        return client_state_dict


class PredictionNaiveSignFlip(NoAttack):
    """Naive prediction sign flip attack, just uses a random one-hot vector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        for client_idx, client_predictions in enumerate(client_prediction_list):
            if self.clients[client_idx].is_byzantine:
                random_logits = torch.randn_like(client_predictions)
                random_predictions = torch.argmax(random_logits, dim=1)
                client_prediction_list[client_idx] = torch.nn.functional.one_hot(random_predictions,
                                                                                 num_classes=client_predictions.shape[
                                                                                     1]).float()
        return client_prediction_list


class PredictionFixedSignFlip(NoAttack):
    """Fixed prediction sign flip attack, just uses a fixed one-hot vector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        fixed_prediction = torch.zeros_like(client_prediction_list[0])
        byz_prediction  = fixed_prediction
        byz_prediction[:,0] = 1.
        if self.config['sample_attack_frac'] not in [None, 'None', 'none']:
            byz_idx = torch.randperm(fixed_prediction.shape[0])[:int(fixed_prediction.shape[0] * self.config['sample_attack_frac'])]

        for client_idx in range(len(client_prediction_list)):
            if self.clients[client_idx].is_byzantine:
                if self.config['sample_attack_frac'] not in [None, 'None', 'none']:
                    client_pred = client_prediction_list[client_idx]
                    client_pred[byz_idx,:] = byz_prediction[byz_idx,:]
                    client_prediction_list[client_idx] = client_pred
                else:
                    client_prediction_list[client_idx] = byz_prediction

        return client_prediction_list


class PredictionAdversarialSignFlip(PredictionNaiveSignFlip):
    """Byzantine clients put full emphasis (one hot) on second most likely class of benign clients."""

    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        # Get the list of predictions, but only the benign ones
        honest_client_predictions = [client_pred for client_idx, client_pred in enumerate(client_prediction_list)
                                     if not self.clients[client_idx].is_byzantine]
        avg_honest_client_predictions = torch.mean(torch.stack(honest_client_predictions, dim=0), dim=0)
        for client_idx, client_predictions in enumerate(client_prediction_list):
            if self.clients[client_idx].is_byzantine:
                # Get the second most likely class and set full probability to that class
                # Index 1 corresponds to the second most likely class
                second_most_likely_class = torch.topk(avg_honest_client_predictions, k=2, dim=1).indices[:, 1]

                client_prediction_list[client_idx] = torch.nn.functional.one_hot(second_most_likely_class,
                                                                                 num_classes=client_predictions.shape[
                                                                                     1]).float()
        return client_prediction_list


class CPA(PredictionNaiveSignFlip):
    """Byzantine clients put full emphasis (one hot) on least corrrelated class."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.config['public_ds'] in [None, 'none', 'None']:
            public_ds = public_datasetAssignmentDict[self.config['dataset']]
        else:
            public_ds = self.config['dataset']
        cpa_tensor_path = os.path.join('byzantine', 'cpa_info', f"{public_ds}-cov.pt")
        self.cpa_tensor = torch.load(cpa_tensor_path, map_location=self.config.device)

    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        honest_client_predictions = [client_pred for client_idx, client_pred in enumerate(client_prediction_list)
                                     if not self.clients[client_idx].is_byzantine]
        honest_client_predictions = torch.stack(honest_client_predictions, dim=0)
        mean_honest_predictions = torch.mean(honest_client_predictions, dim=0)
        honest_max_pred = torch.argmax(mean_honest_predictions, dim=1)

        if self.config['hips'] == True:
            # Get the covariance vector corresponding to the honest_max_pred
            cpa_tensor = self.cpa_tensor[honest_max_pred, :]

            selected_vertices = torch.mul(cpa_tensor, honest_client_predictions).sum(dim=2).argmin(dim=0)
            byz_predictions = honest_client_predictions[selected_vertices, torch.arange(selected_vertices.size()[0]), :]
        else:
            cpa_tensor = self.cpa_tensor.argmin(dim=0)
            byz_label = cpa_tensor[honest_max_pred]
            byz_predictions = torch.nn.functional.one_hot(byz_label, num_classes=client_prediction_list[0].shape[1]).float()

        if self.config['sample_attack_frac'] not in [None, 'None', 'none']:
            assert not self.config['hips'], "Cannot sample attack fraction when hips is True."
            p_honest = 1. - self.config['sample_attack_frac']
            honest_idx = torch.randperm(byz_label.shape[0])[:int(byz_label.shape[0] * p_honest)]
            byz_predictions[honest_idx,:] = mean_honest_predictions[honest_idx,:]

        for client_idx, client_predictions in enumerate(client_prediction_list):
            if self.clients[client_idx].is_byzantine:
                # Get the least likely class and set full probability to that class
                client_prediction_list[client_idx] = byz_predictions
        return client_prediction_list


class CELMAX(PredictionNaiveSignFlip):
    """Byzantine clients put full emphasis (one hot) on the class that is least likely when averaging all honest clients."""

    @torch.no_grad()
    def get_perturbed_client_predictions(self, **kwargs):
        client_prediction_list = self.runner.get_client_predictions(mode='train')
        honest_client_predictions = [client_pred for client_idx, client_pred in enumerate(client_prediction_list)
                                     if not self.clients[client_idx].is_byzantine]
        honest_client_predictions = torch.stack(honest_client_predictions, dim=0)
        mean_honest_predictions = torch.mean(honest_client_predictions, dim=0)

        if self.config['hips'] == True:
            alpha = float(self.config['n_byzantine_clients']) / float(self.config['n_clients'])
            potential_predictions = alpha * mean_honest_predictions.unsqueeze(0) + (1. - alpha) * honest_client_predictions
            deviations = torch.sum(-1. * mean_honest_predictions.unsqueeze(0) * torch.log(potential_predictions),dim=2)
            argmax_deviations = torch.argmax(deviations, dim=0)
            byz_predictions = honest_client_predictions[argmax_deviations,torch.arange(argmax_deviations.size()[0]),:]
        else:
            del honest_client_predictions
            honest_client_least_likely_predictions = torch.argmin(mean_honest_predictions, dim=1)
            byz_predictions = torch.nn.functional.one_hot(honest_client_least_likely_predictions,
                                                                                num_classes=
                                                                                client_prediction_list[0].shape[
                                                                                    1]).float()
        if self.config['sample_attack_frac'] not in [None, 'None', 'none']:
            p_honest = 1. - self.config['sample_attack_frac']
            honest_idx = torch.randperm(byz_predictions.size()[0])[:int(byz_predictions.size()[0] * p_honest)]
            byz_predictions[honest_idx,:] = mean_honest_predictions[honest_idx,:]
        del mean_honest_predictions
        for client_idx, client_predictions in enumerate(client_prediction_list):
            if self.clients[client_idx].is_byzantine:
                client_prediction_list[client_idx] = byz_predictions
        return client_prediction_list

# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         byzantine/defences.py
# Description:  Byzantine Defence classes
# ===========================================================================
import torch
from tqdm.auto import tqdm

from utilities import Utilities as Utils


#### Defence Base Class
class NoDefence:
    """Defence base class"""

    def __init__(self, **kwargs):
        self.config = kwargs['config']
        self.clients = kwargs['clients']
        self.runner = kwargs['runner_instance']

    def get_aggregated_model(self, client_model_list: list):
        return Utils.average_client_models(client_model_list)

    def get_aggregated_predictions(self, client_predictions_list: list):
        avg_pred = Utils.average_client_predictions(client_predictions_list=client_predictions_list,
                                                output_type='softmax')
        pred_stack = torch.stack(client_predictions_list).permute(1,2,0)
        outlier_scores = torch.linalg.norm(pred_stack - avg_pred.unsqueeze(-1), dim=1)
        mean_outlier_scores = torch.mean(outlier_scores, dim=0)
        return avg_pred, mean_outlier_scores


class PredictionMedian(NoDefence):
    """PredictionMedian defence"""

    def get_aggregated_predictions(self, client_predictions_list: list):
        tensor_stack = torch.stack(client_predictions_list)
        median_probabilities = torch.quantile(tensor_stack, 0.5, dim=0, interpolation='nearest')

        # Ensure that the median probabilities sum to 1
        median_probabilities = median_probabilities / median_probabilities.sum(dim=1, keepdim=True)
        outlier_scores = torch.linalg.norm(tensor_stack - median_probabilities, dim=2)
        mean_outlier_scores = torch.mean(outlier_scores, dim=1)
        return median_probabilities, mean_outlier_scores

class PredictionGeoMedian(NoDefence):
    """PredictionGeoMedian defence"""

    def get_aggregated_predictions(self, client_predictions_list: list):
        pred_stack = torch.stack(client_predictions_list)  # shape: [n_clients, pub_ds_size, n_classes]
        geomedian, outlier_scores = Utils.geomedian(pred_stack)

        # ensure all probs are non-negative
        geomedian = torch.clamp(geomedian, min=0.)
        # normalize to ensure we are in the simplex
        geomedian = torch.nn.functional.normalize(geomedian, p=1.,dim=1)
        assert torch.all(geomedian >= 0), 'GM is unstable, neg probabilities'
        assert torch.allclose(torch.sum(geomedian,dim=1),torch.tensor([1.],device=self.clients[0].device)), 'GM unstable, sum>=1'

        mean_outlier_scores = torch.mean(outlier_scores, dim=0)

        return geomedian, mean_outlier_scores

class PredictionFilter(NoDefence):
    """PredictionFilter defence"""


    @torch.no_grad()
    def get_aggregated_predictions(self, client_predictions_list: list):
        if self.config["filter_threshold"] is not None:
            assert self.config["filter_quantile"] is None, "Cannot choose both filter_threshold and filter_quantile"
        else:
            assert self.config["filter_quantile"] is not None, "choose either filter_threshold, or filter_quantile"

        pred_stack = torch.stack(client_predictions_list) # shape: [n_clients, pub_ds_size, n_classes]

        # compute outlier scores
        outlier_scores = Utils.filter_outlier_scores(pred_stack)

        # compute filtered mean
        if self.config["filter_threshold"] is not None:
            threshold = self.config["filter_threshold"]
        else:
            threshold = torch.quantile(torch.abs(outlier_scores), self.config["filter_quantile"], dim=1, interpolation='higher')
        filtered_mean = Utils.filtered_mean(pred_stack, outlier_scores, threshold)
        mean_outlier_scores = torch.mean(torch.abs(outlier_scores).squeeze(-1), dim=0)
        return filtered_mean, mean_outlier_scores


class Cronus(NoDefence):
    """
    Cronus defence:
    Filter out (alpha N / 2) of predictions, then recompute outlier scores and filter out (alpha N / 2) again
    """

    @torch.no_grad()
    def get_aggregated_predictions(self, client_predictions_list: list):
        pred_stack = torch.stack(client_predictions_list) # shape: [n_clients, pub_ds_size, n_classes]
        max_n_byz = Utils.get_highest_byz(self.config["n_clients"])
        quantile = 1 - (1. * max_n_byz / (2. * self.config["n_clients"]))


        # 1. filtering of predictions
        outlier_scores = Utils.filter_outlier_scores(pred_stack)
        k = int(quantile * pred_stack.size()[0])
        if k <1: k = 1
        _, idx = torch.topk(torch.abs(outlier_scores), k, dim=1,largest=False)  # return k smallest
        reduced_pred = torch.zeros(k, pred_stack.size()[1], pred_stack.size()[2])
        for i in range(pred_stack.size()[1]):
            nonzero = idx[i,:,0]
            reduced_pred[:,i,:] = pred_stack[nonzero,i,:]

        # 2. filtering
        outlier_scores = Utils.filter_outlier_scores(reduced_pred)
        threshold = torch.quantile(torch.abs(outlier_scores), quantile, dim=1, interpolation='higher')
        filtered_mean = Utils.filtered_mean(reduced_pred, outlier_scores, threshold)
        outlier_scores = torch.linalg.norm(pred_stack.permute(1,2,0) - filtered_mean.unsqueeze(-1).to(pred_stack.device),dim=1)
        mean_outlier_scores = torch.mean(outlier_scores, dim=0)
        return filtered_mean, mean_outlier_scores

def choose_aggregation_expweights(aggregation_method):
    class ExpWeights(aggregation_method):
        """
        Maintain weights for each client based on outlier score in each communication round
        """
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            assert self.config['exp_stepsize'] is not None, 'speicify stepsize for exp weights'
            self.w = torch.ones(self.config['n_clients']).to(self.clients[0].device)

        def update_weights(self, loss):
            self.w = self.w * torch.exp(-1. * self.config['exp_stepsize'] * loss)
            assert torch.all(self.w >= 0), f'weights need to be non-negative, here: {self.w}'

        def weighted_sum(self, pred_stack):
            p = torch.nn.functional.normalize(self.w, p=1.,dim=0)
            assert torch.all(p >= 0.), f'probabilities have to be >= 0, here: {p}'
            assert torch.all(p <= 1.), f'probabilities have to be <=1, here: {p}'
            assert torch.isclose(torch.abs(torch.sum(p)), torch.tensor([1.]).to(p.device)), f'probabilities have to sum to one, here:{torch.sum(p)}'
            weighted_pred = torch.matmul(pred_stack.permute(1,2,0),p)
            return weighted_pred, p

        @torch.no_grad()
        def get_aggregated_predictions(self, client_predictions_list: list):
            # Compute mean
            pred_stack = torch.stack(client_predictions_list) # shape: [n_clients, pub_ds_size, n_classes]
            mean_pred, mean_outlier_scores = super().get_aggregated_predictions(client_predictions_list)
            self.update_weights(mean_outlier_scores)
            weighted_pred, p = self.weighted_sum(pred_stack)
            return weighted_pred, p
    return ExpWeights


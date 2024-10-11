# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         strategies.py
# Description:  Strategy classes
# ===========================================================================
import abc

import numpy as np

from utilities import Utilities as Utils


# Base Classes
class FederatedLearningBaseClass(abc.ABC):
    """Federated learning base class, defines the general assumptions and functions."""

    def __init__(self, **kwargs):
        self.config = kwargs['config']
        self.runner = kwargs['runner_instance']

    def do_clients_train_on_public_data(self):
        """Returns True if the clients train on public data."""
        return False

    def verify_input(self):
        """Verify strategy input."""
        assert self.config['n_total_local_epochs'] is not None, 'Need to specify the number of total local epochs'
        assert self.config['n_total_local_epochs'] >= 0, 'Number of total local epochs should be positive'

        assert self.config['n_communications'] is not None, 'Need to specify the number of communications'
        assert self.config['n_communications'] >= 0, 'Number of communications should be positive'
        assert self.config['n_communications'] <= self.config[
            'n_total_local_epochs'], 'Number of communications should be smaller than the number of total local epochs'


    def get_phase_length(self, current_round: int) -> int:
        """Returns the number of epochs to train locally in the given round"""

        n_epochs_total = self.config['n_total_local_epochs']
        n_communications = self.config['n_communications']

        # Split the total number of local epochs into the number of phases uniformly
        epochs_per_round, remainder = divmod(n_epochs_total, n_communications)
        epochs_per_round_schedule = [epochs_per_round if idx >= remainder else epochs_per_round + 1 for idx in
                                        range(n_communications)]

        phase_length = epochs_per_round_schedule[current_round - 1]  # index starts from 0
        return phase_length

    def before_local_training(self):
        """Method that is called before local training."""
        pass

    def after_local_training(self):
        """Method that is called after local training."""
        pass

    def at_round_end(self):
        """Method that is called at the end of a round."""
        pass


# Inheriting Classes
class FedAVG(FederatedLearningBaseClass):
    """Federated averaging: The server model is updated by averaging the client models, which are then broadcast to the clients."""

    def before_local_training(self):
        """Before local training, we broadcast the server model to all clients."""
        self.runner.broadcast_server_model_to_clients()

    def after_local_training(self):
        """After local training, we broadcast the server model to all clients."""
        self.runner.broadcast_agg_client_models_to_server()


class FedDistill(FedAVG):
    """FedDistill: Clients train on local data, the server trains on the public predictions but shares the model with the clients."""

    def before_local_training(self):
        """Before local training, we broadcast the server model to all clients."""
        self.runner.broadcast_server_model_to_clients()

    def after_local_training(self):
        """After local training, we simply perform distillation."""
        self.runner.collect_avg_output_and_distill_to_server()

    def at_round_end(self):
        """At the end of a round, we collect the ensemble test accuracy."""
        # Get the prediction ensemble test accuracy as a point of reference
        test_ensemble_predictions_list = self.runner.get_client_predictions(mode='test')
        test_ensemble_prediction = Utils.average_client_predictions(
            client_predictions_list=test_ensemble_predictions_list,
            output_type='soft_prediction')

        self.runner.ensemble_test_acc = self.runner.compute_accuracy(loader=self.runner.dataloaders_public['test'],
                                                                     prediction=test_ensemble_prediction)

# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         runners/ScratchRunner.py
# Description:  Scratch Runner class, used for starting the run from scratch
# ===========================================================================
import sys
import time
from typing import Optional

import torch
import wandb
from torch.cuda.amp import autocast
from torchmetrics.classification import MulticlassAccuracy as Accuracy
from tqdm.auto import tqdm

from actors import Actor
from byzantine import attacks, defences
from runners.BaseRunner import BaseRunner
from utilities import Utilities as Utils


class ScratchRunner(BaseRunner):
    """Handles the federated training by concurrently training clients and the server. We do not fetch pretrained models anymore."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_round = None
        self.artifact = None

        entity, project = wandb.run.entity, wandb.run.project
        self.initial_artifact_name = f"seed_placeholder-{entity}-{project}-{self.config.arch}-{self.config.dataset}-{self.config.run_id}"

    def find_existing_seed(self):
        """Finds an existing wandb artifact and pulls the seed. We do not pull the initial model,
                to ensure that each client has a differently initialized model."""
        # Create a new artifact, this is idempotent, i.e. no artifact is created if this already exists
        try:
            self.artifact = wandb.run.use_artifact(f"{self.initial_artifact_name}:latest")
            seed = self.artifact.metadata["seed"]
            self.seed = seed
        except Exception as e:
            print(e)

        outputStr = f"Found {self.initial_artifact_name} with seed {seed}" if self.artifact is not None else "Nothing found."
        sys.stdout.write(f"Trying to find reference artifact in project: {outputStr}\n")

    def save_artifact_seed(self):
        """Save artifact and seed before training so other runs can fetch it.
            If self.artifact is not None, this is not necessary since the artifact already exists.
        """
        if self.artifact is None:
            self.artifact = wandb.Artifact(self.initial_artifact_name, type='seed_placeholder',
                                           metadata={'seed': self.seed})
            sys.stdout.write(f"Creating {self.initial_artifact_name}.\n")
            wandb.run.use_artifact(self.artifact)

    @torch.no_grad()
    def broadcast_server_model_to_clients(self):
        """Broadcast server model params to all clients."""
        sys.stdout.write("Broadcasting server model to clients.\n")
        server_state_dict = self.server.model.state_dict()
        for client in self.clients:
            client.model.load_state_dict(server_state_dict)

        communication_cost = Utils.get_model_communication_cost(self.server.model)
        self.total_bytes_communicated += communication_cost * len(self.clients)

    @torch.no_grad()
    def broadcast_agg_client_models_to_server(self):
        """Broadcast agg. client models to the server."""
        sys.stdout.write(f"{self.config.attack}: Broadcasting clients models and applying attack.\n")
        client_model_list = self.attack.get_perturbed_client_models()  # Attack perturbs the model

        sys.stdout.write(f"{self.config.defence}: Averaging models with defence mechanism.\n")
        averaged_model = self.defence.get_aggregated_model(client_model_list)

        self.server.model.load_state_dict(averaged_model)

        # We use the server model here since it is roughly the same size as every client model
        communication_cost = Utils.get_model_communication_cost(self.server.model)
        self.total_bytes_communicated += communication_cost * len(self.clients)

    def set_attack_defence(self):
        """Set attack and defence"""
        if self.config.attack not in [None, 'None', 'none',
                                      'NoAttack']:  # For NoAttack, we do not set byzantine clients
            n_byzantine_clients = self.config.n_byzantine_clients or 0
            assert 0 <= n_byzantine_clients <= self.config.n_clients, "Number of byzantine clients must be in [0, n_clients]."
            sys.stdout.write(f"{n_byzantine_clients} byzantine clients with attack {self.config.attack}.\n")

            # Randomly pick n_byzantine_clients clients
            byzantine_client_indices = torch.randperm(len(self.clients))[:n_byzantine_clients]
            byzantine_ids_str = ', '.join(str(x) for x in byzantine_client_indices.tolist())
            sys.stdout.write(f"Client(s): {byzantine_ids_str} are byzantine.\n")
            for idx in byzantine_client_indices:
                self.clients[idx].is_byzantine = True

            # Set the attack
            try:
                self.attack = getattr(attacks, self.config.attack)(clients=self.clients, config=self.config,
                                                                   runner_instance=self)
            except AttributeError:
                raise AttributeError(f"Attack {self.config.attack} not found.")
        else:
            sys.stdout.write(f"No attack.\n")
            assert self.config.n_byzantine_clients in [0, None, 'None'], "If no attack is used, n_byzantine_clients must be 0."
            self.attack = attacks.NoAttack(clients=self.clients, config=self.config, runner_instance=self)

        if self.config.defence not in [None, 'None', 'none']:
            # Set the defence
            if self.config.memory_method is not None:
                robust_method = getattr(defences,
                                        self.config.defence) 
                if self.config.memory_method == 'expweights':
                    self.defence = defences.choose_aggregation_expweights(robust_method)(clients=self.clients,
                                                                                            config=self.config,
                                                                                            runner_instance=self)
                else:
                    raise AttributeError(f"Memory method {self.config.memory_method} not found.")
            else:
                try:
                    self.defence = getattr(defences, self.config.defence)(clients=self.clients, config=self.config,
                                                                          runner_instance=self)
                except AttributeError:
                    raise AttributeError(f"Defence {self.config.defence} not found.")

            sys.stdout.write(f"Using defence {self.config.defence}.\n")
        else:
            sys.stdout.write(f"No defence.\n")
            self.defence = defences.NoDefence(clients=self.clients, config=self.config, runner_instance=self)

    def set_client_models(self):
        """For each client: Initialize the models"""
        for client in self.clients:
            client.set_model(reinit=True, fileName=None)

    def set_client_optimizers(self, reinit_optimizer: bool = True, lr_duration: Optional[int] = None):
        """Sets the optimizers/schedulers of clients.
        Args:
            reinit_optimizer (bool): If True, the optimizers are reinitialized.
            lr_duration (Optional[int]): If given, the learning rate is restarted for lr_duration epochs.
        """
        clients_train_on_public = self.strategy.do_clients_train_on_public_data()
        add_public = 0 if not clients_train_on_public else len(self.dataloaders_public['train'])
        assert add_public == 0, "FED does not work with the current schedulers."
        for client in self.clients:
            n_batches_per_epoch = len(client.dataloader)
            n_epochs = lr_duration or self.config["n_total_local_epochs"]
            client.set_optimizer_and_scheduler(n_epochs=n_epochs, n_batches_per_epoch=n_batches_per_epoch,
                                               reinit_optimizer=reinit_optimizer)

    def set_server_optimizer(self, reinit_server: bool, first_init: bool):
        """Sets the optimizers/schedulers of the server.
        Args:
            reinit_server (bool): If True, the optimizer/scheduler is reinitialized and adapted to phase length.
        """
        n_base_epochs = self.config.n_server_epochs_per_round

        if reinit_server:
            sys.stdout.write(f"Reinitializing server optimizer and scheduler.\n")
            n_epochs = n_base_epochs
        else:
            n_epochs = self.config.n_communications * n_base_epochs
        n_batches_per_epoch = len(self.server.dataloader)
        self.server.set_optimizer_and_scheduler(n_epochs=n_epochs, n_batches_per_epoch=n_batches_per_epoch,
                                                reinit_optimizer=(reinit_server or first_init))

    @torch.no_grad()
    def compute_accuracy(self, loader, prediction):
        """
        Compute accuracy on loader where prediction is a tensor containing all predictions of ensemble
        Args:
            loader (dataloader): Dataloader to evaluate
            prediction (torch.tensor): Containing prediction on all samples.

        Returns: Accuracy as float
        """

        sys.stdout.write(f"Evaluating accuracy of ensemble.\n")
        accuracy_meter = Accuracy(num_classes=self.n_classes).to(device=self.device)
        with tqdm(loader, leave=True) as pbar:
            for _, y_target, indices in pbar:
                y_target = y_target.to(device=self.device)
                accuracy_meter(prediction[indices], y_target)

        return accuracy_meter.compute()

    @torch.no_grad()
    def get_client_predictions(self, mode: str):
        """For each client: Predict the entire public train/test set and output for each sample the predicted probs.
             For this to work, the indices of the subset must have been reset to start from zero.
        Args:
            mode (str): Either 'train' or 'test', depending on whether to use server.trainData or the test set
        Returns: list of client predictions
        """
        assert mode in ['train', 'test']
        loader = self.dataloaders_public[mode]
        sys.stdout.write(f"\nCollecting predictions of all clients.\n")

        prediction_store_tensors = [torch.zeros(len(loader.dataset), self.n_classes, device=self.device) for _ in
                                    range(len(self.clients))]

        with tqdm(loader, leave=True) as pbar:
            for x_input, _, indices in pbar:
                x_input = x_input.to(self.device, non_blocking=True)  # Move to CUDA if possible
                with autocast(enabled=(self.use_amp is True)):
                    for client_idx, client in enumerate(self.clients):
                        output = client.model.eval()(x_input)  # Logits

                        probabilities = torch.nn.functional.softmax(output, dim=1)  # Softmax(Logits)
                        prediction_store_tensors[client_idx][indices] += probabilities

        return prediction_store_tensors

    def distill(self, actor: Actor, avg_output: torch.tensor, is_training: bool = True):
        """Train the actor (server, client) using averaged probabilities from all clients. If not is_training,
        the actor is only evaluated on the public train set.
        Args:
            actor (Actor): Client or Server to train-distill
            avg_output (torch.tensor): Tensor keeping averaged probs/predictions for each sample in pub trainData
            is_training (bool): Whether to train the actor or only evaluate it
        """
        if actor.actor_type == 'server':
            loader = self.dataloaders_public['train_server']
        else:
            loader = self.dataloaders_public['train']
        sys.stdout.write(
            f"\n{'Training' if is_training else 'Evaluating'} {actor.actor_name} on average prediction/probabilities"
            f" (softmax).\n")
        with torch.set_grad_enabled(is_training):
            with tqdm(loader, leave=True) as pbar:
                for x_input, _, indices in pbar:
                    x_input = x_input.to(self.device, non_blocking=True)  # Move to CUDA if possible
                    target = avg_output[indices].to(self.device, non_blocking=True)  # Avg probs/predictions of batch
                    actor.optimizer.zero_grad()

                    with autocast(enabled=(self.use_amp is True)):
                        output = actor.model.train(mode=is_training)(x_input)  # Logits
                        loss = actor.loss_criterion(output, target)
                    if is_training:
                        actor.gradScaler.scale(loss).backward()  # AMP gradient scaling + Backpropagation
                        actor.gradScaler.step(actor.optimizer)  # Optimization step
                        actor.gradScaler.update()  # Update AMP gradScaler
                        actor.scheduler.step()

                    if actor.actor_type == 'server':
                        # We specify y_target as None, since it is not available
                        actor.update_batch_metrics(mode='train', loss=loss, output=output, y_target=None)

    def train_client_local(self, n_epochs: int, current_round: int):
        """Train each client locally on its private dataset for n_epochs."""
        for epoch in range(1, n_epochs + 1, 1):
            for client in self.clients:
                client.reset_averaged_metrics()  # Reset metrics of clients
                if client.is_byzantine:
                    sys.stdout.write(
                        f"\nRound {current_round}/{self.config.n_communications} - Local Epoch {epoch}/{n_epochs}: Skipping byzantine client-{client.client_id}.")
                    continue
                sys.stdout.write(
                    f"\nRound {current_round}/{self.config.n_communications} - Local Epoch {epoch}/{n_epochs}: Locally training client-{client.client_id}.")
                self.train_epoch(actor=client, data='train', epoch=epoch)  # Train on private dataset
                # Evaluate
                if self.config.client_early_stopping:
                    self.evaluate_model(actor=client, data='val')
                if epoch == n_epochs:
                    self.evaluate_model(actor=client, data='test')

                if self.config.client_early_stopping:
                    # Save the checkpoint if it's better than the previous one
                    client.update_checkpoint()

            self.log_clients_at_epoch_end(epoch=self.client_epochs_done + epoch,
                                          commit=True)  # Log clients at the end of each epoch
        self.client_epochs_done += n_epochs

    def collect_avg_output_and_distill_to_server(self):
        sys.stdout.write(f"{self.config.attack}: Broadcasting clients predictions and applying attack.\n")
        client_prediction_list = self.attack.get_perturbed_client_predictions()  # Attack perturbs the client predictions

        sys.stdout.write(f"{self.config.defence}: Averaging predictions with defence mechanism.\n")
        defence_start = time.time()
        averaged_predictions, mean_outlier_scores = self.defence.get_aggregated_predictions(client_prediction_list)
        self.defence_time = time.time() - defence_start

        # Log the outlier scores
        indices = [idx for idx in range(len(mean_outlier_scores))]
        scores = [float(mean_outlier_scores[idx]) for idx in range(len(mean_outlier_scores))]
        Utils.dump_bar_plot_to_wandb(x=indices, y=scores, xlabel="Client ID", ylabel="Outlier Score",
                                     title="Mean Outlier Scores by Client Index",
                                     wandb_identifier="outlier_scores")

        sys.stdout.write(f"\nDistilling to server in round {self.current_round}/{self.config.n_communications}.")
        length = self.config.n_server_epochs_per_round
        for epoch in range(1, length + 1, 1):
            self.server.reset_averaged_metrics()
            self.distill(actor=self.server, avg_output=averaged_predictions, is_training=True)

            self.evaluate_model(actor=self.server, data='val')
            self.evaluate_model(actor=self.server, data='test')

            if self.config.server_early_stopping:
                # Save the checkpoint if it's better than the previous one
                self.server.update_checkpoint()

            if epoch == length:
                # We reset the server val and eval metrics, they have to be recomputed in the train function
                self.server.reset_val_and_test_metrics()
            self.log_server(epoch=self.server_epochs_done + epoch, commit=(epoch < length))  # Log server
        self.total_bytes_communicated += Utils.calculate_communication_cost(client_prediction_list)
        self.server_epochs_done += length

        # Reset the model of the server to the best checkpoint, if early stopping is enabled
        if self.config.server_early_stopping:
            self.server.load_checkpoint()

    def train_federated(self):
        """Train the server and clients in a federated way."""
        for current_round in range(0, self.config.n_communications + 1, 1):
            self.current_round = current_round
            is_training = current_round > 0
            sys.stdout.write(f"\nFL - Round {current_round}/{self.config.n_communications}\n") if is_training \
                else sys.stdout.write(f"\nFL - Evaluation round.\n")
            t_start = time.time()

            # Reset the metrics of all actors
            for client in self.clients:
                client.reset_averaged_metrics()
            self.server.reset_averaged_metrics()
            

            if is_training:
                # Before local training, the server potentially sends aggregated info to clients
                self.strategy.before_local_training()

                # Determine the number of epochs for this round
                round_n_epochs = self.strategy.get_phase_length(current_round=current_round)

                if self.config.restart_client_lr:
                    self.set_client_optimizers(reinit_optimizer=False, lr_duration=round_n_epochs)
                if self.config.reinit_server:
                    self.server.set_model(reinit=True)
                    self.set_server_optimizer(reinit_server=self.config.reinit_server, first_init=False)
                if self.config.warm_restarts:
                    # Warmup the learning rate for the first 5% of the epochs
                    warmup_steps_client = int(0.05 * round_n_epochs * len(self.clients[0].dataloader))
                    sys.stdout.write(f"Warming up momentum for 5% of the iterations.\n")
                    for client in self.clients:
                        client.warmup_scheduler(warmup_steps=warmup_steps_client)

                    server_train_length = self.config.n_server_epochs_per_round or 0
                    if server_train_length > 0:
                        warmup_steps_server = int(
                            0.05 * server_train_length * len(self.dataloaders_public['train_server']))
                        self.server.warmup_scheduler(warmup_steps=warmup_steps_server)

                self.train_client_local(n_epochs=round_n_epochs,
                                        current_round=current_round)  # Clients train locally, then evaluate them
                if self.config.client_early_stopping:
                    for client in self.clients:
                        client.load_checkpoint()

                # After local training, the clients potentially sends aggregated info to the server
                self.strategy.after_local_training()
            else:
                round_n_epochs = 0

            self.evaluate_model(actor=self.server, data='val')
            self.evaluate_model(actor=self.server, data='test')
            self.strategy.at_round_end()  # Strategy-specific actions at the end of the round
            self.total_epochs_completed += round_n_epochs
            self.log_at_round_end(round=current_round, round_n_epochs=round_n_epochs,
                                  round_runtime=time.time() - t_start)

    def run(self):
        """Function controlling the workflow."""
        self.find_existing_seed()  # Check if artifact with same run_id exists, if so use the seed

        # We initialize the models before actually setting the seed!
        self.set_client_models()  # Each client inits its own model
        self.server.set_model(reinit=True)  # Server inits its own model
        self.set_seed()  # Set the seed
        self.save_artifact_seed()  # Save the artifact for others to fetch from Wandb, if needed.

        # We initialize the dataloaders after setting the seed!
        self.assign_dataloaders()  # Assign dataloaders

        self.set_client_optimizers()
        self.set_server_optimizer(reinit_server=self.config.reinit_server, first_init=True)

        self.set_attack_defence()  # Set attack and defence

        self.train_federated()
        self.final_log()  # Log all clients and server

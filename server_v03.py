from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr as fl
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from evaluate_v01 import evaluate
from logging import DEBUG, ERROR, INFO, WARN
from flwr.common.logger import log
from flwr.common import MetricsAggregationFn, Scalar, FitRes, EvaluateRes, Parameters
import wandb
import os
import torch
from unet import UNet
import numpy as np
import pickle
import argparse


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        initial_learning_rate: float = 1e-6,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        save_dir="./checkpoints",
        save_model: bool = True,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit, 
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients, 
            min_evaluate_clients=min_evaluate_clients, 
            min_available_clients=min_available_clients, 
            evaluate_fn=evaluate_fn, 
            on_fit_config_fn=on_fit_config_fn, 
            on_evaluate_config_fn=on_evaluate_config_fn, 
            accept_failures=accept_failures, 
            initial_parameters=initial_parameters, 
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, 
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, 
            inplace=inplace
            )
        self.learning_rate = initial_learning_rate
        self.run_id = 0 
        self.save_dir = save_dir
        self.model = UNet(n_channels=3, n_classes=1, bilinear=False)
        self.sm = save_model
    
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy | FitIns]]:
        """Configure the next round of training."""
        self.run_id = server_round
        config = {
            "learning_rate": self.learning_rate,
            "run_id": self.run_id,
                  }
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        learning_rates = []
        for client, fit_res in results:
            fit_dic = fit_res.metrics
            log(INFO, f"Client {client.cid[:4]}:")
            log(INFO, f"-> lr: {fit_dic['lr']}")
            log(INFO, f"-> loss: {fit_dic['loss']}")
            log(INFO, f"-> dice: {fit_dic['dice']}")
            # log(INFO, f"-> dice_aug: {fit_dic['dice_aug']}")
            wandb.log({
                f'lr_{client.cid[:4]}': fit_dic['lr'],
                f'loss_{client.cid[:4]}': fit_dic['loss'],
                f'dice_{client.cid[:4]}': fit_dic['dice'],
                # f'dice_{client.cid[:4]}_aug': fit_dic['dice_aug']
            })
            if fit_dic['lr'] is not None:
                learning_rates.append(fit_dic['lr'])
        if learning_rates:
            self.learning_rate = sum(learning_rates) / len(learning_rates)
            log(INFO, f"Updated learning rate: {self.learning_rate}")
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            # weights: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # self.model.set_weights(weights)
            # self.save_model(rnd)
            if (rnd % 10) == 0:
                if self.sm:
                    self.save_parameters(aggregated_parameters, rnd)
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[BaseException]
    ) -> Tuple[float, Dict[str, Scalar]]:
        for client, evaluate_res in results:
            dice = evaluate_res.metrics
            log(
                INFO,
                f"-> Mega Dice Score: {dice['Dice Score']}",
            )
            log(
                INFO,
                f"-> Mega Dice Score (aug): {dice['Dice Score aug']}",
            )
            wandb.log({
                'Mega Dice Score': dice['Dice Score'],
                'Mega Dice Score (aug)': dice['Dice Score aug'],
                'Round': rnd,
            })
            break
        return super().aggregate_evaluate(rnd, results, failures)
    
    def save_model(self, rnd: int):
        save_path = os.path.join(self.save_dir, f"fedavg_model_round_{rnd}.h5")
        self.model.save(save_path)
        log(INFO, f"Model saved at round {rnd}")

    def save_parameters(self, parameters: Parameters, rnd: int):      
        weights: List[np.ndarray] = parameters_to_ndarrays(parameters)
        save_path = os.path.join(self.save_dir, f"fedavg_weights.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(weights, f)
        log(INFO, f"Model's weights saved at round {rnd}")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks -> Server Side')
    parser.add_argument('--save-model', '-sm', dest='save_model', type=bool, default=False, help='Whether to save the model?')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-6, dest='lr',
                        help='Initial Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load models weights from a .pkl file')
    parser.add_argument('--rounds', '-r', type=int, default=100, help='Number of training rounds')

    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    strategy = CustomFedAvg(initial_learning_rate=args.lr, save_model=args.save_model)
    wandb.init(project='flwr+cl', resume='allow', anonymous='must')
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy
    )
    

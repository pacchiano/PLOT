import argparse
import pandas as pd

from algorithms import train_baseline, train_epsilon_greedy, train_mahalanobis, train_PLOT
from dataclasses import dataclass
from typing import Any

ALGOS = ["Eps_Greedy", "Greedy", "PLOT", "Pessimistic_PLOT", "Mahalanobis"]
DATASETS = ["Adult", "Bank", "German", "MultiSVM", "MNIST"]

@dataclass
class ExperimentResults:
    regrets: Any
    accuracies: Any
    test_accuracy: Any

def run_experiment(dataset, algos, num_batches=2000):
    epsilon = .1
    alpha = 1
    # batch_size
    # representation_layer_size
    # baseline_steps
    # decaying_epsilon

    results = {}
    baseline_test_accuracy, baseline_model = train_baseline(
        dataset, num_timesteps = 1000, batch_size = 32, 
        MLP = True, representation_layer_size = 10
    )
    for algo in algos:
        if algo == "Eps_Greedy":
            instantaneous_regrets, instantaneous_accuracies, test_accuracy = train_epsilon_greedy(dataset, baseline_model, 
                num_batches = num_batches, batch_size = 32, 
                num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
                representation_layer_size = 10, threshold = .5, verbose = False, decaying_epsilon = True, epsilon = epsilon,
                )
            results[algo] = ExperimentResults(instantaneous_regrets, instantaneous_accuracies, test_accuracy)
        elif algo == "Greedy":
            instantaneous_regrets, instantaneous_accuracies, test_accuracy = train_epsilon_greedy(dataset, baseline_model, 
                num_batches = num_batches, batch_size = 32, 
                num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
                representation_layer_size = 10, threshold = .5, verbose = False, decaying_epsilon = True, epsilon = 0,
                )
            results[algo] = ExperimentResults(instantaneous_regrets, instantaneous_accuracies, test_accuracy)
        elif algo == "PLOT":
            instantaneous_regrets, instantaneous_accuracies, test_accuracy = train_PLOT(dataset, baseline_model, 
                num_batches = num_batches, batch_size = 32, 
                num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
                representation_layer_size = 10, threshold = .5, verbose = False, decaying_epsilon = False,  epsilon = epsilon,  
                pessimistic = False)
            results[algo] = ExperimentResults(instantaneous_regrets, instantaneous_accuracies, test_accuracy)
        elif algo == "Pessimistic_PLOT":
            instantaneous_regrets, instantaneous_accuracies, test_accuracy = train_PLOT(dataset, baseline_model, 
                num_batches = num_batches, batch_size = 32, 
                num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
                representation_layer_size = 10, threshold = .5, verbose = False, decaying_epsilon = False,  epsilon = epsilon,  
                pessimistic = True)
            results[algo] = ExperimentResults(instantaneous_regrets, instantaneous_accuracies, test_accuracy)
        elif algo == "Mahalanobis":
            instantaneous_regrets, instantaneous_accuracies, test_accuracy = train_mahalanobis(dataset, baseline_model, 
                num_batches = num_batches, batch_size = 32, 
                num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
                representation_layer_size = 10, threshold = .5, verbose = False,  alpha = alpha, lambda_reg = 1)
            results[algo] = ExperimentResults(instantaneous_regrets, instantaneous_accuracies, test_accuracy)
        else:
            raise ValueError("Unknown Algo specified.")
    return results, baseline_test_accuracy


def process_results(results):
    rows = []
    for dataset, algo_results in results.items():
        for algo_name, result in algo_results.items():
            rows.append({
                "dataset": dataset, "algo": algo_name, "regrets": result.regrets,
                "accuracies": result.accuracies, "test_accuracy": result.test_accuracy
            })
    pd.DataFrame(rows).to_csv("results.csv")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model parameters.
    parser.add_argument('--T', default=2000, help="Number of timesteps", type=int)
    parser.add_argument('--baseline_steps', default=20_000, help="Number of baseline gradient steps", type=int)
    parser.add_argument('--batch_size', default=32, type=int, help="")

    # Configure datasets and algos.
    parser.add_argument('--datasets', nargs="*", default=DATASETS, choices=DATASETS, help="Individual dataset name")
    parser.add_argument('--algo_names', nargs="*", default=ALGOS, choices=ALGOS, help="Algo name")
    args = parser.parse_args()
    multi_dataset_results = {}
    for dataset in args.datasets:
        algo_results, baseline_test_accuracy = run_experiment(
            dataset, args.algo_names, args.T
        )
        multi_dataset_results[dataset] = algo_results
    print(multi_dataset_results)
    process_results(multi_dataset_results)


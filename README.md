# Neural Pseudo-Label Optimism for the Bank Loan Problem

This repository is the official implementation of Neural Pseudo-Label Optimism for the Bank Loan Problem.

## Requirements

NOTE: To run our experiments, we require pytorch to be compiled with GPU support, and for your machine to have at least one GPU.
To effectively and efficiently reproduce our multi-experiment runs, we recommend a system with 5 (or N_EXPERIMENTS) GPUs.
This is much simpler on a Linux system, which supports pytorch CUDA binaries.
Installation instructions are provided for Linux.

To install requirements:

```setup
conda env create -f environment.yml
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

## Experiments

There are two experiment scripts in this repo. 
`original_experiments.py` contains the full reproduction of the paper results. 
`simple_experiments.py` contains logic to reproduce the experiments in the [Google Collab demo](https://colab.research.google.com/drive/1kjukVierl8g-fpmvrCJ2yNI6Yog3qGO5#scrollTo=L6yaz6olHNvc). 

### Original Experiments
The original experiment script parallelizes experiment replicates via Ray, but does not parallelize on algorithms.
The script supports 4 algorithms: Epsilon Greedy, Greedy, Mahalanobis (Neural UCB), and PLOT.
The script also supports 4 datasets: Adult, Bank, MultiSVM and MNIST.

In the paper, we run all algorithms with 5 replicates, for 2000 timesteps.
To repro this exactly, run this command:

```experiments
python original_experiments.py --num_experiments=5 --ray
```
With Ray enabled, this will run 5 experiments in parallel. Each experiment requires a dedicated GPU, so please do not run more experiments than available GPUs.

To run only one replicate, simply run (Ray is unnecessary):
```experiment
python original_experiments.py --num_experiments=1
```
 
To run small test experiments (e.g. for dev), one can specify a specific dataset(s), a specific algorithm(s), and a short timescale. 
```experiment
python original_experiments.py --num_experiments=1 --datasets Adult --algo_nams Eps_Greedy --T=100
```

Hyperparameters are already specified in the model file, and can easily be viewed inside of `pytorch_experiments.py`.

#### Results

Our algorithm achieves the following performances:

| Dataset          | Cumulative Regret@T=2000 | Std. Dev of Cumulative Regret@T=2000 |
| ---------------- |------------------------- | -----------------------------------  |
| Adult            |            6.145         |      1.529                           |
| Bank             |            0.736         |      0.528                           |
| MNIST            |            1.711         |      0.406                           |

These results are output to the `experiment_results` folder after running the experiment code.
To get these printed, please copy `final_results/analyze.py` to the dataset folder you wish to analyze.
E.g. for Adult: `experiment_results/Adult/PLOT/data/`. From that directory, run `python analyze.py`, and the cumulative regret (mean+stddev) of the method will be printed.


### Simplified Experiments
We also offer a script for simplified training of the relevant algorithms. 
This script does not use the plotting/parallelization structure of the original experiments, allowing for more flexible iteration. 
The script supports one additional algorithm: Pessimistic_PLOT, as well as one additional dataset: German.

The script provides much simpler results output, written to a csv file:
1. Instantaneous regrets (r_t) w.r.t. baseline
3. Instantaneous_accuracies (a_t) w.r.t. baseline
4. Final test_accuracy (a_T) for each algorithm and dataset. 

Run all:
```experiments
python simple_experiment.py
```

Select datasets/algos/timesteps:
```experiments
python simple_experiment.py --datasets Adult --algo_names Greedy --T=100
```

This script makes use of the implementations in `algorithms.py`, just as in the Colab. 

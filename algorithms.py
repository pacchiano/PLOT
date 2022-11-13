import numpy as np
import torch
import IPython

import random

from datasets import get_dataset_simple, GrowingNumpyDataSet
from models import (
    TorchBinaryLogisticRegression
)


from newmodels import (
TorchMultilayerRegressionMahalanobis,
TorchMultilayerRegression
)



from model_training_utilities import evaluate_model, train_model

from algorithmsmodsel import train_mahalanobis_modsel, train_opt_reg_modsel









def train_baseline(dataset, num_timesteps, batch_size, 
    representation_layer_sizes = [10, 10], threshold = .5, mode = "classification"):
    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000)

    # IPython.embed()
    # raise ValueError("Asdflkm")

    if mode == "classification":
        output_filter = 'logistic'
    else:
        output_filter = "none"        


    baseline_model = TorchMultilayerRegression(
        representation_layer_sizes=representation_layer_sizes,
        dim = train_dataset.dimension,
        output_filter = output_filter
    )

    #IPython.embed()
    baseline_model = train_model(
        baseline_model, num_timesteps, train_dataset, batch_size
    )

    print("Finished training baseline model")

    
    #### Computing test accuracy baseline
    with torch.no_grad():
        baseline_batch_test = test_dataset.get_batch(10000000000) 
        
        batch_X, batch_y = baseline_batch_test
        baseline_test_accuracy = baseline_model.get_accuracy(batch_X, batch_y, threshold)
        baseline_test_loss = baseline_model.get_loss(batch_X, batch_y)
    print("Baseline model test accuracy {}".format(baseline_test_accuracy))
    print("Baseline test loss {}".format(baseline_test_loss))
    #IPython.embed()

    with torch.no_grad():
        baseline_batch_train = train_dataset.get_batch(10000000000) 
        
        batch_X, batch_y = baseline_batch_train
        baseline_train_accuracy = baseline_model.get_accuracy(batch_X, batch_y, threshold)
        baseline_train_loss = baseline_model.get_loss(batch_X, batch_y)

    print("Baseline model train accuracy {}".format(baseline_train_accuracy))
    print("Baseline train loss {}".format(baseline_train_loss))
    return (baseline_test_accuracy.item(), baseline_train_accuracy.item(),baseline_test_loss.item(), baseline_train_loss.item() ), baseline_model









def train_epsilon_greedy(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size, 
    representation_layer_sizes = [10, 10], threshold = .5, epsilon = .1,
    verbose = False, decaying_epsilon = False, 
    restart_model_full_minimization = False):
    
    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000)


    model = TorchMultilayerRegression(
        representation_layer_sizes=representation_layer_sizes,
        dim = train_dataset.dimension,
        output_filter = 'logistic'
    )



    growing_training_dataset = GrowingNumpyDataSet()
    instantaneous_regrets = []
    instantaneous_accuracies = []
    eps_multiplier = 1.0

    num_positives = []
    num_negatives = []
    false_neg_rates = []
    false_positive_rates = []


    for i in range(num_batches):
        if verbose:
            print("Processing epsilon greedy batch ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)

        with torch.no_grad():
            
            if decaying_epsilon:
                eps_multiplier = 1.0/(np.sqrt(i+1))
            predictions = model.get_thresholded_predictions(batch_X, threshold)
            baseline_predictions = baseline_model.get_thresholded_predictions(batch_X, threshold)

            if torch.cuda.is_available():
                epsilon_greedy_mask = torch.bernoulli(torch.ones(predictions.shape)*epsilon*eps_multiplier).bool()#.cuda()
            else:
                epsilon_greedy_mask = torch.bernoulli(torch.ones(predictions.shape)*epsilon*eps_multiplier).bool()

            mask = torch.max(epsilon_greedy_mask,predictions)

            boolean_labels_y = batch_y.bool()
            accuracy = (torch.sum(mask*boolean_labels_y) +torch.sum( ~mask*~boolean_labels_y))*1.0/batch_size
            
            #### TOTAL NUM POSITIVES
            total_num_positives = torch.sum(mask)

            #### TOTAL NUM NEGATIVES   
            total_num_negatives = torch.sum(~mask)

            #### FALSE NEGATIVE RATE
            false_neg_rate = torch.sum(~mask*boolean_labels_y)*1.0/(torch.sum(~mask)+.00000000001)


            #### FALSE POSITIVE RATE            
            false_positive_rate = torch.sum(mask*~boolean_labels_y)*1.0/(torch.sum(mask)+.00000000001)



            num_positives.append(total_num_positives.item())
            num_negatives.append(total_num_negatives.item())
            false_neg_rates.append(false_neg_rate.item())
            false_positive_rates.append(false_positive_rate)            



            accuracy_baseline = (torch.sum(baseline_predictions*boolean_labels_y) +torch.sum( ~baseline_predictions*~boolean_labels_y))*1.0/batch_size
            instantaneous_regret = accuracy_baseline - accuracy

            instantaneous_regrets.append(instantaneous_regret.item())
            instantaneous_accuracies.append(accuracy.item())

            filtered_batch_X = batch_X[mask, :]
            filtered_batch_y = batch_y[mask]

        growing_training_dataset.add_data(filtered_batch_X, filtered_batch_y)

        #### Filter the batch using the predictions
        #### Add the accepted points and their labels to the growing training dataset

        model = train_model( model, num_opt_steps, 
                growing_training_dataset, opt_batch_size, 
                restart_model_full_minimization = restart_model_full_minimization)



                 
    print("Finished training epsilon-greedy model {}".format(epsilon))
    test_accuracy = evaluate_model(test_dataset, model, threshold).item()
    print("Final model test accuracy {}".format(test_accuracy))

    results = dict([])
    results["instantaneous_regrets"] = instantaneous_regrets
    results["test_accuracy"] = test_accuracy
    results["instantaneous_accuracies"] = instantaneous_accuracies
    results["num_negatives"] = num_negatives
    results["num_positives"] = num_positives
    results["false_neg_rates"] = false_neg_rates
    results["false_positive_rates"] = false_positive_rates

    return results# instantaneous_regrets, instantaneous_accuracies, test_accuracy






def train_mahalanobis(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size, 
    representation_layer_sizes = [10, 10], threshold = .5, alpha = 1, lambda_reg = 1,
    verbose = False, 
    restart_model_full_minimization = False):
    
    return train_mahalanobis_modsel(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size,
    representation_layer_sizes = representation_layer_sizes, threshold = threshold, alphas = [alpha], 
    lambda_reg = lambda_reg,
    verbose = verbose,
    restart_model_full_minimization = restart_model_full_minimization, modselalgo = "Corral")




def train_opt_reg(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size, 
    representation_layer_sizes = [10, 10], threshold = .5, reg = 1,
    verbose = False, 
    restart_model_full_minimization = False):
    
    return train_opt_reg_modsel(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size,
    representation_layer_sizes = representation_layer_sizes, threshold = threshold, regs = [reg], 
    verbose = verbose,
    restart_model_full_minimization = restart_model_full_minimization, modselalgo = "Corral")



def train_PLOT(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size, MLP = True, 
    representation_layer_size = 10, threshold = .5, epsilon = .1,
    verbose = False, decaying_epsilon = False, 
    restart_model_full_minimization = False, pessimistic = False, 
    weight = None, radius = float("inf")):
    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000)

    model = TorchBinaryLogisticRegression(
        random_init=True,
        alpha=0,
        MLP=MLP,
        representation_layer_size=representation_layer_size,
        dim=train_dataset.dimension
    )
    pseudo_label_model = TorchBinaryLogisticRegression(
        random_init=True,
        alpha=0,
        MLP=MLP,
        representation_layer_size=representation_layer_size,
        dim=train_dataset.dimension
    )

    growing_training_dataset = GrowingNumpyDataSet()
    instantaneous_regrets = []
    instantaneous_accuracies = []
    eps_multiplier = 1.0

    for i in range(num_batches):
        if verbose:
            print("Processing PLOT batch ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)

        with torch.no_grad():
            if decaying_epsilon:
                eps_multiplier = 1.0/(np.sqrt(i+1))

            mle_predictions = model.get_thresholded_predictions(batch_X, threshold)
            epsilon_greedy_mask = torch.bernoulli(torch.ones(mle_predictions.shape)*epsilon*eps_multiplier).bool()#.cuda()
            pseudo_label_filtered_mask = epsilon_greedy_mask*~mle_predictions
            pseudo_indices = torch.nonzero(pseudo_label_filtered_mask).squeeze(dim=1)

            ### Compute the pseudo_label_filtered_batch            
            pseudo_label_filtered_batch_X = batch_X[pseudo_label_filtered_mask, :]
            if weight:
              delta = 0.9
              weight = 4*np.sqrt(i * np.ln((6*(i**2) * np.ln(i)) / delta))
              pseudo_label_filtered_batch_X = pseudo_label_filtered_batch_X.repeat(weight, 1)
            if pessimistic:
              pseudo_labels = torch.zeros(pseudo_label_filtered_batch_X.shape[0]).type(batch_y.dtype)#.cuda()
            
            else:
              pseudo_labels = torch.ones(pseudo_label_filtered_batch_X.shape[0]).type(batch_y.dtype)#.cuda()

        ### If the pseudo label filtered batch is nonempty train pseudo-label model

        if pseudo_label_filtered_batch_X.shape[0] != 0:
            growing_training_dataset.add_data(pseudo_label_filtered_batch_X, pseudo_labels)
            pseudo_label_model.network.load_state_dict(model.network.state_dict())
            pseudo_label_model = train_model( pseudo_label_model, num_opt_steps, 
                growing_training_dataset, opt_batch_size, 
                restart_model_full_minimization=False)

            if verbose:
                print("Trained pseudo-label model ")


            ### Restore the data buffer to its last state
            growing_training_dataset.pop_last_data()

        ### Figure the optimistic predictions 
        with torch.no_grad():            
            pseudo_label_predictions = pseudo_label_model.get_thresholded_predictions(batch_X, threshold)[pseudo_indices]
            optimistic_predictions = mle_predictions.clone().detach()
            optimistic_predictions[pseudo_indices] = pseudo_label_predictions
            baseline_predictions = baseline_model.get_thresholded_predictions(batch_X, threshold)

            boolean_labels_y = batch_y.bool()
            accuracy = (torch.sum(optimistic_predictions*boolean_labels_y) +torch.sum( ~optimistic_predictions*~boolean_labels_y))*1.0/batch_size
           
            accuracy_baseline = (torch.sum(baseline_predictions*boolean_labels_y) +torch.sum( ~baseline_predictions*~boolean_labels_y))*1.0/batch_size
            instantaneous_regret = accuracy_baseline - accuracy


            filtered_batch_X  = batch_X[optimistic_predictions, :]
            filtered_batch_y = batch_y[optimistic_predictions]


        instantaneous_regrets.append(instantaneous_regret.detach().item())
        instantaneous_accuracies.append(accuracy.detach().item())

        #### Filter the batch using the predictions
        #### Add the accepted points and their labels to the growing training dataset
        growing_training_dataset.add_data(filtered_batch_X, filtered_batch_y)

        ### Train MLE
        model = train_model( model, num_opt_steps, 
                growing_training_dataset, opt_batch_size, 
                restart_model_full_minimization=restart_model_full_minimization)

                
    print("Finished training PLOT model - pessimistic {}".format(pessimistic))
    test_accuracy = evaluate_model(test_dataset, model, threshold).item()
    return instantaneous_regrets, instantaneous_accuracies, test_accuracy




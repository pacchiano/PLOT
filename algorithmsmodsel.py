import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import random
import itertools
import sys

import IPython

from dataclasses import dataclass
from torchvision import datasets, transforms
from typing import Any

from math import log, exp

from model_training_utilities import evaluate_model, train_model



from datasets import get_batches, get_dataset_simple, GrowingNumpyDataSet
from models import (
    TorchBinaryLogisticRegression,
    get_predictions,
    get_accuracies,
    get_accuracies_simple,
    get_breakdown_no_model,
    get_error_breakdown,
    get_special_breakdown
)

def binary_search(func,xmin,xmax,tol=1e-5):
    ''' func: function
    [xmin,xmax] is the interval where func is increasing
    returns x in [xmin, xmax] such that func(x) =~ 1 and xmin otherwise'''

    assert isinstance(xmin, float)
    assert isinstance(func(0.5*(xmax+xmin)), float)

    l = xmin
    r = xmax
    while abs(r-l) > tol:
        x = 0.5*(r + l)
        if func(x) > 1.0:
            r = x
        else:
            l = x

    x = 0.5*(r + l)
    return x


class CorralHyperparam:

    def __init__(self,m,T=1000,eta=0.1):
        #self.hyperparam_list = hyperparam_list
        self.m = m# len(self.hyperparam_list)
        self.base_probas = np.ones(self.m)/self.m
        self.gamma = 1.0/T
        self.beta = exp(1/log(T))
        self.rho = np.asarray([2*self.m]*self.m)
        self.etas = np.ones(self.m)*eta
        self.T = T
        self.counter = 0


    def sample_base_index(self):
        sample_array = np.random.choice(range(self.m), 1, p=self.base_probas)
        return sample_array[0]


    def get_distribution(self):
        return self.base_probas

    
    
    def update_distribution(self, arm_idx, reward):
      loss = 1-reward

      l = np.zeros(self.m)
      p = self.base_probas[arm_idx]
      assert(p>1e-8)
      l[arm_idx] = loss/p  #importance weighted loss vector
      probas_new = self.log_barrier_OMD(self.base_probas,l,self.etas)
      self.base_probas = (1-self.gamma)*probas_new + self.gamma*1.0/self.m
      assert(min(self.base_probas) > 1e-8)

      self.update_etas()

      self.counter += 1


    def update_etas(self):
        '''Updates the eta vector'''
        for i in range(self.m):
            if 1.0/self.base_probas[i] > self.rho[i]:
                self.rho[i] = 2.0/self.base_probas[i]
                self.etas[i] = self.beta*self.etas[i]

    def log_barrier_OMD(self,p,loss,etas, tol=1e-5):
        '''Implements Algorithm 2 in the paper
        Updates the probabilities using log barrier function'''
        assert(len(p)==len(loss) and len(loss) == len(etas))
        assert(abs(np.sum(p)-1) < 1e-3)

        xmin = min(loss)
        xmax = max(loss)
        pinv = np.divide(1.0,p)
        thresh = min(np.divide(pinv,etas) + loss) # the max value such that all denominators are positive
        xmax = min(xmax,thresh)

        def log_barrier(x):
            assert isinstance(x,float)
            inv_val_vec = ( np.divide(1.0,p) + etas*(loss-x) )
            if (np.min(np.abs(inv_val_vec))<1e-5):
                print(thresh,xmin,x,loss)
            assert( np.min(np.abs(inv_val_vec))>1e-5)
            val = np.sum( np.divide(1.0,inv_val_vec) )
            return val

        x = binary_search(log_barrier,xmin,xmax,tol)

        assert(abs(log_barrier(x)-1) < 1e-2)

        inv_probas_new = np.divide(1.0,self.base_probas) + etas*(loss-x)
        assert(np.min(inv_probas_new) > 1e-6)
        probas_new = np.divide(1.0,inv_probas_new)
        assert(abs(sum(probas_new)-1) < 1e-1)
        probas_new = probas_new/np.sum(probas_new)

        return probas_new






# class BalancingHyperParam:
    



def train_epsilon_greedy_modsel(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size, MLP = True, 
    representation_layer_size = 10, threshold = .5, epsilons = [.1, .05, .01],
    verbose = False, fit_intercept = True, decaying_epsilon = False, 
    restart_model_full_minimization = False, modselalgo = "Corral"):
    
    # IPython.embed()
    # raise ValueError("asldfkmalsdkfm")

    if modselalgo == "Corral":
        modsel_manager = CorralHyperparam(len(epsilons), T = num_batches) ### hack
    else:
        raise ValueError("Modselalgo type {} not recognized.".format(modselalgo))

    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000, 
        fit_intercept = True)

    model = TorchBinaryLogisticRegression(
        random_init=True,
        alpha=0,
        MLP=MLP,
        representation_layer_size=representation_layer_size,
        dim = train_dataset.dimension
    )

    growing_training_dataset = GrowingNumpyDataSet()
    instantaneous_regrets = []
    instantaneous_accuracies = []
    eps_multiplier = 1.0

    for i in range(num_batches):
        if verbose:
            print("Processing modsel epsilon greedy batch ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)

        with torch.no_grad():
            
            if decaying_epsilon:
                eps_multiplier = 1.0/(np.sqrt(i+1))
            predictions = model.get_thresholded_predictions(batch_X, threshold)
            baseline_predictions = baseline_model.get_thresholded_predictions(batch_X, threshold)


            ### Sample epsilon using model selection
            sample_idx = modsel_manager.sample_base_index()
            epsilon = epsilons[sample_idx]
            print(i, " sample epsilon ", epsilon)
            print("Epsilons distribution ", modsel_manager.get_distribution())


            if torch.cuda.is_available():
                epsilon_greedy_mask = torch.bernoulli(torch.ones(predictions.shape)*epsilon*eps_multiplier).bool().cuda()
            else:
                epsilon_greedy_mask = torch.bernoulli(torch.ones(predictions.shape)*epsilon*eps_multiplier).bool()
  
            mask = torch.max(epsilon_greedy_mask,predictions)

            boolean_labels_y = batch_y.bool()
            accuracy = (torch.sum(mask*boolean_labels_y) +torch.sum( ~mask*~boolean_labels_y))*1.0/batch_size
           
            ### Update mod selection manager
            modesel_reward = (2*torch.sum(mask*boolean_labels_y) - torch.sum(mask))/batch_size

            modsel_manager.update_distribution(sample_idx, modesel_reward.item())

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

                
    print("Finished training modsel epsilon-greedy model {}".format(epsilon))
    test_accuracy = evaluate_model(test_dataset, model, threshold).item()
    print("Final model test accuracy {}".format(test_accuracy))
    return instantaneous_regrets, instantaneous_accuracies, test_accuracy




def train_mahalanobis_modsel(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size, MLP = True, 
    representation_layer_size = 10, threshold = .5, alphas = [1, .1, .01], lambda_reg = 1,
    verbose = False, fit_intercept = True, 
    restart_model_full_minimization = False, modselalgo = "Corral"):
    
    if modselalgo == "Corral":
        modsel_manager = CorralHyperparam(len(epsilons), T = min(num_batches, 1000)) ### hack
    else:
        raise ValueError("Modselalgo type {} not recognized.".format(modselalgo))


    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000, 
        fit_intercept = True)

    dataset_dimension = train_dataset.dimension

    
    model = TorchBinaryLogisticRegression(
        random_init=True,
        alpha=alpha,
        MLP=MLP,
        representation_layer_size=representation_layer_size,
        dim = train_dataset.dimension
    )

    growing_training_dataset = GrowingNumpyDataSet()
    instantaneous_regrets = []
    instantaneous_accuracies = []

    if not MLP:
        covariance  = lambda_reg*torch.eye(dataset_dimension).cuda()
    else:
        covariance = lambda_reg*torch.eye(representation_layer_size).cuda()

    for i in range(num_batches):
        if verbose:
            print("Processing mahalanobis batch ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)


        with torch.no_grad():
            
            ### Sample epsilon using model selection
            sample_idx = modsel_manager.sample_base_index()
            alpha = alphas[sample_idx]
            print(i, " sample alpha ", alpha)
            print("Alphas distribution ", modsel_manager.get_distribution())
            model.alpha = alpha




            inverse_covariance = torch.linalg.inv(covariance)
            optimistic_predictions = model.get_thresholded_predictions(batch_X, threshold, inverse_covariance)

            baseline_predictions = baseline_model.get_thresholded_predictions(batch_X, threshold)


            boolean_labels_y = batch_y.bool()
            accuracy = (torch.sum(optimistic_predictions*boolean_labels_y) +torch.sum( ~optimistic_predictions*~boolean_labels_y))*1.0/batch_size
            

            ### Update mod selection manager
            modesel_reward = (2*torch.sum(mask*boolean_labels_y) - torch.sum(mask))/batch_size
            modsel_manager.update_distribution(sample_idx, modesel_reward.item())


            accuracy_baseline = (torch.sum(baseline_predictions*boolean_labels_y) +torch.sum( ~baseline_predictions*~boolean_labels_y))*1.0/batch_size
            instantaneous_regret = accuracy_baseline - accuracy

            instantaneous_regrets.append(instantaneous_regret.item())
            instantaneous_accuracies.append(accuracy.item())


            filtered_batch_X = batch_X[optimistic_predictions, :]
            
            ### Update the covariance
            filtered_representations_batch = model.get_representation(filtered_batch_X)
            covariance += torch.transpose(filtered_representations_batch, 0,1)@filtered_representations_batch

            filtered_batch_y = batch_y[optimistic_predictions]



        growing_training_dataset.add_data(filtered_batch_X, filtered_batch_y)

        #### Filter the batch using the predictions
        #### Add the accepted points and their labels to the growing training dataset
        model = train_model( model, num_opt_steps, 
                growing_training_dataset, opt_batch_size, 
                restart_model_full_minimization = restart_model_full_minimization)

                
    print("Finished training mahalanobis model alpha - {}".format(alpha))
    test_accuracy = evaluate_model(test_dataset, model, threshold).item()
    return instantaneous_regrets, instantaneous_accuracies, test_accuracy











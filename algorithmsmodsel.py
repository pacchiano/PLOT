import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import random
import itertools
import sys

import IPython

from dataclasses import dataclass
#from torchvision import datasets, transforms
from typing import Any

from math import log, exp

from model_training_utilities import evaluate_model, train_model


from datasets import get_batches, get_dataset_simple, GrowingNumpyDataSet
from newmodels import (
    TorchMultilayerRegression,
    TorchMultilayerRegressionMahalanobis    )

from model_training_utilities import train_model_opt_reg


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

class UCBalgorithm:
    def __init__(self, num_arms, burn_in = 1, min_range = 0, max_range = 1):
        self.num_arms = num_arms
        self.mean_estimators = [0 for _ in range(num_arms)]
        self.counts = [0 for _ in range(num_arms)]
        self.reward_sums = [0 for _ in range(num_arms)]
        self.burn_in = burn_in
        self.min_range = min_range
        self.max_range = max_range

    def update_arm_statistics(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.reward_sums[arm_index] += reward
        self.mean_estimators[arm_index] = self.reward_sums[arm_index]/self.counts[arm_index] 


    def get_ucb_arm(self, confidence_radius ):
        if sum(self.counts) <=  self.burn_in:
            #print("HERE")
            ucb_arm_index = random.choice(range(self.num_arms))
            ucb_arm_value = self.max_range
            lcb_arm_value = self.min_range
        else:
            ucb_bonuses = [confidence_radius*np.sqrt(1/(count + .0000000001)) for count in self.counts ]
            ucb_arm_values = [min(self.mean_estimators[i] + ucb_bonuses[i], self.max_range) for i in range(self.num_arms)]
            ucb_arm_index = np.argmax(ucb_arm_values)
            ucb_arm_value = ucb_arm_values[ucb_arm_index]
            lcb_arm_values = [max(self.mean_estimators[i] - ucb_bonuses[i], self.min_range) for i in range(self.num_arms)]

            lcb_arm_value = lcb_arm_values[ucb_arm_index]
        return ucb_arm_index, ucb_arm_value, lcb_arm_value


class UCBHyperparam:

    def __init__(self,m, burn_in = 1, confidence_radius = 2, 
        min_range = 0, max_range = 1):
        #self.hyperparam_list = hyperparam_list
        self.ucb_algorithm = UCBalgorithm(m, burn_in = 1, min_range = 0, max_range = 1)
        #self.discount_factor = discount_factor
        #self.forced_exploration_factor = forced_exploration_factor
        self.m = m
        self.confidence_radius = confidence_radius
        self.burn_in = burn_in
        self.T = 1

        #self.m = m# len(self.hyperparam_list)
        self.base_probas = np.ones(self.m)/(1.0*self.m)
        #self.importance_weighted_cum_rewards = np.zeros(self.m)
        #self.T = T
        #self.counter = 0
        #self.anytime = False
        #self.forced_exploration_factor = forced_exploration_factor
        #self.discount_factor = discount_factor
        # if self.anytime:
        #     self.T = 1


    def sample_base_index(self):
        index, _, _ = self.ucb_algorithm.get_ucb_arm(self.confidence_radius)
        if self.T <= self.burn_in:
            self.base_probas = np.ones(self.m)/(1.0*self.m)
        else:
            self.base_probas = np.zeros(self.m)
            self.base_probas[index] = 1
        self.T += 1
        return index


    def get_distribution(self):
        return self.base_probas

    
    
    def update_distribution(self, arm_idx, reward, more_info = dict([])):        
        self.ucb_algorithm.update_arm_statistics(arm_idx, reward)





class EXP3Hyperparam:

    def __init__(self,m,T=1000, anytime = False, discount_factor = .9, forced_exploration_factor = 0):
        #self.hyperparam_list = hyperparam_list
        self.m = m# len(self.hyperparam_list)
        self.base_probas = np.ones(self.m)/self.m
        self.importance_weighted_cum_rewards = np.zeros(self.m)
        self.T = T
        self.counter = 0
        self.anytime = False
        self.forced_exploration_factor = forced_exploration_factor

        self.discount_factor = discount_factor

        if self.anytime:
            self.T = 1


    def sample_base_index(self):
        sample_array = np.random.choice(range(self.m), 1, p=self.base_probas)
        return sample_array[0]


    def get_distribution(self):
        return self.base_probas

    
    
    def update_distribution(self, arm_idx, reward, more_info = dict([])):
        self.importance_weighted_cum_rewards[arm_idx] *= self.discount_factor
        self.importance_weighted_cum_rewards[arm_idx] += reward/self.base_probas[arm_idx]
        


        eta = np.sqrt( np.log(self.m)/(self.m*self.T))
        
        normalization_factor = np.sum( np.exp( self.importance_weighted_cum_rewards*eta) )

        self.base_probas = np.exp( self.importance_weighted_cum_rewards*eta )/normalization_factor

        self.counter += 1
        if self.anytime:
            self.T += 1







class CorralHyperparam:

    def __init__(self,m,T=1000,eta=0.1, anytime = False):
        #self.hyperparam_list = hyperparam_list
        self.m = m# len(self.hyperparam_list)
        self.base_probas = np.ones(self.m)/self.m
        self.gamma = 1.0/T
        self.beta = exp(1/log(T))
        self.rho = np.asarray([2*self.m]*self.m)
        self.etas = np.ones(self.m)*eta
        self.T = T
        self.counter = 0
        self.anytime = False
        if self.anytime:
            self.T = 1


    def sample_base_index(self):
        sample_array = np.random.choice(range(self.m), 1, p=self.base_probas)
        return sample_array[0]


    def get_distribution(self):
        return self.base_probas

    
    
    def update_distribution(self, arm_idx, reward, more_info = dict([])):
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
      if self.anytime:
          self.T += 1


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
class BalancingHyperparam:
    def __init__(self, m, putative_bounds_multipliers, delta =0.01, 
        importance_weighted = True, balancing_type = "BalancingSimple", burn_in_pulls = 5 ):
        #self.hyperparam_list = hyperparam_list
        self.m = m
        self.putative_bounds_multipliers = putative_bounds_multipliers
        self.T = 1
        self.delta = delta
        self.algorithm_mask = [1 for _ in range(self.m)]
        self.counter = 0
        #self.distribution_base_parameters = [1.0/(x**2) for x in self.putative_bounds_multipliers]
        self.distribution_base_parameters = [1.0/x for x in self.putative_bounds_multipliers]

        self.reward_statistics = [0 for _ in range(m)]
        self.optimism_statistics = [0 for _ in range(m)]

        self.pessimism_statistics = [0 for _ in range(m)]

        self.normalize_distribution()
        self.importance_weighted = importance_weighted
        self.pulls_per_arm = [0 for _ in range(m)]

        self.all_rewards = 0

        self.burn_in_pulls = burn_in_pulls

        self.balancing_type = balancing_type

    def sample_base_index(self):
        sample_array = np.random.choice(range(self.m), 1, p=self.base_probas)
        return sample_array[0]


    def normalize_distribution(self):
        masked_distribution_base_params = [x*y for (x,y) in zip(self.algorithm_mask, self.distribution_base_parameters)]
        normalization_factor = np.sum(masked_distribution_base_params)
        self.base_probas = [x/normalization_factor for x in masked_distribution_base_params]
       

    def get_distribution(self):
        return self.base_probas


    def update_distribution(self, algo_idx, reward, more_info = dict([])):
        proba = self.base_probas[algo_idx]
        self.pulls_per_arm[algo_idx] += 1
        self.all_rewards += reward





        if proba == 0:
            raise ValueError("Probability of taking this action was zero in balancing")
        if self.importance_weighted:
            self.reward_statistics[algo_idx] += reward/proba
            self.pessimism_statistics[algo_idx]  += more_info["pessimistic_reward_predictions"]/proba
            self.optimism_statistics[algo_idx]  += more_info["optimistic_reward_predictions"]/proba
        else:
            curr_arm_pulls = self.pulls_per_arm[algo_idx]
            curr_avg = self.reward_statistics[algo_idx]
            self.reward_statistics[algo_idx] = (curr_avg*(curr_arm_pulls-1) + reward)/curr_arm_pulls
            curr_pess_avg = self.pessimism_statistics[algo_idx]
            pess_reward = more_info["pessimistic_reward_predictions"]
            self.pessimism_statistics[algo_idx]  += (curr_pess_avg*(curr_arm_pulls-1) + pess_reward)/curr_arm_pulls
            curr_opt_avg = self.optimism_statistics[algo_idx]
            opt_reward = more_info["optimistic_reward_predictions"]
            self.optimism_statistics[algo_idx]  += (curr_opt_avg*(curr_arm_pulls-1) + opt_reward)/curr_arm_pulls


        upper_bounds = []
        lower_bounds = []



        #if self.balancing_type in ["BalancingSimple", "BalancingAnalyticHybrid", "BalancingAnalytic"]


        for i in range(self.m):
            
            putative_multiplier = self.putative_bounds_multipliers[i]
            
            if self.balancing_type == "BalancingSimple":


                if self.importance_weighted:
                    upper_bounds.append(self.reward_statistics[i] + (putative_multiplier**2)*np.sqrt(self.T) )
                    lower_bounds.append(self.reward_statistics[i] - putative_multiplier*np.sqrt(self.T) )
                else:
                    upper_bounds.append(self.reward_statistics[i] + (putative_multiplier+1)/np.sqrt(self.pulls_per_arm[i]) )
                    lower_bounds.append(self.reward_statistics[i] - 1.0/np.sqrt(self.pulls_per_arm[i] + .000000001) )


            elif self.balancing_type == "BalancingAnalyticHybrid":

                if self.importance_weighted:
                    upper_bounds.append(self.optimism_statistics[i] + (putative_multiplier**2)*np.sqrt(self.T) )
                    lower_bounds.append(self.reward_statistics[i] - putative_multiplier*np.sqrt(self.T) )
                else:
                    upper_bounds.append(self.optimism_statistics[i] )
                    lower_bounds.append(self.reward_statistics[i] - 1.0/np.sqrt(self.pulls_per_arm[i]+ .000000001) )


            elif self.balancing_type == "BalancingAnalytic":

                if self.importance_weighted:
                    upper_bounds.append(self.optimism_statistics[i] + (putative_multiplier**2)*np.sqrt(self.T) ) 
                    lower_bounds.append(self.pessimism_statistics[i] - putative_multiplier*np.sqrt(self.T) )   
                else:
                    upper_bounds.append(self.optimism_statistics[i] ) 
                    lower_bounds.append(self.pessimism_statistics[i] )


            

        print(self.balancing_type)
        print("Rewards statistics ", self.reward_statistics)
        print("pulls per arm ", self.pulls_per_arm)
        print("Balancing Upper Bounds ", upper_bounds)
        print("Balancing Lower Bounds ", lower_bounds)
        print("Balancing algorithm masks ", self.algorithm_mask)
        print("Balancing probabilities ",self.base_probas)

        max_lower_bound = np.max(lower_bounds)







        for i, mask in enumerate(self.algorithm_mask):
            if mask  == 0: ## If the mask equals zero, get rid of the 
                continue

            if self.pulls_per_arm[i] < self.burn_in_pulls:
                continue

            if upper_bounds[i] < max_lower_bound:

                print("The balancing algorithm eliminated a base learner.")
                self.algorithm_mask[i] = 0

        self.T += 1

        self.normalize_distribution()



# class BalancingHyperParam:
class EpochBalancingHyperparam:
    def __init__(self, m, putative_bounds_multipliers, delta =0.01, 
        burn_in_pulls = 10, balancing_test_multiplier = .1 ):
        self.m = m
        self.putative_bounds_multipliers = putative_bounds_multipliers
        ### check these putative bounds are going up
        curr_val = -float("inf")
        for x in self.putative_bounds_multipliers:
            if x < curr_val:
                raise ValueError("The putative bound multipliers for EpochBalancing are not in increasing order.")

            curr_val = x

        self.balancing_test_multiplier = balancing_test_multiplier

        self.T = 1
        self.delta = delta
        
        self.min_suriving_algo_index = 0
        self.algorithm_mask = [1 for _ in range(self.m)]


        self.counter = 0
        self.distribution_base_parameters = [1.0/x for x in self.putative_bounds_multipliers]
        #self.distribution_base_parameters = [1.0/(x**2) for x in self.putative_bounds_multipliers]
        
        self.all_rewards = 0
        self.epoch_reward = 0
        self.epoch_steps = 0
        self.epoch_index = 0

        self.epoch_optimistic_estimators = 0
        self.epoch_pessimistic_estimators = 0

        self.normalize_distribution()
        self.burn_in_pulls = burn_in_pulls


    def sample_base_index(self):
        sample_array = np.random.choice(range(self.m), 1, p=self.base_probas)
        return sample_array[0]


    def normalize_distribution(self):
        masked_distribution_base_params = [x*y for (x,y) in zip(self.algorithm_mask, self.distribution_base_parameters)]
        normalization_factor = np.sum(masked_distribution_base_params)
        self.base_probas = [x/normalization_factor for x in masked_distribution_base_params]
       

    def get_distribution(self):
        return self.base_probas


    def update_distribution(self, algo_idx, reward, more_info = dict([])):
        #proba = self.base_probas[algo_idx]
        #self.pulls_per_arm[algo_idx] += 1
        self.all_rewards += reward

        self.epoch_reward += reward
        self.epoch_steps += 1
        self.epoch_optimistic_estimators += more_info["optimistic_reward_predictions"]
        self.epoch_pessimistic_estimators += more_info["pessimistic_reward_predictions"]


        print("Curr reward ", reward)
        print("Opt reward pred ", more_info["optimistic_reward_predictions"])
        print("Pess reward pred ", more_info["pessimistic_reward_predictions"])
        print("All rewards ", self.all_rewards)
        print("Epoch reward ", self.epoch_reward)
        print("Epoch Steps ", self.epoch_steps)
        print("Epoch index ", self.epoch_index)
        print("Epoch optimistic estimators ", self.epoch_optimistic_estimators)
        print("Epoch pessimistic estimators ", self.epoch_pessimistic_estimators)

        print("Balancing algorithm masks ", self.algorithm_mask)
        print("Balancing probabilities ",self.base_probas)

        self.T += 1

        print("Test low ", self.epoch_reward - self.balancing_test_multiplier*np.sqrt(self.epoch_steps), " > ", self.epoch_optimistic_estimators)
        print("Test high ", self.epoch_reward + self.balancing_test_multiplier*np.sqrt(self.epoch_steps), " < ", self.epoch_pessimistic_estimators)



        ### TEST:
        if (self.epoch_optimistic_estimators  < self.epoch_reward - self.balancing_test_multiplier*np.sqrt(self.epoch_steps) or self.epoch_reward + self.balancing_test_multiplier*np.sqrt(self.epoch_steps) < self.epoch_pessimistic_estimators) and self.epoch_steps > self.burn_in_pulls and sum(self.algorithm_mask) > 1:

            ### RESET EPOCH
            self.epoch_reward = 0
            self.epoch_steps = 0
            self.epoch_optimistic_estimators = 0
            self.epoch_pessimistic_estimators = 0

            self.epoch_index += 1

            #IPython.embed()

            self.min_suriving_algo_index += 1
            for i in range(self.min_suriving_algo_index):
                self.algorithm_mask[i] = 0





        self.normalize_distribution()







def train_epsilon_greedy_modsel(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size, 
    representation_layer_sizes = [10, 10], threshold = .5, epsilons = [.1, .05, .01],
    verbose = False, decaying_epsilon = False, 
    restart_model_full_minimization = False, modselalgo = "Corral"):
    
    # IPython.embed()
    # raise ValueError("asldfkmalsdkfm")

    if modselalgo == "Corral":
        modsel_manager = CorralHyperparam(len(epsilons), T = num_batches) 
    elif modselalgo == "CorralAnytime":
        modsel_manager = CorralHyperparam(len(epsilons), T = num_batches, anytime = True) 
    elif modselalgo == "BalancingSimple":
        modsel_manager = SimpleBalancingHyperparam(len(epsilons), 
            [1 for _ in range(len(epsilons))], delta =0.01)
    else:
        raise ValueError("Modselalgo type {} not recognized.".format(modselalgo))

    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000, 
        )


    model = TorchMultilayerRegression(
        representation_layer_sizes=representation_layer_sizes,
        dim = train_dataset.dimension,
        output_filter = 'logistic',
    )


    # model = TorchBinaryLogisticRegression(
    #     random_init=True,
    #     alpha=0,
    #     MLP=MLP,
    #     representation_layer_size=representation_layer_size,
    #     dim = train_dataset.dimension
    # )

    growing_training_dataset = GrowingNumpyDataSet()
    instantaneous_regrets = []
    instantaneous_accuracies = []
    eps_multiplier = 1.0

    modselect_info = []
    
    num_positives = []
    num_negatives = []
    false_neg_rates = []
    false_positive_rates = []



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


            modselect_info.append(modsel_manager.get_distribution())


            if torch.cuda.is_available():
                epsilon_greedy_mask = torch.bernoulli(torch.ones(predictions.shape)*epsilon*eps_multiplier).bool()#.cuda()
            else:
                epsilon_greedy_mask = torch.bernoulli(torch.ones(predictions.shape)*epsilon*eps_multiplier).bool()
  
            mask = torch.max(epsilon_greedy_mask,predictions)

            boolean_labels_y = batch_y.bool().squeeze()
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
            false_positive_rates.append(false_positive_rate.item())            



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


    results = dict([])
    results["instantaneous_regrets"] = instantaneous_regrets
    results["test_accuracy"] = test_accuracy
    results["instantaneous_accuracies"] = instantaneous_accuracies
    results["num_negatives"] = num_negatives
    results["num_positives"] = num_positives
    results["false_neg_rates"] = false_neg_rates
    results["false_positive_rates"] = false_positive_rates
    results["modselect_info"] = modselect_info

    return results# instantaneous_regrets, instantaneous_accuracies, test_accuracy, modselect_info




def train_mahalanobis_modsel(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size,
    representation_layer_sizes = [10, 10], threshold = .5, alphas = [1, .1, .01], lambda_reg = 1,
    verbose = False,
    restart_model_full_minimization = False, modselalgo = "Corral", split = False):
    

    num_alphas = len(alphas)

    if modselalgo == "Corral":
        modsel_manager = CorralHyperparam(len(alphas), T = num_batches) ### hack
    elif modselalgo == "CorralAnytime":
        modsel_manager = CorralHyperparam(len(alphas), T = num_batches, anytime = True) 
    elif modselalgo == "BalancingSimple":
        modsel_manager = BalancingHyperparam(len(alphas), 
           [ x*representation_layer_sizes[-1] for x in alphas], delta =0.01, balancing_type = "BalancingSimple" )
    elif modselalgo == "BalancingAnalytic":
        modsel_manager = BalancingHyperparam(len(alphas), 
            [ x*representation_layer_sizes[-1] for x in alphas], delta =0.01, balancing_type = "BalancingAnalytic")
    elif modselalgo == "BalancingAnalyticHybrid":
        modsel_manager = BalancingHyperparam(len(alphas), 
            [ x*representation_layer_sizes[-1] for x in alphas], delta =0.01, balancing_type = "BalancingAnalyticHybrid")
    elif modselalgo == "EpochBalancing":
        modsel_manager = EpochBalancingHyperparam(len(alphas),   [ x*representation_layer_sizes[-1] for x in alphas])
    else:
        raise ValueError("Modselalgo type {} not recognized.".format(modselalgo))
    alpha = alphas[0]
    ### THE above is going to fail for linear representations.


    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000)


    # IPython.embed()
    # raise ValueError("asdlf;km")



    dataset_dimension = train_dataset.dimension


    if not split:


        model = TorchMultilayerRegressionMahalanobis(
            alpha=alpha,
            representation_layer_sizes=representation_layer_sizes,
            dim = train_dataset.dimension,
            output_filter = 'logistic'
        )

        growing_training_dataset = GrowingNumpyDataSet()

    if split:

        models = [ TorchMultilayerRegressionMahalanobis(
            alpha=alpha,
            representation_layer_sizes=representation_layer_sizes,
            dim = train_dataset.dimension,
            output_filter = 'logistic'
        ) for alpha in alphas]

        growing_training_datasets = [GrowingNumpyDataSet() for _ in range(num_alphas)]





    instantaneous_regrets = []
    instantaneous_accuracies = []
    modselect_info = []
    instantaneous_baseline_accuracies = []

    num_positives = []
    num_negatives = []
    false_neg_rates = []
    false_positive_rates = []

    optimistic_reward_predictions_list = []

    pessimistic_reward_predictions_list = []

    rewards = []

    if len(representation_layer_sizes) == 0:
        covariance  = lambda_reg*torch.eye(dataset_dimension)#.cuda()
    else:
        if torch.cuda.is_available():

            covariance = lambda_reg*torch.eye(representation_layer_sizes[-1])#.cuda()
        else:
            covariance = lambda_reg*torch.eye(representation_layer_sizes[-1])


    for i in range(num_batches):
        if verbose:
            print("Processing mahalanobis batch ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)


            
        ### Sample epsilon using model selection
        sample_idx = modsel_manager.sample_base_index()
        alpha = alphas[sample_idx]
        print(i, " batch number ", " sample alpha ", alpha)
        print("Alphas distribution ", modsel_manager.get_distribution())
        print("is split {}".format(split))
        if not split:
           model.alpha = alpha

        else:
            model = models[sample_idx]
            growing_training_dataset = growing_training_datasets[sample_idx]

        modselect_info.append(modsel_manager.get_distribution())

        with torch.no_grad():

            inverse_covariance = torch.linalg.inv(covariance)



            ##### Get thresholded predictions and uncertanties
            optimistic_thresholded_predictions, optimistic_prob_predictions, pessimistic_prob_predictions = model.get_all_predictions_info(batch_X, threshold, inverse_covariance)
            # IPython.embed()

            # raise ValueError("alsdkfm")





            baseline_predictions = baseline_model.get_thresholded_predictions(batch_X, threshold)


            boolean_labels_y = batch_y.bool().squeeze()
            accuracy = (torch.sum(optimistic_thresholded_predictions*boolean_labels_y) +torch.sum( ~optimistic_thresholded_predictions*~boolean_labels_y))*1.0/batch_size


            #### TOTAL NUM POSITIVES
            total_num_positives = torch.sum(optimistic_thresholded_predictions)

            #### TOTAL NUM NEGATIVES   
            total_num_negatives = torch.sum(~optimistic_thresholded_predictions)

            #### FALSE NEGATIVE RATE
            false_neg_rate = torch.sum(~optimistic_thresholded_predictions*boolean_labels_y)*1.0/(torch.sum(~optimistic_thresholded_predictions)+.00000000001)


            #### FALSE POSITIVE RATE            
            false_positive_rate = torch.sum(optimistic_thresholded_predictions*~boolean_labels_y)*1.0/(torch.sum(optimistic_thresholded_predictions)+.00000000001)
            



            num_positives.append(total_num_positives.item())
            num_negatives.append(total_num_negatives.item())
            false_neg_rates.append(false_neg_rate.item())
            false_positive_rates.append(false_positive_rate)            




            ### Update mod selection manager
            modsel_info = dict([])
            

            modesel_reward = (2*torch.sum(optimistic_thresholded_predictions*boolean_labels_y) - torch.sum(optimistic_thresholded_predictions))/batch_size
            rewards.append(modesel_reward)


            baseline_reward = (2*torch.sum(baseline_predictions*boolean_labels_y) - torch.sum(baseline_predictions))/batch_size



            optimistic_reward_predictions_list.append(((2*torch.sum(optimistic_thresholded_predictions*optimistic_prob_predictions) - torch.sum(optimistic_thresholded_predictions))/batch_size).item())
            modsel_info["optimistic_reward_predictions"] = optimistic_reward_predictions_list[-1]

            pessimistic_reward_predictions_list.append(((2*torch.sum(optimistic_thresholded_predictions*pessimistic_prob_predictions) - torch.sum(optimistic_thresholded_predictions))/batch_size).item())
            modsel_info["pessimistic_reward_predictions"] = pessimistic_reward_predictions_list[-1]

            modsel_manager.update_distribution(sample_idx, modesel_reward.item(), modsel_info )


            accuracy_baseline = (torch.sum(baseline_predictions*boolean_labels_y) +torch.sum( ~baseline_predictions*~boolean_labels_y))*1.0/batch_size
            #IPython.embed()

            instantaneous_baseline_accuracies.append(accuracy_baseline)

            instantaneous_regret = baseline_reward - modesel_reward
          
            print("Baseline Reward - {}".format(baseline_reward))
            print("Modsel Reward -   {}".format(modesel_reward))

            print("Baseline Accuracy - {}".format(accuracy_baseline))
            print("Accuracy alph reg - {}".format(accuracy))


            instantaneous_regrets.append(instantaneous_regret.item())
            instantaneous_accuracies.append(accuracy.item())


            filtered_batch_X = batch_X[optimistic_thresholded_predictions, :]
            
            ### Update the covariance
            filtered_representations_batch = model.get_representation(filtered_batch_X)
            covariance += torch.transpose(filtered_representations_batch, 0,1)@filtered_representations_batch

            filtered_batch_y = batch_y[optimistic_thresholded_predictions]



        growing_training_dataset.add_data(filtered_batch_X, filtered_batch_y)

        #### Filter the batch using the predictions
        #### Add the accepted points and their labels to the growing training dataset
        model = train_model( model, num_opt_steps, 
                growing_training_dataset, opt_batch_size, 
                restart_model_full_minimization = restart_model_full_minimization)

                
    print("Finished training mahalanobis model alpha - {}".format(alpha))
    test_accuracy = evaluate_model(test_dataset, model, threshold).item()

    results = dict([])
    results["instantaneous_regrets"] = instantaneous_regrets
    results["test_accuracy"] = test_accuracy
    results["instantaneous_accuracies"] = instantaneous_accuracies

    results["instantaneous_baseline_accuracies"] = instantaneous_baseline_accuracies

    results["num_negatives"] = num_negatives
    results["num_positives"] = num_positives
    results["false_neg_rates"] = false_neg_rates
    results["false_positive_rates"] = false_positive_rates
    results["modselect_info"] = modselect_info

    results["rewards"] = rewards
    results["optimistic_reward_predictions"] = optimistic_reward_predictions_list
    results["pessimistic_reward_predictions"] = pessimistic_reward_predictions_list

    return results# instantaneous_regrets, instantaneous_accuracies, test_accuracy, modselect_info














def train_opt_reg_modsel(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size,
    representation_layer_sizes = [10, 10], threshold = .5, regs = [1, .1, .01],
    verbose = False,
    restart_model_full_minimization = False, modselalgo = "Corral", split = False):
    

    # IPython.embed()
    # raise ValueError("train opt reg modsel")

    num_regs = len(regs)

    if modselalgo == "Corral":
        modsel_manager = CorralHyperparam(len(regs), T = num_batches) ### hack
    elif modselalgo == "CorralAnytime":
        modsel_manager = CorralHyperparam(len(regs), T = num_batches, anytime = True) 
    elif modselalgo == "BalancingSimple":
        modsel_manager = BalancingHyperparam(len(regs), 
           [ x*representation_layer_sizes[-1] for x in regs], delta =0.01, balancing_type = "BalancingSimple" )
    elif modselalgo == "BalancingAnalytic":
        modsel_manager = BalancingHyperparam(len(regs), 
            [ x*representation_layer_sizes[-1] for x in regs], delta =0.01, balancing_type = "BalancingAnalytic")
    elif modselalgo == "BalancingAnalyticHybrid":
        modsel_manager = BalancingHyperparam(len(regs), 
            [ x*representation_layer_sizes[-1] for x in regs], delta =0.01, balancing_type = "BalancingAnalyticHybrid")
    elif modselalgo == "EpochBalancing":
        modsel_manager = EpochBalancingHyperparam(len(regs), [max(x, .0000000001) for x in regs])
    else:
        raise ValueError("Modselalgo type {} not recognized.".format(modselalgo))
    reg = regs[0]
    ### THE above is going to fail for linear representations.


    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000)


    dataset_dimension = train_dataset.dimension


    if not split:

        model = TorchMultilayerRegression(
            representation_layer_sizes=representation_layer_sizes,
            dim = train_dataset.dimension,
            output_filter = 'logistic'
        )

        growing_training_dataset = GrowingNumpyDataSet()

    if split:

        models = [ TorchMultilayerRegression(
            representation_layer_sizes=representation_layer_sizes,
            dim = train_dataset.dimension,
            output_filter = 'logistic'
        ) for reg in regs]

        growing_training_datasets = [GrowingNumpyDataSet() for _ in range(num_regs)]





    instantaneous_regrets = []
    instantaneous_accuracies = []
    instantaneous_baseline_accuracies = []
    modselect_info = []


    num_positives = []
    num_negatives = []
    false_neg_rates = []
    false_positive_rates = []

    optimistic_reward_predictions_list = []

    pessimistic_reward_predictions_list = []

    rewards = []
    baseline_rewards = []


    for i in range(num_batches):
        if verbose:
            print("Processing OptReg batch ", i, " ", dataset)
        batch_X, batch_y = train_dataset.get_batch(batch_size)

            
        ### Sample epsilon using model selection
        sample_idx = modsel_manager.sample_base_index()
        reg = regs[sample_idx]
        print(i, " batch number ", " sample opt_reg ", reg)
        print("OptReg distribution ", modsel_manager.get_distribution())
        if split:
            model = models[sample_idx]
            growing_training_dataset = growing_training_datasets[sample_idx]

        
        modselect_info.append(modsel_manager.get_distribution())






        ##### Train optimistic model.
        model = train_model_opt_reg(model, num_opt_steps, growing_training_dataset, 
            batch_X, opt_batch_size, opt_reg = reg, restart_model_full_minimization = restart_model_full_minimization )
        optimistic_prob_predictions = model.predict(batch_X)
        optimistic_thresholded_predictions = model.get_thresholded_predictions(batch_X, threshold)
        print("Finished training optimistic OptReg model reg - {}".format(reg))
        print("optimistic predictions ", optimistic_prob_predictions)


        ### Train pessimistic model.
        model = train_model_opt_reg(model, num_opt_steps, growing_training_dataset, 
            batch_X, opt_batch_size, opt_reg = -reg, restart_model_full_minimization = restart_model_full_minimization )
        pessimistic_prob_predictions = model.predict(batch_X)
        print("Finished training pessimistic OptReg model reg - {}".format(reg))
        print("pessimistic predictions ", pessimistic_prob_predictions)
        #IPython.embed()
        #raise ValueError("asdf")
        if len(regs) > 1:
            print("is split {}".format(split))
        print("        ########################################")
        print("prediction differential ", optimistic_prob_predictions - pessimistic_prob_predictions)
        print("        ########################################")







        with torch.no_grad():        

            baseline_predictions = baseline_model.get_thresholded_predictions(batch_X, threshold)

            boolean_labels_y = batch_y.bool().squeeze()
            accuracy = (torch.sum(optimistic_thresholded_predictions*boolean_labels_y) +torch.sum( ~optimistic_thresholded_predictions*~boolean_labels_y))*1.0/batch_size


            #### TOTAL NUM POSITIVES
            total_num_positives = torch.sum(optimistic_thresholded_predictions)

            #### TOTAL NUM NEGATIVES   
            total_num_negatives = torch.sum(~optimistic_thresholded_predictions)

            #### FALSE NEGATIVE RATE
            false_neg_rate = torch.sum(~optimistic_thresholded_predictions*boolean_labels_y)*1.0/(torch.sum(~optimistic_thresholded_predictions)+.00000000001)


            #### FALSE POSITIVE RATE            
            false_positive_rate = torch.sum(optimistic_thresholded_predictions*~boolean_labels_y)*1.0/(torch.sum(optimistic_thresholded_predictions)+.00000000001)
            



            num_positives.append(total_num_positives.item())
            num_negatives.append(total_num_negatives.item())
            false_neg_rates.append(false_neg_rate.item())
            false_positive_rates.append(false_positive_rate)            



            ### Update mod selection manager
            modsel_info = dict([])
            

            modesel_reward = (2*torch.sum(optimistic_thresholded_predictions*boolean_labels_y) - torch.sum(optimistic_thresholded_predictions))/batch_size

            rewards.append(modesel_reward)

            baseline_reward = (2*torch.sum(baseline_predictions*boolean_labels_y) - torch.sum(baseline_predictions))/batch_size

            baseline_rewards.append(baseline_reward)

            optimistic_reward_predictions_list.append(((2*torch.sum(optimistic_thresholded_predictions*optimistic_prob_predictions) - torch.sum(optimistic_thresholded_predictions))/batch_size).item())

            modsel_info["optimistic_reward_predictions"] = optimistic_reward_predictions_list[-1]



            pessimistic_reward_predictions_list.append(((2*torch.sum(optimistic_thresholded_predictions*pessimistic_prob_predictions) - torch.sum(optimistic_thresholded_predictions))/batch_size).item())

            modsel_info["pessimistic_reward_predictions"] = pessimistic_reward_predictions_list[-1]


            #IPython.embed()

            modsel_manager.update_distribution(sample_idx, modesel_reward.item(), modsel_info )


            accuracy_baseline = (torch.sum(baseline_predictions*boolean_labels_y) +torch.sum( ~baseline_predictions*~boolean_labels_y))*1.0/batch_size
            print("Baseline Accuracy - {}".format(accuracy_baseline))
            print("Accuracy opt reg  - {}".format(accuracy))


            #IPython.embed()

            instantaneous_baseline_accuracies.append(accuracy_baseline)

            instantaneous_regret = baseline_reward - modesel_reward

            instantaneous_regrets.append(instantaneous_regret.item())
            instantaneous_accuracies.append(accuracy.item())


            filtered_batch_X = batch_X[optimistic_thresholded_predictions, :]
            
            filtered_batch_y = batch_y[optimistic_thresholded_predictions]



        growing_training_dataset.add_data(filtered_batch_X, filtered_batch_y)

                
    print("Testing accuracy for pessimistic model reg - {}".format(reg))
    test_accuracy = evaluate_model(test_dataset, model, threshold).item()
    print("Accuracy {}".format(test_accuracy))
    results = dict([])
    results["instantaneous_regrets"] = instantaneous_regrets
    results["test_accuracy"] = test_accuracy
    results["instantaneous_accuracies"] = instantaneous_accuracies
    results["instantaneous_baseline_accuracies"] = instantaneous_baseline_accuracies
    results["num_negatives"] = num_negatives
    results["num_positives"] = num_positives
    results["false_neg_rates"] = false_neg_rates
    results["false_positive_rates"] = false_positive_rates
    results["modselect_info"] = modselect_info


    results["rewards"] = rewards
    results["optimistic_reward_predictions"] = optimistic_reward_predictions_list
    results["pessimistic_reward_predictions"] = pessimistic_reward_predictions_list
    


    return results




















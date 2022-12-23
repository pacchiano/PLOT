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



# def intersect(interval1, interval2):
#     if interval1[0] > interval1[1] or interval2[0] > interval2[1]:
#         raise ValueError("Intervals are malformed")

#     return max(interval1[0], interval2[0]) <= min(interval1[1], interval2[1]):
        
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
    def __init__(self, num_arms, burn_in = 1, min_range = -float("inf"), max_range = float("inf")):
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
            #ucb_arm_index = np.argmax(ucb_arm_values)
            ucb_arm_values = np.array(ucb_arm_values)
            ucb_arm_index = np.random.choice(np.flatnonzero(ucb_arm_values == ucb_arm_values.max()))
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

    def __init__(self,m,T=1000, anytime = False, discount_factor = .9, 
        eta_multiplier = 1, forced_exploration_factor = 0):
        #self.hyperparam_list = hyperparam_list
        self.m = m# len(self.hyperparam_list)
        self.base_probas = np.ones(self.m)/self.m
        self.importance_weighted_cum_rewards = np.zeros(self.m)
        self.T = T
        self.counter = 0
        self.anytime = False
        self.forced_exploration_factor = forced_exploration_factor
        self.eta_multiplier = eta_multiplier
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
        


        eta = self.eta_multiplier*np.sqrt( np.log(self.m)/(self.m*self.T))
        
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




### when the descending keyword is active the balancing algorithm starts with 
### high putative bounds and reduces them
class BalancingHyperparamDoubling:
    def __init__(self, m, initial_putative_bound, delta =0.01, 
        balancing_test_multiplier = 1 , resurrecting = False, classic = False, empirical = False ):
        
        self.minimum_putative = .0001
        self.maximum_putative = 10000

        self.classic = classic

        self.m = m
        self.resurrecting = resurrecting
        self.initial_putative_bound = max(initial_putative_bound, self.minimum_putative)
        self.putative_bounds_multipliers = [max(initial_putative_bound, self.minimum_putative) for _ in range(m)]
        ### check these putative bounds are going up
        curr_val = -float("inf")
        for x in self.putative_bounds_multipliers:
            if x < curr_val:
                raise ValueError("The putative bound multipliers for EpochBalancing are not in increasing order.")

            curr_val = x

        self.balancing_test_multiplier = balancing_test_multiplier

        self.T = 1
        self.delta = delta
        
        #self.min_suriving_algo_index = 0
        self.algorithm_mask = [1 for _ in range(self.m)]



        self.all_rewards = 0

        ### these store the optimistic and pessimistic estimators of Vstar for all 
        ### base algorithms.
        self.optimistic_estimators = [0 for _ in range(self.m)]
        self.pessimistic_estimators = [0 for _ in range(self.m)]

        self.cumulative_rewards = [0 for _ in range(self.m)]
        self.mean_rewards = [0 for _ in range(self.m)]

        self.num_plays = [0 for _ in range(self.m)]

        self.vstar_lowerbounds = [-float("inf") for _ in range(self.m)]

        self.vstar_upperbounds = [float("inf") for _ in range(self.m)]


        self.normalize_distribution()
        


    def sample_base_index(self):
        if self.classic:
            evaluated_putative_bounds = [self.putative_bounds_multipliers[i]*np.sqrt(self.num_plays[i]) for i in range(self.m)]
            return np.argmin(evaluated_putative_bounds)
        else:
            if sum([np.isnan(x) for x in self.base_probas]) > 0:
                print("Found Nan Values in the sampling procedure for base index")
                IPython.embed()
            sample_array = np.random.choice(range(self.m), 1, p=self.base_probas)
            return sample_array[0]


    def normalize_distribution(self):
        if self.classic:
            self.base_probas = [0 for _ in range(self.m)]
            self.base_probas[self.sample_base_index()] = 1 

        else:
            self.distribution_base_parameters = [1.0/(x**2) for x in self.putative_bounds_multipliers]

            normalization_factor = np.sum(self.distribution_base_parameters)
            self.base_probas = [x/normalization_factor for x in self.distribution_base_parameters]
    


    def get_distribution(self):
        return self.base_probas


    def get_misspecified_algos(self,lower_bounds, upper_bounds):
        misspecified_algo_indices = []

        max_lower_bound = max(lower_bounds)

        misspecified_algo_indices = []
        wellspecified_algo_indices = []
        for i in range(self.m):
            if upper_bounds[i] < max_lower_bound:
                misspecified_algo_indices.append(i)
            else:
                wellspecified_algo_indices.append(i)
        return misspecified_algo_indices, wellspecified_algo_indices




    def update_distribution(self, algo_idx, reward, more_info = dict([])):
        self.all_rewards += reward

        self.cumulative_rewards[algo_idx] += reward
        self.num_plays[algo_idx] += 1

        #### Update average reward per algorithm so far. 
        self.mean_rewards[algo_idx] = self.cumulative_rewards[algo_idx]*1.0/self.num_plays[algo_idx]


        vstar_lowerbounds = [0 for _ in range(self.m)]
        vstar_upperbounds = [0 for _ in range(self.m)]

        for i in range(self.m):
            vstar_lowerbounds[i] = self.mean_rewards[i] - self.balancing_test_multiplier/np.sqrt(self.num_plays[i])
            vstar_upperbounds[i] =  self.mean_rewards[i] + self.putative_bounds_multipliers[i]/np.sqrt(self.num_plays[i])



        for i in range(self.m):
            self.vstar_lowerbounds[i] = self.mean_rewards[i] - self.balancing_test_multiplier/np.sqrt(self.num_plays[i])
            self.vstar_upperbounds[i] =  self.mean_rewards[i] + self.putative_bounds_multipliers[i]/np.sqrt(self.num_plays[i])

        
        misspecified_algo_indices, wellspecified_algo_indices = self.get_misspecified_algos(vstar_lowerbounds, vstar_upperbounds)

        ### all putative bounds for misspecified algo indices need to be increased  
        for i in misspecified_algo_indices:
            new_putative_bound = 2*self.putative_bounds_multipliers[i]
            self.putative_bounds_multipliers[i] = min(new_putative_bound, self.maximum_putative)


        ### Compute upper bounds using half the current putative bounds
        vstar_upperbounds = [0 for _ in range(self.m)]

        for i in range(self.m):
            vstar_upperbounds[i] =  self.mean_rewards[i] + .5*self.putative_bounds_multipliers[i]/np.sqrt(self.num_plays[i])


        misspecified_algo_indices, wellspecified_algo_indices = self.get_misspecified_algos(vstar_lowerbounds, vstar_upperbounds)

        ### Misspecified algorithms shouldn't be halved. Well specified should.
        ### all putative bounds for misspecified algo indices need to be increased (only in resurrecting mode)  

        if self.resurrecting:
            for i in wellspecified_algo_indices:
                new_putative_bound = .5*self.putative_bounds_multipliers[i]
                self.putative_bounds_multipliers[i] = max(new_putative_bound, self.minimum_putative)


        for i in range(self.m):
            self.vstar_lowerbounds[i] = self.mean_rewards[i] - self.balancing_test_multiplier/np.sqrt(self.num_plays[i])
            self.vstar_upperbounds[i] =  self.mean_rewards[i] + self.putative_bounds_multipliers[i]/np.sqrt(self.num_plays[i])




        print("Curr reward ", reward)
        print("Opt reward pred ", more_info["optimistic_reward_predictions"])
        print("Pess reward pred ", more_info["pessimistic_reward_predictions"])
        print("All rewards ", self.all_rewards)
        print("Cumulative rewards ", self.cumulative_rewards)
        print("Num plays ", self.num_plays)
        print("Mean rewards ", self.mean_rewards)
        print("Balancing algorithm masks ", self.algorithm_mask)
        print("Balancing probabilities ",self.base_probas)

        self.T += 1



        self.normalize_distribution()









class BalancingHyperparamSharp:
    ### Burn in pulls does not work right now
    def __init__(self, m, putative_bounds_multipliers, delta =0.01, 
        burn_in_pulls = 10, balancing_test_multiplier = 1, uniform_sampling = False ):
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


        #self.counter = 0
        self.uniform_sampling = uniform_sampling

        #self.distribution_base_parameters = [1.0/x for x in self.putative_bounds_multipliers]
        self.distribution_base_parameters = [1.0/(x**2) for x in self.putative_bounds_multipliers]
        #self.base_probas = 

        self.all_rewards = 0


        #self.epoch_reward = 0
        #self.epoch_steps = 0
        #self.epoch_index = 0

        #self.epoch_optimistic_estimators = 0
        #self.epoch_pessimistic_estimators = 0

        ### these store the optimistic and pessimistic estimators of Vstar for all 
        ### base algorithms.
        self.optimistic_estimators = [0 for _ in range(self.m)]
        self.pessimistic_estimators = [0 for _ in range(self.m)]

        self.cumulative_rewards = [0 for _ in range(self.m)]
        self.mean_rewards = [0 for _ in range(self.m)]

        self.num_plays = [0 for _ in range(self.m)]

        self.vstar_lowerbounds = [-float("inf") for _ in range(self.m)]

        self.vstar_upperbounds = [float("inf") for _ in range(self.m)]


        self.normalize_distribution()
        self.burn_in_pulls = burn_in_pulls


    def sample_base_index(self):
        if sum([np.isnan(x) for x in self.base_probas]) > 0:
            print("Found Nan Values")
            IPython.embed()
        sample_array = np.random.choice(range(self.m), 1, p=self.base_probas)
        return sample_array[0]


    def normalize_distribution(self):
        if not self.uniform_sampling:
            #self.distribution_base_parameters = [1.0/x for x in self.putative_bounds_multipliers]
            #else:
            masked_distribution_base_params = [x*y for (x,y) in zip(self.algorithm_mask, self.distribution_base_parameters)]
            normalization_factor = np.sum(masked_distribution_base_params)
            self.base_probas = [x/normalization_factor for x in masked_distribution_base_params]
       
        else:
            self.base_probas = np.ones(self.m)/(1.0*self.m)
    


    def get_distribution(self):
        return self.base_probas


    def update_distribution(self, algo_idx, reward, more_info = dict([])):
        #proba = self.base_probas[algo_idx]
        #self.pulls_per_arm[algo_idx] += 1
        self.all_rewards += reward

        self.cumulative_rewards[algo_idx] += reward
        self.num_plays[algo_idx] += 1

        #### Update average reward per algorithm so far. 
        self.mean_rewards[algo_idx] = self.cumulative_rewards[algo_idx]*1.0/self.num_plays[algo_idx]


        self.vstar_lowerbounds[algo_idx] = self.mean_rewards[algo_idx] - self.balancing_test_multiplier/np.sqrt(self.num_plays[algo_idx])

        ### Using the putative bounds:
        #self.vstar_upperbounds[algo_idx] = self.mean_rewards[algo_idx] + self.putative_bounds_multipliers[algo_idx]/np.sqrt(self.num_plays[algo_idx])

        ### alternatively
        self.vstar_upperbounds[algo_idx] = more_info["optimistic_reward_predictions"]

        ### check if more_info contains a list of all current optimistic reward predictions, not only those for the chosen algorithm.
        ### TODO


        print("Curr reward ", reward)
        print("Opt reward pred ", more_info["optimistic_reward_predictions"])
        print("Pess reward pred ", more_info["pessimistic_reward_predictions"])
        print("All rewards ", self.all_rewards)
        print("Cumulative rewards ", self.cumulative_rewards)
        print("Num plays ", self.num_plays)
        print("Mean rewards ", self.mean_rewards)
        print("Balancing algorithm masks ", self.algorithm_mask)
        print("Balancing probabilities ",self.base_probas)

        self.T += 1


        ### TEST Conditions
        if self.vstar_upperbounds[algo_idx] < self.vstar_lowerbounds[algo_idx]:
            ### algo_idx is misspecified
            print("Eliminated algorithm index ", algo_idx)
            self.min_suriving_algo_index = max(self.min_suriving_algo_index, algo_idx+1)

        sandwich_intervals = list(zip(self.vstar_lowerbounds, self.vstar_upperbounds))

        for i in range(self.min_suriving_algo_index, self.m):
            for j in range(i+1, self.m):

                    ### Algorithm i may be misspecified
                    if sandwich_intervals[i][1] < sandwich_intervals[j][0]:
                        print("Eliminated algorithm index ", i)
                        self.min_suriving_algo_index = max(self.min_suriving_algo_index, i+1)
                
        
                    ### Algorithm j may be misspecified
                    if sandwich_intervals[j][1] < sandwich_intervals[i][0]:
                        print("Eliminated algorithm index ", j)
                        self.min_suriving_algo_index = max(self.min_suriving_algo_index, j+1)


        self.min_suriving_algo_index = min(self.min_suriving_algo_index, self.m-1)

        ## Fix the algorithm mask
        for i in range(self.min_suriving_algo_index):
                 self.algorithm_mask[i] = 0


        self.normalize_distribution()











def train_epsilon_greedy_modsel(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size, 
    representation_layer_sizes = [10, 10], threshold = .5, epsilons = [.1, .05, .01],
    verbose = False, decaying_epsilon = False, 
    restart_model_full_minimization = False, modselalgo = "Corral"):
    

    modsel_manager = get_modsel_manager(modselalgo, epsilons, num_batches)

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

    growing_training_dataset = GrowingNumpyDataSet(num_batches, batch_size, train_dataset.dimension)
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





def get_modsel_manager(modselalgo, parameters, num_batches):
    if modselalgo == "Corral":
        modsel_manager = CorralHyperparam(len(parameters), T = num_batches) ### hack
    elif modselalgo == "CorralAnytime":
        modsel_manager = CorralHyperparam(len(parameters), T = num_batches, eta = 1.0/np.sqrt(num_batches), anytime = True) 
    elif modselalgo == "EXP3":
        modsel_manager = EXP3Hyperparam(len(parameters), T = num_batches)
    elif modselalgo == "EXP3Anytime":
        modsel_manager = EXP3Hyperparam(len(parameters), T = num_batches, anytime = True)
    elif modselalgo == "UCB":
        modsel_manager = UCBHyperparam(len(parameters))
    elif modselalgo == "BalancingSharp":
        modsel_manager = BalancingHyperparamSharp(len(parameters), [max(x, .0000000001) for x in parameters])
    elif modselalgo == "BalancingDoubling":
        modsel_manager = BalancingHyperparamDoubling(len(parameters), min(parameters))
    elif modselalgo == "BalancingDoResurrect":
        modsel_manager = BalancingHyperparamDoubling(len(parameters), min(parameters), resurrecting = True)
    elif modselalgo == "BalancingSharp":
        modsel_manager = BalancingHyperparamSharp(len(parameters), [max(x, .0000000001) for x in parameters])
    elif modselalgo == "BalancingDoubling":
        modsel_manager = BalancingHyperparamDoubling(len(parameters), min(parameters))
    elif modselalgo == "BalancingDoResurrect":
        modsel_manager = BalancingHyperparamDoubling(len(parameters), min(parameters), resurrecting = True)
    elif modselalgo == "BalancingDoResurrectDown":
        modsel_manager = BalancingHyperparamDoubling(len(parameters), 10, resurrecting = True)
    elif modselalgo == "BalancingDoResurrectClassic":
        modsel_manager = BalancingHyperparamDoubling(len(parameters), min(parameters), 
            resurrecting = True, classic = True)
    else:
        raise ValueError("Modselalgo type {} not recognized.".format(modselalgo))

    return modsel_manager







def train_mahalanobis_modsel(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size,
    representation_layer_sizes = [10, 10], threshold = .5, alphas = [1, .1, .01], lambda_reg = 1,
    verbose = False,
    restart_model_full_minimization = False, modselalgo = "Corral", split = False, retraining_frequency = 1, 
    burn_in = -1):
    

    num_alphas = len(alphas)

    modsel_manager = get_modsel_manager(modselalgo, alphas, num_batches)

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

        growing_training_dataset = GrowingNumpyDataSet(num_batches, batch_size, train_dataset.dimension)

    if split:

        models = [ TorchMultilayerRegressionMahalanobis(
            alpha=alpha,
            representation_layer_sizes=representation_layer_sizes,
            dim = train_dataset.dimension,
            output_filter = 'logistic'
        ) for alpha in alphas]

        growing_training_datasets = [GrowingNumpyDataSet(num_batches, batch_size, train_dataset.dimension) for _ in range(num_alphas)]





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
        print("Modselalgo {}".format(modselalgo))
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
            # raise ValueError("asdlfkm")

            if i <= burn_in:
                #IPython.embed()
                optimistic_thresholded_predictions = optimistic_thresholded_predictions >= -10


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
        if i%retraining_frequency == 0:
            #IPython.embed()
            print("Training Model --- iteration {}".format(i))
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
    restart_model_full_minimization = False, modselalgo = "Corral", split = False, burn_in  = 0):
    

    num_regs = len(regs)
    modsel_manager = get_modsel_manager(modselalgo, regs, num_batches)

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

        growing_training_dataset = GrowingNumpyDataSet(num_batches, batch_size, train_dataset.dimension)

    if split:

        models = [ TorchMultilayerRegression(
            representation_layer_sizes=representation_layer_sizes,
            dim = train_dataset.dimension,
            output_filter = 'logistic'
        ) for reg in regs]

        growing_training_datasets = [GrowingNumpyDataSet(num_batches, batch_size, train_dataset.dimension) for _ in range(num_regs)]





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
        if i < burn_in:
            #IPython.embed()
            optimistic_thresholded_predictions = optimistic_thresholded_predictions >= 0

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
        
        print("batch y labels ", batch_y.squeeze())

        print("        ########################################")
        print("prediction differential ", optimistic_prob_predictions - pessimistic_prob_predictions)
        print("        ########################################")







        with torch.no_grad():        

            baseline_predictions = baseline_model.get_thresholded_predictions(batch_X, threshold)


            boolean_labels_y = batch_y >= threshold
            boolean_labels_y = boolean_labels_y.bool().squeeze()



            accuracy = (torch.sum(optimistic_thresholded_predictions*boolean_labels_y) +torch.sum( ~optimistic_thresholded_predictions*~boolean_labels_y))*1.0/batch_size


            #### TOTAL NUM POSITIVES
            total_num_positives = torch.sum(optimistic_thresholded_predictions)
            print("Num positives ",total_num_positives )
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




















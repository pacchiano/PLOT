import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

import IPython
from algorithmsmodsel import CorralHyperparam, EXP3Hyperparam, UCBHyperparam, BalancingHyperparamSharp, BalancingHyperparamDoubling
from algorithmsmodsel import UCBalgorithm

from simple_modsel_test import BernoulliBandit, GaussianBandit, test_MAB_modsel

np.random.seed(1000)
random.seed(1000)



def test_MAB_modsel_split(rewards_regrets_all, num_timesteps,  confidence_radius,
	modselalgo = "Corral"):
	
	num_base_learners = len(rewards_regrets_all)



	if modselalgo == "Corral":
		modsel_manager = CorralHyperparam(num_base_learners, T = num_timesteps) ### hack
	elif modselalgo == "CorralHigh":
		modsel_manager = CorralHyperparam(num_base_learners,  eta = 10, T = num_timesteps) ### hack
	elif modselalgo == "CorralLow":
		modsel_manager = CorralHyperparam(num_base_learners,  eta = .01, T = num_timesteps) ### hack
	elif modselalgo == "CorralAnytime":
		modsel_manager = CorralHyperparam(num_base_learners, T = num_timesteps, eta = 1.0/np.sqrt(num_timesteps), anytime = True) 
	elif modselalgo == "EXP3":
		modsel_manager = EXP3Hyperparam(num_base_learners, T = num_timesteps)
	elif modselalgo == "EXP3Anytime":
		modsel_manager = EXP3Hyperparam(num_base_learners, T = num_timesteps, anytime = True)
	elif modselalgo == "UCB":
		modsel_manager = UCBHyperparam(num_base_learners)
	
	elif modselalgo == "EXP3Low":
			modsel_manager = EXP3Hyperparam(num_base_learners, T = num_timesteps, eta_multiplier = .1)

	elif modselalgo == "EXP3High":
			modsel_manager = EXP3Hyperparam(num_base_learners, T = num_timesteps, eta_multiplier = 10)

	elif modselalgo == "BalancingSharp":
		modsel_manager = BalancingHyperparamSharp(num_base_learners, [max(x, .0000000001) for x in confidence_radii])
	elif modselalgo == "BalancingDoubling":
		modsel_manager = BalancingHyperparamDoubling(num_base_learners, confidence_radius)
	elif modselalgo == "BalancingDoResurrect":
		modsel_manager = BalancingHyperparamDoubling(num_base_learners, confidence_radius, resurrecting = True)
	elif modselalgo == "BalancingDoResurrectDown":
		modsel_manager = BalancingHyperparamDoubling(num_base_learners, 10, resurrecting = True)
	elif modselalgo == "BalancingDoResurrectClassic":
		modsel_manager = BalancingHyperparamDoubling(num_base_learners, confidence_radius, 
			resurrecting = True, classic = True)

	else:
		raise ValueError("Modselalgo type {} not recognized.".format(modselalgo))


	if algotype == "bernoulli":

		bandit = BernoulliBandit(means, scalings)
	elif algotype == "gaussian":
		bandit = GaussianBandit(means, stds)
	else:
		raise ValueError("unrecognized bandit type {}".format(algotype))

	num_arms = len(means)
	#empirical_means = [0 for _ in range(num_arms)]




	# play_arm_index = random.choice(range(num_arms))

	#ucb_algorithms = [UCBalgorithm(num_arms) for _ in range(num_base_learners)]

	

	rewards = []
	mean_rewards = []
	instantaneous_regrets = []
	probabilities = []


	#rewards_regrets_all

	base_learners_index = [0 for _ in range(num_base_learners)]



	#per_algorithm_regrets = [[] for _ in range(num_base_learners)]

	#arm_pulls = [0 for _ in range(num_arms)]
	#confidence_radius_pulls = [0 for _ in range(num_base_learners)]

	for t in range(num_timesteps):
		print("Timestep {}".format(t))
		modsel_sample_idx = modsel_manager.sample_base_index()
		probabilities.append(modsel_manager.get_distribution())

		
		#confidence_radius = confidence_radii[modsel_sample_idx]
		print("Selected confidence radius {}".format(confidence_radius))
		

		# play_arm_index, ucb_arm_value, lcb_arm_value = ucb_algorithm.get_ucb_arm(confidence_radius)
		# arm_pulls[play_arm_index] += 1
		

		reward = rewards_regrets_all[modsel_sample_idx][0][base_learners_index[modsel_sample_idx]]
		rewards.append(reward)

		#modsel_info = dict([])
		#modsel_info["optimistic_reward_predictions"] = ucb_arm_value
		#modsel_info["pessimistic_reward_predictions"] = lcb_arm_value

		#ucb_algorithm.update_arm_statistics(play_arm_index, reward)
		
		#mean_reward = bandit.get_arm_mean(play_arm_index)

		#mean_rewards.append(mean_reward)

		##### NEED TO ADD THE UPDATE TO THE MODEL SEL ALGO


		modsel_info = rewards_regrets_all[modsel_sample_idx][2][base_learners_index[modsel_sample_idx]]


		modsel_manager.update_distribution(modsel_sample_idx, reward, modsel_info )



		instantaneous_regret = rewards_regrets_all[modsel_sample_idx][1][base_learners_index[modsel_sample_idx]]
		instantaneous_regrets.append(instantaneous_regret)

		base_learners_index[modsel_sample_idx] += 1

		#per_algorithm_regrets[modsel_sample_idx].append(instantaneous_regret)

	return rewards, instantaneous_regrets, base_learners_index, probabilities






if __name__ == "__main__":

	num_timesteps = int(sys.argv[1])
	exp_type = str(sys.argv[2])
	num_experiments = int(sys.argv[3]) ### In this experiment this variable corresponds to the number of confidence radii


	if exp_type == "exp1":
		means = [.7, .8]
		stds = []
		scalings = []
		confidence_radius = .16 ## increase radii
		algotype = "bernoulli"
		experiment_name = "exp6"


	else:
		raise ValueError("experiment type not recognized")

	confidence_radii = [confidence_radius]*num_experiments



	exp_data_dir = "./splitModsel/{}".format(experiment_name)
	exp_info = "means - {} \n conf_radii - {}".format(means, confidence_radii)

	if not os.path.exists(exp_data_dir):
		os.mkdir(exp_data_dir)

	exp_data_dir_T = "{}/T{}".format(exp_data_dir, num_timesteps)
	if not os.path.exists(exp_data_dir_T):
		os.mkdir(exp_data_dir_T)

	per_experiment_data = "{}/detailed".format(exp_data_dir_T)
	if not os.path.exists(per_experiment_data):
		os.mkdir(per_experiment_data)

	with open('{}/info.txt'.format(exp_data_dir), 'w') as f:
	    f.write(exp_info)


	colors = ["red", "orange", "violet", "black", "brown", "yellow", "green", "gray"]	
	modselalgos = ["UCB", 'BalancingSharp',  "EXP3", "Corral", 'BalancingDoResurrectClassic','BalancingDoResurrectDown', "BalancingDoResurrect"]#"BalancingDoubling"]# "BalancingDoubling"]#"BalancingDoResurrect", "BalancingSharp", "UCB", "EXP3", "Corral" ]

	normalization_visualization = 1.0/np.sqrt( np.arange(num_timesteps) + 1)
	normalization_visualization *= 1.0/np.log( np.arange(num_timesteps) + 2)



	#### RUN THE BASELINES
	
	cum_regrets_all = []	
	rewards_regrets_all = []		

	baselines_results = []

	for confidence_radius in confidence_radii:
			#confidence_radius_pulls_all = []
			

			rewards, mean_rewards, instantaneous_regrets, arm_pulls,_, _, _, modsel_infos = test_MAB_modsel(means, stds, scalings, num_timesteps, 
					[confidence_radius],  modselalgo = "Corral", algotype = algotype) ### Here we can use any modselalgo, it is dummy in this case.

			cum_regrets_all.append(np.cumsum(instantaneous_regrets))
			rewards_regrets_all.append((rewards, instantaneous_regrets, modsel_infos))

			# mean_cum_regrets = np.mean(cum_regrets_all,0)
			# std_cum_regrets = np.std(cum_regrets_all,0)

			# mean_cum_regrets *= normalization_visualization
			# std_cum_regrets *= normalization_visualization

			baselines_results.append(np.cumsum(instantaneous_regrets))

			#plt.plot(np.arange(num_timesteps) + 1, mean_cum_regrets, label = "radius {}".format(confidence_radius), color = color )
			#plt.fill_between(np.arange(num_timesteps) + 1,mean_cum_regrets - .5*std_cum_regrets,mean_cum_regrets + .5*std_cum_regrets, alpha = .2 , color = color )



	# for split in [False, True]:
		
	# 	split_tag = ""
	# 	if split:
	# 		split_tag = " - split"


	for modselalgo in modselalgos:
		
			modsel_cum_regrets_all = []	
			modsel_confidence_radius_pulls_all = []
			probabilities_all = []
			#per_algorithm_regrets_stats = []
			for _ in range(num_experiments):
				modsel_rewards, modsel_instantaneous_regrets, modsel_base_learners_index, probabilities_modsel = test_MAB_modsel_split(rewards_regrets_all, num_timesteps,  confidence_radius,
					modselalgo = modselalgo)
				
				modsel_cum_regrets_all.append(np.cumsum(modsel_instantaneous_regrets))
				modsel_confidence_radius_pulls_all.append(modsel_base_learners_index)
				probabilities_all.append(probabilities_modsel)
				#per_algorithm_regrets_stats.append(per_algorithm_regrets)





			# IPython.embed()
			# raise ValueError("Asflkm")

			mean_probabilities = np.mean(probabilities_all, 0)
			std_probabilities = np.std(probabilities_all, 0)
			
			#IPython.embed()

			for i,confidence_radius,color in zip(range(len(confidence_radii)), confidence_radii, colors):

				plt.plot(np.arange(num_timesteps) + 1, mean_probabilities[:,i], 
					label = str(confidence_radius), color = color )
				plt.fill_between(np.arange(num_timesteps) + 1, mean_probabilities[:,i] - .5*std_probabilities[:,i],
						mean_probabilities[:,i] + .5*std_probabilities[:,i], alpha = .1, color = color)


			plt.title("Probabilities Evolution {} split".format(modselalgo))

			plt.legend(fontsize=8, loc="upper left")

			plt.savefig("{}/prob_evolution_split_{}_T{}.png".format(exp_data_dir_T, modselalgo,num_timesteps))
			
			plt.close("all")


			mean_modsel_cum_regrets = np.mean(modsel_cum_regrets_all,0)
			std_modsel_cum_regrets = np.std(modsel_cum_regrets_all,0)
			mean_modsel_cum_regrets *= normalization_visualization
			std_modsel_cum_regrets *= normalization_visualization




			#IPython.embed()
			plt.plot(np.arange(num_timesteps) + 1, mean_modsel_cum_regrets, label = modselalgo, color = "blue" )

			plt.fill_between(np.arange(num_timesteps) + 1,mean_modsel_cum_regrets -.5*std_modsel_cum_regrets, mean_modsel_cum_regrets +.5*std_modsel_cum_regrets, color = "blue", alpha = .2   )

			mean_modsel_confidence_radius_pulls = np.mean( modsel_confidence_radius_pulls_all, 0)


			### Increase the learning rate CORRAL
			### Get prob plot.
			### Check SPLIT
			### Get histograms of base algorithm choices 

			### UCB - when does it fail? Maybe back to the contextual case. 
			### [first] Fix Balancing to have comparisons between base learners.

			### EXP3 check probabilities.
			### Add artificially wrong learners.?



			### plot all the model selection together .... Add an extra plot. 
			### Implement SuperClassic^2 --- original version (with elimination)


			### Try our original balancing with the widths of the base learners. 
			### halving algorithm

			### Fair tuning of the baselines. 
			### different learning rates for CORRAL / EXP3. (sensitivity analysis). Pick the best LR. 

			### Multiple levels of noise. --- We want to see the algorithm is flexible enough to operate well 
			### accross different scenarios. 
			### when the variance is small. Bernstein stuff works. 

			### Model selection algos will do well.

			### Nested Corrals to pick learning rates.

			## Doubling with a smaller putative bound. (say log t)
			## CRITEO dataset



			## Smoothing window for balancing Classic. 
			## Network architecture selection. Try many networks?
			## Run UCB / Corral / Etc on the same experiments. 
			## Is doubling too aggressive? This is a hyperparameter. 

			



			for confidence_radius, baseline_result, color in zip(confidence_radii, baselines_results, colors):
				#mean_cum_regrets, std_cum_regrets = baseline_result_tuple
				plt.plot(np.arange(num_timesteps) + 1, baseline_result, label = "radius {}".format(confidence_radius), color = color )
				#plt.fill_between(np.arange(num_timesteps) + 1,mean_cum_regrets - .5*std_cum_regrets,mean_cum_regrets + .5*std_cum_regrets, alpha = .2 , color = color )



			#plt.show()
			plt.title("Cumulative Regret - Split")
			plt.legend(fontsize=8, loc="upper left")
			#plt.ylim(0,50)
			#plt.show()
			
			plt.savefig("{}/regret_modsel_test_split_{}_T{}.png".format(exp_data_dir_T, modselalgo,num_timesteps))
			

			plt.close("all")

			#IPython.embed()

			

				


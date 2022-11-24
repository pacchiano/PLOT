import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

import IPython
from algorithmsmodsel import CorralHyperparam, EpochBalancingHyperparam, EXP3Hyperparam, UCBHyperparam, BalancingHyperparamSharp, BalancingHyperparamDoubling
from algorithmsmodsel import UCBalgorithm

np.random.seed(1000)
random.seed(1000)

class BernoulliBandit:
	def __init__(self, base_means, scalings = []):
		self.base_means = base_means
		
		self.num_arms = len(base_means)
		
		if len(scalings) == 0:
			self.scalings = [1 for _ in range(self.num_arms)]
		else:
			self.scalings = scalings

		self.means = [self.base_means[i]*self.scalings[i] for i in range(self.num_arms)]
		self.max_mean = max(self.means)

	def get_reward(self, arm_index):
		if arm_index >= self.num_arms or arm_index < 0:
			raise ValueError("Invalid arm index {}".format(arm_index))

		random_uniform_sample = random.random()
		if random_uniform_sample <= self.base_means[arm_index]:
			return 1*self.scalings[arm_index]
		else:
			return 0

	def get_max_mean(self):
		return self.max_mean

	def get_arm_mean(self, arm_index):
		return self.means[arm_index]



class GaussianBandit:
	def __init__(self, means, stds):
		self.means = means
		self.stds = stds
		self.num_arms = len(means)
		self.max_mean = max(self.means)


	def get_reward(self, arm_index):
		if arm_index >= self.num_arms or arm_index < 0:
			raise ValueError("Invalid arm index {}".format(arm_index))


		return np.random.normal(self.means[arm_index], self.stds[arm_index])


	def get_max_mean(self):
		return self.max_mean

	def get_arm_mean(self, arm_index):
		return self.means[arm_index]





def test_MAB_modsel(means, stds, scalings, num_timesteps, confidence_radii,  
	modselalgo = "Corral", split = False, algotype = "bernoulli"):
	



	if modselalgo == "Corral":
		modsel_manager = CorralHyperparam(len(confidence_radii), T = num_timesteps) ### hack
	elif modselalgo == "CorralAnytime":
		modsel_manager = CorralHyperparam(len(confidence_radii), T = num_timesteps, eta = 1.0/np.sqrt(num_timesteps), anytime = True) 
	elif modselalgo == "EXP3":
		modsel_manager = EXP3Hyperparam(len(confidence_radii), T = num_timesteps)
	elif modselalgo == "EXP3Anytime":
		modsel_manager = EXP3Hyperparam(len(confidence_radii), T = num_timesteps, anytime = True)
	elif modselalgo == "UCB":
		modsel_manager = UCBHyperparam(len(confidence_radii))
	
	elif modselalgo == "BalancingSharp":
		modsel_manager = BalancingHyperparamSharp(len(confidence_radii), [max(x, .0000000001) for x in confidence_radii])
	elif modselalgo == "BalancingDoubling":
		modsel_manager = BalancingHyperparamDoubling(len(confidence_radii), min(confidence_radii))
	elif modselalgo == "BalancingDoResurrect":
		modsel_manager = BalancingHyperparamDoubling(len(confidence_radii), min(confidence_radii), resurrecting = True)

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

	if split:
		ucb_algorithms = [UCBalgorithm(num_arms) for _ in range(len(confidence_radii))]

	else:
		ucb_algorithm = UCBalgorithm(num_arms)


	rewards = []
	mean_rewards = []
	instantaneous_regrets = []
	probabilities = []


	per_algorithm_regrets = [[] for _ in range(len(confidence_radii))]

	arm_pulls = [0 for _ in range(num_arms)]
	confidence_radius_pulls = [0 for _ in range(len(confidence_radii))]

	for t in range(num_timesteps):
		print("Timestep {}".format(t))
		modsel_sample_idx = modsel_manager.sample_base_index()
		probabilities.append(modsel_manager.get_distribution())
		confidence_radius_pulls[modsel_sample_idx] += 1
		confidence_radius = confidence_radii[modsel_sample_idx]
		print("Selected confidence radius {}".format(confidence_radius))
		if split:
			ucb_algorithm = ucb_algorithms[modsel_sample_idx]


		play_arm_index, ucb_arm_value, lcb_arm_value = ucb_algorithm.get_ucb_arm(confidence_radius)
		arm_pulls[play_arm_index] += 1
		reward = bandit.get_reward(play_arm_index)
		rewards.append(reward)

		modsel_info = dict([])
		modsel_info["optimistic_reward_predictions"] = ucb_arm_value
		modsel_info["pessimistic_reward_predictions"] = lcb_arm_value

		ucb_algorithm.update_arm_statistics(play_arm_index, reward)
		
		mean_reward = bandit.get_arm_mean(play_arm_index)

		mean_rewards.append(mean_reward)

		##### NEED TO ADD THE UPDATE TO THE MODEL SEL ALGO

		modsel_manager.update_distribution(modsel_sample_idx, reward, modsel_info )



		instantaneous_regret = bandit.get_max_mean() - bandit.get_arm_mean(play_arm_index)
		instantaneous_regrets.append(instantaneous_regret)

		per_algorithm_regrets[modsel_sample_idx].append(instantaneous_regret)

	return rewards, mean_rewards, instantaneous_regrets, arm_pulls, confidence_radius_pulls, probabilities, per_algorithm_regrets






if __name__ == "__main__":

	num_timesteps = int(sys.argv[1])
	exp_type = str(sys.argv[2])

	if exp_type == "exp1":
		means = [.1, .2, .5, .55]
		stds = []
		scalings = []
		confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		algotype = "bernoulli"
		experiment_name = "exp1"

	elif exp_type == "exp2":
		means = [.7, .8]
		stds = [.01, 5]
		scalings = []
		confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		algotype = "gaussian"
		experiment_name = "exp2"

	elif exp_type == "exp3":
		means = [.7, .01]
		stds = []
		scalings = [1,80]
		confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		algotype = "bernoulli"
		experiment_name = "exp3"

	elif exp_type == "exp4":
		means = [.7, .01, .8, .9, .91, .7, .01, .8,.7, .01, .8,]
		stds = []
		scalings = [1,1,1,1,1,1,1,1,1,1,1]
		confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		algotype = "bernoulli"
		experiment_name = "exp4"

	elif exp_type == "exp5":
		means = [.7, .01, .8, .9, .91, .7, .01, .8,.7, .01, .8]
		stds = [.1, .8, .1, .1, .1, .1, .9, .1,.1, .8, .1]
		scalings = [1,1,1,1,1,1,1,1,1,1,1]
		confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		algotype = "gaussian"
		experiment_name = "exp5"

	else:
		raise ValueError("experiment type not recognized")



	exp_data_dir = "./debugModsel/{}".format(experiment_name)
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


	#raise ValueError("alsdkfmaslkdfmaslkdfmalsdkmf")



	colors = ["red", "orange", "violet", "black", "brown", "yellow", "green", "gray"]
	#num_timesteps = 5000

	num_experiments = 200

	#split = True

	
	modselalgos = [ "BalancingDoubling"]#"BalancingDoResurrect", "BalancingSharp", "UCB", "EXP3", "Corral" ]

	normalization_visualization = 1.0/np.sqrt( np.arange(num_timesteps) + 1)
	normalization_visualization *= 1.0/np.log( np.arange(num_timesteps) + 2)



	#### RUN THE BASELINES

	baselines_results = []

	for confidence_radius in confidence_radii:
			cum_regrets_all = []	
			#confidence_radius_pulls_all = []
			for _ in range(num_experiments):

				rewards, mean_rewards, instantaneous_regrets, arm_pulls,_, _, _ = test_MAB_modsel(means, stds, scalings, num_timesteps, 
					[confidence_radius],  modselalgo = "Corral", algotype = algotype) ### Here we can use any modselalgo, it is dummy in this case.

				cum_regrets_all.append(np.cumsum(instantaneous_regrets))

			mean_cum_regrets = np.mean(cum_regrets_all,0)
			std_cum_regrets = np.std(cum_regrets_all,0)

			mean_cum_regrets *= normalization_visualization
			std_cum_regrets *= normalization_visualization

			baselines_results.append((mean_cum_regrets, std_cum_regrets))

			#plt.plot(np.arange(num_timesteps) + 1, mean_cum_regrets, label = "radius {}".format(confidence_radius), color = color )
			#plt.fill_between(np.arange(num_timesteps) + 1,mean_cum_regrets - .5*std_cum_regrets,mean_cum_regrets + .5*std_cum_regrets, alpha = .2 , color = color )



	for split in [False, True]:
		
		split_tag = ""
		if split:
			split_tag = " - split"


		for modselalgo in modselalgos:
		
			modsel_cum_regrets_all = []	
			modsel_confidence_radius_pulls_all = []
			probabilities_all = []
			per_algorithm_regrets_stats = []
			for _ in range(num_experiments):
				modsel_rewards, modsel_mean_rewards, modsel_instantaneous_regrets, modsel_arm_pulls, modsel_confidence_radius_pulls, probabilities_modsel, per_algorithm_regrets = test_MAB_modsel(means, stds, scalings, 
					num_timesteps, 
					confidence_radii,  modselalgo = modselalgo, 
					split = split, algotype = algotype)
				
				modsel_cum_regrets_all.append(np.cumsum(modsel_instantaneous_regrets))
				modsel_confidence_radius_pulls_all.append(modsel_confidence_radius_pulls)
				probabilities_all.append(probabilities_modsel)
				per_algorithm_regrets_stats.append(per_algorithm_regrets)





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
				#plt.fill_between(np.arange(num_timesteps) + 1, mean_probabilities[:,i],
				#		mean_probabilities[:,i] , alpha = .1, color = color)


			plt.title("Probabilities Evolution {} {}".format(modselalgo, split_tag))

			plt.legend(fontsize=8, loc="upper left")

			if split:
				plt.savefig("{}/prob_evolution_split_{}_T{}.png".format(exp_data_dir_T, modselalgo,num_timesteps))
			else:
				plt.savefig("{}/prob_evolution_{}_T{}.png".format(exp_data_dir_T, modselalgo,num_timesteps))

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

			### Do one run. Plot the actual regrets of the base learners played by 
			### the model selection algorithm.

			### plot the computed regret upper bounds for the bases.
			### change np.argmax in the UCB algorithm


			### Double check that random seed setting is actually correct. 




			### Try our original balancing with the widths of the base learners. 



			### Same experiment with multiple levels of noise. 
			### Model selection algos will do well.




			for confidence_radius, baseline_result_tuple, color in zip(confidence_radii, baselines_results, colors):
				mean_cum_regrets, std_cum_regrets = baseline_result_tuple
				plt.plot(np.arange(num_timesteps) + 1, mean_cum_regrets, label = "radius {}".format(confidence_radius), color = color )
				plt.fill_between(np.arange(num_timesteps) + 1,mean_cum_regrets - .5*std_cum_regrets,mean_cum_regrets + .5*std_cum_regrets, alpha = .2 , color = color )



			#plt.show()
			plt.title("Cumulative Regret - {}".format(split_tag))
			plt.legend(fontsize=8, loc="upper left")
			#plt.ylim(0,50)
			#plt.show()
			if split:
				plt.savefig("{}/regret_modsel_test_split_{}_T{}.png".format(exp_data_dir_T, modselalgo,num_timesteps))
			else:
				plt.savefig("{}/regret_modsel_test_{}_T{}.png".format(exp_data_dir_T, modselalgo,num_timesteps))

			plt.close("all")

			
			### PLOT last modsel experiment info
			for i in range(num_experiments):
				plt.title("Single Experiment Regrets - {}".format(split_tag))
				plt.legend(fontsize=8, loc="upper left")


				plt.plot(np.arange(num_timesteps) + 1, modsel_cum_regrets_all[i], label = modselalgo, color = "blue" )
				#plt.fill_between(np.arange(num_timesteps) + 1,mean_modsel_cum_regrets -.5*std_modsel_cum_regrets, mean_modsel_cum_regrets +.5*std_modsel_cum_regrets, color = "blue", alpha = .2   )

				for confidence_radius, info_list, color in zip(confidence_radii, per_algorithm_regrets_stats[i], colors):
						cum_regret = np.cumsum(info_list)
						plt.plot(np.arange(len(cum_regret)) + 1, cum_regret, label = "radius {}".format(confidence_radius), color = color )
						#plt.fill_between(np.arange(num_timesteps) + 1,mean_cum_regrets - .5*std_cum_regrets,mean_cum_regrets + .5*std_cum_regrets, alpha = .2 , color = color )

				plt.legend(fontsize=8, loc="upper left")

				if split:
					plt.savefig("{}/singlerun_modsel_test_split_exp{}_{}_T{}.png".format( per_experiment_data, i+1, modselalgo,num_timesteps))
				else:
					plt.savefig("{}/singlerun_modsel_test_exp{}_{}_T{}.png".format(per_experiment_data, i+1, modselalgo,num_timesteps))

				plt.close("all")

			


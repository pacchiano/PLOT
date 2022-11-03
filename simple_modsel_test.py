import random
import numpy as np
import sys
import matplotlib.pyplot as plt

import IPython
from algorithmsmodsel import CorralHyperparam, EpochBalancingHyperparam, EXP3Hyperparam, UCBHyperparam
from algorithmsmodsel import UCBalgorithm

np.random.seed(1000)
random.seed(1000)

class BernoulliBandit:
	def __init__(self, means):
		self.means = means
		self.num_arms = len(means)
		self.max_mean = max(self.means)


	def get_reward(self, arm_index):
		if arm_index >= self.num_arms or arm_index < 0:
			raise ValueError("Invalid arm index {}".format(arm_index))

		random_uniform_sample = random.random()
		if random_uniform_sample <= self.means[arm_index]:
			return 1
		else:
			return 0

	def get_max_mean(self):
		return self.max_mean

	def get_arm_mean(self, arm_index):
		return self.means[arm_index]








def test_bernoulli_MAB_modsel(means, num_timesteps, confidence_radii,  
	modselalgo = "Corral", split = False):
	
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

	# elif modselalgo == "BalancingSimple":
	# 	modsel_manager = BalancingHyperparam(len(confidence_radii), 
	# 	   confidence_radii, delta =0.01, balancing_type = "BalancingSimple" )
	# elif modselalgo == "BalancingAnalytic":
	# 	modsel_manager = BalancingHyperparam(len(confidence_radii), 
	# 		confidence_radii, delta =0.01, balancing_type = "BalancingAnalytic")
	# elif modselalgo == "BalancingAnalyticHybrid":
	# 	modsel_manager = BalancingHyperparam(len(confidence_radii), 
	# 		confidence_radii, delta =0.01, balancing_type = "BalancingAnalyticHybrid")
	elif modselalgo == "EpochBalancing":
		modsel_manager = EpochBalancingHyperparam(len(confidence_radii), [max(x, .0000000001) for x in confidence_radii])
	else:
		raise ValueError("Modselalgo type {} not recognized.".format(modselalgo))

	bandit = BernoulliBandit(means)
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

	return rewards, mean_rewards, instantaneous_regrets, arm_pulls, confidence_radius_pulls, probabilities






if __name__ == "__main__":

	split_settings = sys.argv[1]
	if split_settings not in ["True", "False"]:
		raise ValueError("Split settings not recognized.")
	split = split_settings == "True"

	num_timesteps = int(sys.argv[2])

	means = [.1, .2, .5, .55]
	confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25] ## increase radii
	colors = ["red", "orange", "violet", "black", "brown", "yellow", "green", "gray"]
	#num_timesteps = 5000

	num_experiments = 10

	#split = True
	split_tag = ""
	if split:
		split_tag = " - split"


	modselalgos =[ "UCB", "EpochBalancing", "EXP3", "EXP3Anytime", "Corral", "CorralAnytime"]#"EpochBalancing" ,

	normalization_visualization = 1.0/np.sqrt( np.arange(num_timesteps) + 1)
	normalization_visualization *= 1.0/np.log( np.arange(num_timesteps) + 2)

	for modselalgo in modselalgos:
	
		modsel_cum_regrets_all = []	
		modsel_confidence_radius_pulls_all = []
		probabilities_all = []
		for _ in range(num_experiments):
			modsel_rewards, modsel_mean_rewards, modsel_instantaneous_regrets, modsel_arm_pulls, modsel_confidence_radius_pulls, probabilities_modsel = test_bernoulli_MAB_modsel(means, num_timesteps, 
				confidence_radii,  modselalgo = modselalgo, split = split)
			modsel_cum_regrets_all.append(np.cumsum(modsel_instantaneous_regrets))
			modsel_confidence_radius_pulls_all.append(modsel_confidence_radius_pulls)
			probabilities_all.append(probabilities_modsel)


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
			plt.savefig("./debugModsel/simple_prob_evolution_split_{}_T{}.png".format(modselalgo,num_timesteps))
		else:
			plt.savefig("./debugModsel/simple_prob_evolution_{}_T{}.png".format(modselalgo,num_timesteps))

		plt.close("all")


		mean_modsel_cum_regrets = np.mean(modsel_cum_regrets_all,0)
		std_modsel_cum_regrets = np.std(modsel_cum_regrets_all,0)
		mean_modsel_cum_regrets *= normalization_visualization
		std_modsel_cum_regrets *= normalization_visualization




		#IPython.embed()
		plt.plot(np.arange(num_timesteps) + 1, mean_modsel_cum_regrets, label = modselalgo, color = "blue" )

		plt.fill_between(np.arange(num_timesteps) + 1,mean_modsel_cum_regrets -.5*std_modsel_cum_regrets, mean_modsel_cum_regrets +.5*std_modsel_cum_regrets, color = "blue", alpha = .2   )

		mean_modsel_confidence_radius_pulls = np.mean( modsel_confidence_radius_pulls_all, 0)


		### Divide regrets by SQRT(t) for visualization.
		### Increase the learning rate CORRAL
		### Get prob plot.
		### Check SPLIT


		### Run 50K
		### 

		for confidence_radius, color in zip(confidence_radii, colors):
			cum_regrets_all = []	
			#confidence_radius_pulls_all = []
			for _ in range(num_experiments):

				rewards, mean_rewards, instantaneous_regrets, arm_pulls,_, _ = test_bernoulli_MAB_modsel(means, num_timesteps, 
					[confidence_radius],  modselalgo = modselalgo)

				cum_regrets_all.append(np.cumsum(instantaneous_regrets))

			mean_cum_regrets = np.mean(cum_regrets_all,0)
			std_cum_regrets = np.std(cum_regrets_all,0)

			mean_cum_regrets *= normalization_visualization
			std_cum_regrets *= normalization_visualization


			plt.plot(np.arange(num_timesteps) + 1, mean_cum_regrets, label = "radius {}".format(confidence_radius), color = color )

			plt.fill_between(np.arange(num_timesteps) + 1,mean_cum_regrets - .5*std_cum_regrets,mean_cum_regrets + .5*std_cum_regrets, alpha = .2 , color = color )



		#IPython.embed()

		# plt.title("Mean Rewards Evolution")
		# plt.plot(np.arange(num_timesteps) + 1, mean_rewards, label = "modsel" )
		# plt.legend(fontsize=8, loc="upper left")
		#plt.show()
		plt.title("Cumulative Regret - {}".format(split_tag))
		plt.legend(fontsize=8, loc="upper left")
		#plt.ylim(0,50)
		#plt.show()
		if split:
			plt.savefig("./debugModsel/simple_modsel_test_split_{}_T{}.png".format(modselalgo,num_timesteps))
		else:
			plt.savefig("./debugModsel/simple_modsel_test_{}_T{}.png".format(modselalgo,num_timesteps))

		plt.close("all")
		
	#IPython.embed()



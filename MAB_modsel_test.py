from model_experiments import get_modsel_manager, AlgorithmEnvironment, test_modsel_algorithm
from algorithmsmodsel import UCBalgorithm


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



class UCBandEnv(AlgorithmEnvironment):
	def __init__(self, algotype, confidence_radius = 1, means = [], scalings = [], stds = []):
		super().__init__()

		if algotype == "bernoulli":
			self.bandit = BernoulliBandit(means, scalings)
		elif algotype == "gaussian":
			self.bandit = GaussianBandit(means, stds)
		else:
			raise ValueError("unrecognized bandit type {}".format(algotype))

		self.num_arms = len(means)
		self.ucb_algorithm = UCBalgorithm(self.num_arms)
		self.confidence_radius = confidence_radius
	
	def step(self):

		play_arm_index, ucb_arm_value, lcb_arm_value = self.ucb_algorithm.get_ucb_arm(self.confidence_radius)
		#arm_pulls[play_arm_index] += 1
		instantenous_reward = self.bandit.get_reward(play_arm_index)
		#rewards.append(reward)

		extra_info = dict([])
		extra_info["optimistic_reward_predictions"] = ucb_arm_value
		extra_info["pessimistic_reward_predictions"] = lcb_arm_value


		self.ucb_algorithm.update_arm_statistics(play_arm_index, instantenous_reward)
		
		instantenous_pseudo_reward = self.bandit.get_arm_mean(play_arm_index)

		instantaneous_regret = self.bandit.get_max_mean() - self.bandit.get_arm_mean(play_arm_index)


		### Update Data Containers

		self.update_data_containers(instantenous_reward,instantenous_pseudo_reward, instantenous_regret )
		

		return instantenous_reward, instantenous_pseudo_reward, instantenous_regret, extra_info


	def reset_algorithm_and_environment(self):
		self.ucb_algorithm = UCBalgorithm(self.num_arms)



	def set_algorithm_parameters(self, params):
		confidence_radius = params["confidence_radius"]
		self.confidence_radius = confidence_radius






if __name__ == "__main__":

	num_timesteps = int(sys.argv[1])
	exp_type = str(sys.argv[2])
	num_experiments = int(sys.argv[3])

	if exp_type == "exp1":
		means = [.1, .2, .5, .55]
		stds = []
		scalings = []
		confidence_radii = [.08, .16, .64, 1.24, 2.5, 5, 10, 25	] ## increase radii
		algotype = "bernoulli"
		experiment_name = "exp1"

	elif exp_type == "exp2":
		means = [.7, .8]
		stds = []
		scalings = []
		confidence_radii = [.08, .16, .64, 1.24	] ## increase radii
		algotype = "bernoulli"
		experiment_name = "exp2"






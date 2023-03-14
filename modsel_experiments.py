def get_modsel_manager(modselalgo,confidence_radii, num_timesteps ):
	if modselalgo == "Corral":
		modsel_manager = CorralHyperparam(len(confidence_radii), T = num_timesteps) ### hack
	elif modselalgo == "CorralHigh":
		modsel_manager = CorralHyperparam(len(confidence_radii),  eta = 10, T = num_timesteps) ### hack
	elif modselalgo == "CorralLow":
		modsel_manager = CorralHyperparam(len(confidence_radii),  eta = .01, T = num_timesteps) ### hack
	elif modselalgo == "CorralAnytime":
		modsel_manager = CorralHyperparam(len(confidence_radii), T = num_timesteps, eta = 1.0/np.sqrt(num_timesteps), anytime = True) 
	elif modselalgo == "EXP3":
		modsel_manager = EXP3Hyperparam(len(confidence_radii), T = num_timesteps)
	elif modselalgo == "EXP3Anytime":
		modsel_manager = EXP3Hyperparam(len(confidence_radii), T = num_timesteps, anytime = True)
	elif modselalgo == "UCB":
		modsel_manager = UCBHyperparam(len(confidence_radii))
	
	elif modselalgo == "EXP3Low":
			modsel_manager = EXP3Hyperparam(len(confidence_radii), T = num_timesteps, eta_multiplier = .1)

	elif modselalgo == "EXP3High":
			modsel_manager = EXP3Hyperparam(len(confidence_radii), T = num_timesteps, eta_multiplier = 10)

	elif modselalgo == "BalancingSharp":
		modsel_manager = BalancingHyperparamSharp(len(confidence_radii), [max(x, .0000000001) for x in confidence_radii])
	elif modselalgo == "BalancingDoubling":
		modsel_manager = BalancingHyperparamDoubling(len(confidence_radii), min(confidence_radii))
	elif modselalgo == "BalancingDoResurrect":
		modsel_manager = BalancingHyperparamDoubling(len(confidence_radii), min(confidence_radii), resurrecting = True)
	elif modselalgo == "BalancingDoResurrectDown":
		modsel_manager = BalancingHyperparamDoubling(len(confidence_radii), 10, resurrecting = True)
	elif modselalgo == "BalancingDoResurrectClassic":
		modsel_manager = BalancingHyperparamDoubling(len(confidence_radii), min(confidence_radii), 
			resurrecting = True, classic = True)
	elif modselalgo == "Greedy":
		modsel_manager = UCBHyperparam(len(confidence_radii), confidence_radius = 0)
	elif modselalgo == "EpsilonGreedy":
		modsel_manager = UCBHyperparam(len(confidence_radii), confidence_radius = 0, epsilon = 0.05)
	else:
		raise ValueError("Modselalgo type {} not recognized.".format(modselalgo))


	return modsel_manager




### TODO: description of this class.
class AlgorithmEnvironment:

	def __init__(self, dataset, batch_size):
		self.reset_data_containers()
		self.reset_algorithm_and_environment()
		(
        train_dataset,
        test_dataset,
    	) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000)

    	self.train_dataset = train_dataset




	def reset_data_containers(self):
		self.rewards = []
		self.instantaneous_regrets = []
		self.pseudorewards = []

	def update_data_containers(self, instantenous_reward,instantenous_pseudo_reward, instantenous_regret ):
		self.rewards.append(instantenous_reward)
		self.pseudorewards.append(instantenous_reward)
		self.instantaneous_regrets.append(instantenous_regret)

	def step(self):
		instantenous_reward = 0
		instantenous_pseudo_reward = 0
		instantenous_regret = 0
		extra_info = dict([])

		
		raise ValueError("step function not implemented")
		return instantenous_reward, instantenous_pseudo_reward, instantenous_regret, extra_info

	def reset_algorithm_and_environment(self):
		raise ValueError("reset_algorithm_and_environment not implemented")

	def set_algorithm_parameters(self, params):
		raise ValueError("set_algorithm_parameters not implemented. ")


### TODO: description of this class.
class ContainerEnvironment:
	def __init__(self):
		pass







def test_modsel_algorithm(algorithm_environments, num_timesteps, modselalgo = "Corral", parameters_list = [] ):
	num_base_algorithms = len(inst_reward_data)
	index_per_algorithm = [0 for _ in range(num_base_algorithms)]

	modsel_manager = get_modsel_manager(modselalgo,confidence_radii, num_timesteps )

	### if algorithm_environments is of size one, we are in data sharing mode, else we are in the split data regime

	if len(algorithm_environments) > 1 and len(parameters_list) > 0:
		raise ValueError("Num algorithm_environments is more than one. Length of parameter list is nonzero.")

	if len(algorithm_environments) == 1 and len(parameters_list) = 0:
		raise ValueError("Num algorithm_environments equals one. Length of parameter list is zero.")

	rewards = []
	pseudo_rewards = []
	instantaneous_regrets = []
	probabilities = []
	modsel_infos = []

	for t in range(num_timesteps):
		print("Timestep {}".format(t))
		modsel_sample_idx = modsel_manager.sample_base_index()
		probabilities.append(modsel_manager.get_distribution())
		print("Selected base index {}".format(modsel_sample_idx))

		if parameters_list > 0:
			params = parameters_list[modsel_sample_idx]
			algorithm_environments[0].set_algorithm_parameters(params)
			algorithm_environment = algorithm_environments[0]

		else:
			algorithm_environment = algorithm_environments[modsel_sample_idx]


		instantenous_reward, instantenous_pseudo_reward, instantenous_regret, modsel_info = algorithm_environment.step()

		rewards.append(instantenous_reward)
		pseudo_rewards.append(instantenous_pseudo_reward)
		instantaneous_regrets.append(instantenous_regret)
		modsel_infos.append(model_info)

		modsel_manager.update_distribution(modsel_sample_idx, instantenous_reward, modsel_info)


	return rewards, pseudo_rewards, instantaneous_regrets, probabilities, modsel_infos












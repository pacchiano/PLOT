import numpy as np
import matplotlib.pyplot as plt
import ray

import IPython

from algorithmsmodsel import train_epsilon_greedy_modsel, train_mahalanobis_modsel
from algorithms import train_epsilon_greedy, train_mahalanobis, train_baseline
from algorithms_remote import train_epsilon_greedy_remote, train_epsilon_greedy_modsel_remote, train_baseline_remote


def process_results(results_list):
    mean = np.mean(results_list, 0)
    standard_dev = np.std(results_list, 0)

 
    return mean, standard_dev



USE_RAY = True

dataset = "Adult" ## "Adult", "Bank", "German"
num_batches = 2000
epsilon = .1
alpha = 10
epsilons = [.2, .1, .01, .05]#, .05]
alphas = [10, .1]#, .01, .001]
decaying_epsilon = False

batch_size = 10

colors = ["blue", "red", "orange", "black", "violet", "orange", "green", "brown", "gray"]


modselalgo = "Corral"

num_experiments = 10

results_dictionary = dict([])

if USE_RAY:
	baseline_results = [train_baseline_remote.remote(dataset, num_timesteps = 1000, 
    batch_size = 32, 
    MLP = True, representation_layer_size = 10) for _ in range(num_experiments)]
	baseline_results = ray.get(baseline_results)
else:

	# baseline_test_accuracy, baseline_model = train_baseline(dataset, num_timesteps = 1000, 
 #    batch_size = 32, 
 #    MLP = True, representation_layer_size = 10)
	baseline_results = [train_baseline(dataset, num_timesteps = 1000, 
    batch_size = 32, 
    MLP = True, representation_layer_size = 10) for _ in range(num_experiments)]

baseline_model = baseline_results[0][1]

results_dictionary["baseline"] = [x[1] for x in baseline_results]



### Run epsilon-greedy model selection experiments


if USE_RAY:
	
	epsilon_greedy_modsel_results = [train_epsilon_greedy_modsel_remote.remote(dataset, baseline_model, 
    num_batches = num_batches, batch_size = batch_size, 
    num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
    representation_layer_size = 10, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, epsilons = epsilons,
    restart_model_full_minimization = False, modselalgo = modselalgo) for _ in range(num_experiments)]
	epsilon_greedy_modsel_results = ray.get(epsilon_greedy_modsel_results)

else:


	epsilon_greedy_modsel_results = [train_epsilon_greedy_modsel(dataset, baseline_model, 
    num_batches = num_batches, batch_size = batch_size, 
    num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
    representation_layer_size = 10, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, epsilons = epsilons,
    restart_model_full_minimization = False, modselalgo = modselalgo) for _ in range(num_experiments)]
	

results_dictionary["epsilon {}".format(modselalgo)] = epsilon_greedy_modsel_results



### Run epsilon-greedy experiments

for epsilon in epsilons:

	if USE_RAY:
		epsilon_greedy_results =  [train_epsilon_greedy_remote.remote(dataset, baseline_model, 
		    num_batches = num_batches, batch_size = batch_size, 
		    num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
		    representation_layer_size = 10, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, epsilon = epsilon,
		    restart_model_full_minimization = False) for _ in range(num_experiments)]
		epsilon_greedy_results = ray.get(epsilon_greedy_results)
	else:



		epsilon_greedy_results = [train_epsilon_greedy(dataset, baseline_model, 
		    num_batches = num_batches, batch_size = batch_size, 
		    num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
		    representation_layer_size = 10, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, epsilon = epsilon,
		    restart_model_full_minimization = False)]

	results_dictionary["epsilon-{}".format(epsilon)] = epsilon_greedy_results





# for alpha in alphas:

# 	if USE_RAY:

# 		#train_epsilon_greedy_corral
# 		# instantaneous_epsilon_corral_regrets, instantaneous_epsilon_corral_accuracies, test_epsilon_corral_accuracy = train_epsilon_greedy_modsel(dataset, baseline_model, 
# 		#     num_batches = num_batches, batch_size = 32, 
# 		#     num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
# 		#     representation_layer_size = 10, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, epsilons = epsilons,
# 		#     restart_model_full_minimization = False, modselalgo = "Corral")

# 		mahalanobis_results = [train_epsilon_greedy_modsel_remote.remote(dataset, baseline_model, 
# 		    num_batches = num_batches, batch_size = 32, 
# 		    num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
# 		    representation_layer_size = 10, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, epsilons = epsilons,
# 		    restart_model_full_minimization = False, modselalgo = "Corral") for _ in range()]

# 		mahalanobis_results = ray.get(mahalanobis_results)

# 	else:


# 		mahalanobis_results = [train_epsilon_greedy_modsel(dataset, baseline_model, 
# 		    num_batches = num_batches, batch_size = 32, 
# 		    num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
# 		    representation_layer_size = 10, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, epsilons = epsilons,
# 		    restart_model_full_minimization = False, modselalgo = "Corral") for _ in range()]

# 	results_dictionary["alpha-{}".format(alpha)] = mahalanobis_results


Ts = np.arange(num_batches)+1
color_index = 0


# Plot epsilon-greedy model selection
epsilon_modsel_results = results_dictionary["epsilon {}".format(modselalgo)]
cummulative_epsilon_modsel_regrets = [np.cumsum(x[0]) for x in epsilon_modsel_results]
cummulative_epsilon_modsel_regrets_mean = np.mean(cummulative_epsilon_modsel_regrets,0)
cummulative_epsilon_modsel_regrets_std = np.std(cummulative_epsilon_modsel_regrets,0)

plt.plot(Ts, cummulative_epsilon_modsel_regrets_mean, color = colors[color_index] ,  label = "epsilon {}".format(modselalgo))
plt.fill_between(Ts, cummulative_epsilon_modsel_regrets_mean-.5*cummulative_epsilon_modsel_regrets_std, 
	cummulative_epsilon_modsel_regrets_mean+.5*cummulative_epsilon_modsel_regrets_std, color = colors[color_index], alpha = .2)

color_index += 1





# Plot epsilon-greedy models

for epsilon in epsilons:

	epsilon_results = results_dictionary["epsilon-{}".format(epsilon)] 
	cummulative_epsilon_regrets = [np.cumsum(x[0]) for x in epsilon_results]
	cummulative_epsilon_regrets_mean = np.mean(cummulative_epsilon_regrets,0)
	cummulative_epsilon_regrets_std = np.std(cummulative_epsilon_regrets,0)

	plt.plot(Ts, cummulative_epsilon_regrets_mean, color = colors[color_index] ,  label = "epsilon-{}".format(epsilon))
	plt.fill_between(Ts, cummulative_epsilon_regrets_mean-.5*cummulative_epsilon_regrets_std, 
		cummulative_epsilon_regrets_mean+.5*cummulative_epsilon_regrets_std, color = colors[color_index], alpha = .2)


	color_index += 1

	# IPython.embed()
	# raise ValueError("asldfkm")



plt.title("Regrets {}".format(dataset))
plt.xlabel("Number of batches")

plt.ylabel("Regret")
# plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
plt.legend(fontsize=8, loc="upper left")

plt.savefig("modsel_regret-epsilons.png")



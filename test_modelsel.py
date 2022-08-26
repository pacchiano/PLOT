import numpy as np
import matplotlib.pyplot as plt
import ray

import IPython

from algorithmsmodsel import train_epsilon_greedy_modsel, train_mahalanobis_modsel
from algorithms import train_epsilon_greedy, train_mahalanobis, train_baseline
from algorithms_remote import train_epsilon_greedy_remote, train_epsilon_greedy_modsel_remote, train_baseline_remote, train_mahalanobis_remote, train_mahalanobis_modsel_remote


def process_results(results_list):
    mean = np.mean(results_list, 0)
    standard_dev = np.std(results_list, 0)

 
    return mean, standard_dev



USE_RAY = True

PLOT_EPSILON = True
PLOT_MAHALANOBIS = True


## What is the fractrion of rejected labels for each algorithm and each dataset. 
dataset = "Adult" ## "Adult", "Bank", "German"
num_batches = 2000
epsilon = .1
alpha = 10
epsilons = [.2, .1, .01, .05]#, .05]
alphas = [10, 1, .1, .01]#, .01, .001]
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

if PLOT_EPSILON:

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




if PLOT_MAHALANOBIS:


	if USE_RAY:
		
		mahalanobis_greedy_modsel_results = [train_mahalanobis_modsel_remote.remote(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
	    representation_layer_size = 10, threshold = .5, verbose = True, alphas = alphas,
	    restart_model_full_minimization = False, modselalgo = modselalgo) for _ in range(num_experiments)]
		mahalanobis_greedy_modsel_results = ray.get(mahalanobis_greedy_modsel_results)

	else:


		mahalanobis_greedy_modsel_results = [train_mahalanobis_modsel(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
	    representation_layer_size = 10, threshold = .5, verbose = True, alphas = alphas,
	    restart_model_full_minimization = False, modselalgo = modselalgo) for _ in range(num_experiments)]
		

	results_dictionary["alpha {}".format(modselalgo)] = mahalanobis_greedy_modsel_results


	for alpha in alphas:

		if USE_RAY:

			#train_epsilon_greedy_corral
			# instantaneous_epsilon_corral_regrets, instantaneous_epsilon_corral_accuracies, test_epsilon_corral_accuracy = train_epsilon_greedy_modsel(dataset, baseline_model, 
			#     num_batches = num_batches, batch_size = 32, 
			#     num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
			#     representation_layer_size = 10, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, epsilons = epsilons,
			#     restart_model_full_minimization = False, modselalgo = "Corral")

			mahalanobis_results = [ train_mahalanobis_remote.remote(dataset, baseline_model, 
	   				 num_batches = num_batches, batch_size = batch_size, 
	   				 num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
	   				 representation_layer_size = 10, threshold = .5, verbose = True,  alpha = alpha, lambda_reg = 1, 
	   				 restart_model_full_minimization = False) for _ in range(num_experiments)]

			mahalanobis_results = ray.get(mahalanobis_results)

		else:


			mahalanobis_results = [train_mahalanobis(dataset, baseline_model, 
	   				 num_batches = num_batches, batch_size = batch_size, 
	   				 num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
	   				 representation_layer_size = 10, threshold = .5, verbose = True,  alpha = alpha, lambda_reg = 1, 
	   				 restart_model_full_minimization = False) for _ in range(num_experiments)]

		results_dictionary["alpha-{}".format(alpha)] = mahalanobis_results


Ts = np.arange(num_batches)+1
color_index = 0





if PLOT_EPSILON:


	# Plot epsilon-greedy model selection
	epsilon_modsel_results = results_dictionary["epsilon {}".format(modselalgo)]


	probs =[x[-1] for x in epsilon_modsel_results]

	color_index = 1

	#IPython.embed()
	mean_probs = np.mean(probs, 0)
	std_probs = np.std(probs, 0)

	for i in range(len(epsilons)):
		plt.plot(Ts, mean_probs[:, i], color = colors[color_index], label = "epsilon {}".format(epsilons[i]))
		plt.fill_between(Ts, mean_probs[:, i] - .5*std_probs[:, i], mean_probs[:, i] + .5*std_probs[:, i], color = colors[color_index], alpha = .2)

		color_index+=1


	plt.title("Probabilities evolution {} {}".format(modselalgo, dataset))
	plt.xlabel("Number of batches")
	plt.legend(fontsize=8, loc="upper left")

	plt.savefig("modsel_probabilities-epsilon_{}.png".format(dataset))

	plt.close("all")
	color_index = 0



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




	plt.title("Regrets {} {}".format(modselalgo, dataset))
	plt.xlabel("Number of batches")

	plt.ylabel("Regret")
	# plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
	plt.legend(fontsize=8, loc="upper left")


	plt.savefig("modsel_regret-epsilons_{}_{}.png".format(modselalgo,dataset))

	plt.close("all")


if PLOT_MAHALANOBIS:


	# Plot epsilon-greedy model selection
	mahalanobis_modsel_results = results_dictionary["alpha {}".format(modselalgo)]
	probs =[x[-1] for x in mahalanobis_modsel_results]



	#IPython.embed()
	mean_probs = np.mean(probs, 0)
	std_probs = np.std(probs, 0)

	color_index = 1
	for i in range(len(alphas)):
		plt.plot(Ts, mean_probs[:, i], color = colors[color_index], label = "alpha {}".format(alphas[i]))
		plt.fill_between(Ts, mean_probs[:, i] - .5*std_probs[:, i], mean_probs[:, i] + .5*std_probs[:, i], color = colors[color_index], alpha = .2)

		color_index+=1


	plt.title("Probabilities evolution {} {}".format(modselalgo,dataset))
	plt.xlabel("Number of batches")
	plt.legend(fontsize=8, loc="upper left")

	plt.savefig("modsel_probabilities-mahalanobis_{}_{}.png".format(modselalgo,dataset))

	plt.close("all")


	#IPython.embed()

	color_index = 0

	cummulative_mahalanobis_modsel_regrets = [np.cumsum(x[0]) for x in mahalanobis_modsel_results]
	cummulative_mahalanobis_modsel_regrets_mean = np.mean(cummulative_mahalanobis_modsel_regrets,0)
	cummulative_mahalanobis_modsel_regrets_std = np.std(cummulative_mahalanobis_modsel_regrets,0)

	plt.plot(Ts, cummulative_mahalanobis_modsel_regrets_mean, color = colors[color_index] ,  label = "alpha {}".format(modselalgo))
	plt.fill_between(Ts, cummulative_mahalanobis_modsel_regrets_mean-.5*cummulative_mahalanobis_modsel_regrets_std, 
		cummulative_mahalanobis_modsel_regrets_mean+.5*cummulative_mahalanobis_modsel_regrets_std, color = colors[color_index], alpha = .2)

	color_index += 1




	# Plot epsilon-greedy models

	for alpha in alphas:

		mahalanobis_results = results_dictionary["alpha-{}".format(alpha)] 
		cummulative_mahalanobis_regrets = [np.cumsum(x[0]) for x in mahalanobis_results]
		cummulative_mahalanobis_regrets_mean = np.mean(cummulative_mahalanobis_regrets,0)
		cummulative_mahalanobis_regrets_std = np.std(cummulative_mahalanobis_regrets,0)

		plt.plot(Ts, cummulative_mahalanobis_regrets_mean, color = colors[color_index] ,  label = "alpha-{}".format(alpha))
		plt.fill_between(Ts, cummulative_mahalanobis_regrets_mean-.5*cummulative_mahalanobis_regrets_std, 
			cummulative_mahalanobis_regrets_mean+.5*cummulative_mahalanobis_regrets_std, color = colors[color_index], alpha = .2)


		color_index += 1





	plt.title("Regrets {} {}".format(modselalgo, dataset))
	plt.xlabel("Number of batches")

	plt.ylabel("Regret")
	# plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
	plt.legend(fontsize=8, loc="upper left")



	plt.savefig("modsel_regret-mahalanobis_{} {}.png".format(modselalgo,dataset))
	plt.close("all")



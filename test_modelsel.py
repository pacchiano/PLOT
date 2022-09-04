import numpy as np
import matplotlib.pyplot as plt
import ray
import pickle
import IPython

from algorithmsmodsel import train_epsilon_greedy_modsel, train_mahalanobis_modsel
from algorithms import train_epsilon_greedy, train_mahalanobis, train_baseline
from algorithms_remote import train_epsilon_greedy_remote, train_epsilon_greedy_modsel_remote, train_baseline_remote, train_mahalanobis_remote, train_mahalanobis_modsel_remote


def process_results(results_list):
    mean = np.mean(results_list, 0)
    standard_dev = np.std(results_list, 0)

 
    return mean, standard_dev



def run_train_baseline(dataset, num_experiments, batch_size = 32, num_timesteps = 1000, representation_layer_size = 10):
	### BASELINE training
	if USE_RAY:
		baseline_results = [train_baseline_remote.remote(dataset, num_timesteps = num_timesteps, 
	    batch_size = batch_size, 
	    MLP = True, representation_layer_size = representation_layer_size) for _ in range(num_experiments)]
		baseline_results = ray.get(baseline_results)
	else:
		baseline_results = [train_baseline(dataset, num_timesteps = num_timesteps, 
	    batch_size = batch_size, 
	    MLP = True, representation_layer_size = representation_layer_size) for _ in range(num_experiments)]

	baseline_model = baseline_results[0][1]	

	return baseline_results, baseline_model


def run_epsilon_greedy_experiments(dataset, epsilons, modselalgo, num_experiments, baseline_model, 
	num_batches, batch_size, decaying_epsilon, num_opt_steps = 1000, 
	opt_batch_size = 20, representation_layer_size = 10, restart_model_full_minimization = False):
	if USE_RAY:
		
		epsilon_greedy_modsel_results = [train_epsilon_greedy_modsel_remote.remote(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, MLP = True, 
	    representation_layer_size = representation_layer_size, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, 
	    		epsilons = epsilons, restart_model_full_minimization = restart_model_full_minimization, 
	    		modselalgo = modselalgo) for _ in range(num_experiments)]
		epsilon_greedy_modsel_results = ray.get(epsilon_greedy_modsel_results)

	else:


		epsilon_greedy_modsel_results = [train_epsilon_greedy_modsel(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, MLP = True, 
	    representation_layer_size = representation_layer_size, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, 
	    	epsilons = epsilons, restart_model_full_minimization = restart_model_full_minimization, 
	    	modselalgo = modselalgo) for _ in range(num_experiments)]
		

	#results_dictionary["epsilon {}".format(modselalgo)] = epsilon_greedy_modsel_results


	epsilon_greedy_results_list = []

	### Run epsilon-greedy experiments
	for epsilon in epsilons:

		if USE_RAY:
			epsilon_greedy_results =  [train_epsilon_greedy_remote.remote(dataset, baseline_model, 
			    num_batches = num_batches, batch_size = batch_size, 
			    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, MLP = True, 
			    representation_layer_size = representation_layer_size, threshold = .5, verbose = True, 
			    	decaying_epsilon = decaying_epsilon, epsilon = epsilon,
			    	restart_model_full_minimization = restart_model_full_minimization) for _ in range(num_experiments)]
			epsilon_greedy_results = ray.get(epsilon_greedy_results)
		else:



			epsilon_greedy_results = [train_epsilon_greedy(dataset, baseline_model, 
			    num_batches = num_batches, batch_size = batch_size, 
			    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, MLP = True, 
			    representation_layer_size = representation_layer_size, threshold = .5, verbose = True, 
			    decaying_epsilon = decaying_epsilon, epsilon = epsilon,
			    restart_model_full_minimization = restart_model_full_minimization)]

		#results_dictionary["epsilon-{}".format(epsilon)] = epsilon_greedy_results

		epsilon_greedy_results_list.append(("epsilon-{}".format(epsilon), epsilon_greedy_results))

	return ("epsilon {}".format(modselalgo), epsilon_greedy_modsel_results), epsilon_greedy_results_list



def run_mahalanobis_experiments(dataset, alphas, modselalgo, num_experiments, baseline_model, num_batches, 
	batch_size, decaying_epsilon, num_opt_steps = 1000, 
	opt_batch_size = 20, representation_layer_size = 10, restart_model_full_minimization = False):


	if USE_RAY:
		
		mahalanobis_modsel_results = [train_mahalanobis_modsel_remote.remote(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, MLP = True, 
	    representation_layer_size = representation_layer_size, threshold = .5, verbose = True, alphas = alphas,
	    restart_model_full_minimization = False, modselalgo = modselalgo) for _ in range(num_experiments)]
		mahalanobis_modsel_results = ray.get(mahalanobis_modsel_results)

	else:


		mahalanobis_modsel_results = [train_mahalanobis_modsel(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, MLP = True, 
	    representation_layer_size = representation_layer_size, threshold = .5, verbose = True, alphas = alphas,
	    restart_model_full_minimization = False, modselalgo = modselalgo) for _ in range(num_experiments)]
		

	results_dictionary["alpha {}".format(modselalgo)] = mahalanobis_modsel_results

	mahalanobis_results_list = []

	for alpha in alphas:

		if USE_RAY:


			mahalanobis_results = [ train_mahalanobis_remote.remote(dataset, baseline_model, 
	   				 num_batches = num_batches, batch_size = batch_size, 
	   				 num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, MLP = True, 
	   				 representation_layer_size = representation_layer_size, threshold = .5, verbose = True,  alpha = alpha, 
	   				 lambda_reg = 1, restart_model_full_minimization = False) for _ in range(num_experiments)]

			mahalanobis_results = ray.get(mahalanobis_results)

		else:


			mahalanobis_results = [train_mahalanobis(dataset, baseline_model, 
	   				 num_batches = num_batches, batch_size = batch_size, 
	   				 num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, MLP = True, 
	   				 representation_layer_size = representation_layer_size, threshold = .5, verbose = True,  alpha = alpha, 
	   				 lambda_reg = 1, restart_model_full_minimization = False) for _ in range(num_experiments)]

		mahalanobis_results_list.append(("alpha-{}".format(alpha), mahalanobis_results))
		#results_dictionary["alpha-{}".format(alpha)] = mahalanobis_results
	
	return ("alpha {}".format(modselalgo),mahalanobis_modsel_results ), mahalanobis_results_list


def plot_modsel_probabilities(algo_name, dataset, num_batches, batch_size, modselalgo, 
	results_dictionary, hyperparams, colors):
	modsel_results = results_dictionary["{} {}".format(algo_name, modselalgo)]


	probs =[x["modselect_info"] for x in modsel_results]

	color_index = 1

	#IPython.embed()
	mean_probs = np.mean(probs, 0)
	std_probs = np.std(probs, 0)

	for i in range(len(hyperparams)):
		plt.plot(Ts, mean_probs[:, i], color = colors[color_index], label = "{} {}".format(algo_name, hyperparams[i]))
		plt.fill_between(Ts, mean_probs[:, i] - .5*std_probs[:, i], mean_probs[:, i] + .5*std_probs[:, i], 
			color = colors[color_index], alpha = .2)

		color_index+=1


	plt.title("Probabilities evolution {} {} B{}".format(modselalgo, dataset, batch_size))
	plt.xlabel("Number of batches")
	plt.legend(fontsize=8, loc="upper left")

	plt.savefig("./ModselResults/modsel_probabilities-{}_{}_T{}_B{}.png".format(algo_name, dataset,num_batches,batch_size))

	plt.close("all")
	





def plot_results(algo_name, dataset, results_type, num_batches, batch_size, modselalgo, 
	results_dictionary, hyperparams, colors, cummulative_plot = False ):


	if results_type != "instantaneous_regrets" and cummulative_plot == True:
		raise ValueError("Results type {} does not support cummulative plot".format(results_type))


	##### PLOTTING instantaneous regrets.
	modsel_results = results_dictionary["{} {}".format(algo_name, modselalgo)]

	color_index = 0


	if cummulative_plot:
		modsel_stats = [np.cumsum(x[results_type]) for x in modsel_results]
	else:
		modsel_stats = [x[results_type] for x in modsel_results]


	modsel_stat_mean = np.mean(modsel_stats,0)
	modsel_stat_std = np.std(modsel_stats,0)



	plt.plot(Ts, modsel_stat_mean, color = colors[color_index] ,  label = "{} {}".format(algo_name, modselalgo))
	plt.fill_between(Ts, modsel_stat_mean-.5*modsel_stat_std, 
		modsel_stat_mean+.5*modsel_stat_std, color = colors[color_index], alpha = .2)

	color_index += 1


	
	# Plot epsilon-greedy models

	for hyperparam in hyperparams:

		hyperparam_results = results_dictionary["{}-{}".format(algo_name, hyperparam)] 
		if cummulative_plot:
			hyperparam_stats = [np.cumsum(x[results_type]) for x in hyperparam_results]
		else:
			hyperparam_stats = [x[results_type] for x in hyperparam_results]

		#IPython.embed()


		hyperparam_results_mean = np.mean(hyperparam_stats,0)
		hyperparam_results_std = np.std(hyperparam_stats,0)

		plt.plot(Ts, hyperparam_results_mean, color = colors[color_index] ,  label = "{}-{}".format(algo_name,hyperparam))
		plt.fill_between(Ts, hyperparam_results_mean-.5*hyperparam_results_std, 
			hyperparam_results_mean+.5*hyperparam_results_std, color = colors[color_index], alpha = .2)


		color_index += 1

		# IPython.embed()
		# raise ValueError("asldfkm")




	##### get label
	label = results_type
	if results_type == "instantaneous_regrets" and cummulative_plot == True:
		label = "Regret"

	elif results_type == "instantaneous_regrets" and cummulative_plot == False:
		label = "Instantaneous regrets"


	elif results_type == "instantaneous_accuracies":
		label = "Instantaneous accuracies"

	elif results_type == "num_negatives":
		label = "Negatives"

	elif results_type == "num_positives":
		label = "Positives"

	elif results_type == "false_neg_rates":
		label = "False Negatives"

	elif results_type == "false_positive_rates":
		label = "False Positives"
	else:
		raise ValueError("Unrecognized option {}".format(results_type))


	plt.title("{} {} {} B{}".format( label, modselalgo, dataset, batch_size))
	plt.xlabel("Number of batches")

	plt.ylabel(label)
	# plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
	plt.legend(fontsize=8, loc="upper left")


	plt.savefig("./ModselResults/modsel_{}_cum_{}-{}_{}_{}_T{}_B{}.png".format(results_type, cummulative_plot, 
		algo_name, modselalgo,dataset, num_batches, batch_size))

	plt.close("all")














USE_RAY = True

PLOT_EPSILON = True
PLOT_MAHALANOBIS = True


## What is the fractrion of rejected labels for each algorithm and each dataset. 
dataset = "Bank"
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

baseline_results, baseline_model = run_train_baseline(dataset, num_experiments)

results_dictionary["baseline"] = [x[1] for x in baseline_results]


# results_dictionary["epsilon {}".format(modselalgo)] = epsilon_greedy_modsel_results


### Run epsilon-greedy model selection experiments


if PLOT_EPSILON:

	epsilon_greedy_modsel_results_tuple, epsilon_greedy_results_list  =	run_epsilon_greedy_experiments(dataset, epsilons, modselalgo, num_experiments, baseline_model, num_batches, batch_size, decaying_epsilon, num_opt_steps = 1000, 
		opt_batch_size = 20, representation_layer_size = 10, restart_model_full_minimization = False)

	results_dictionary[epsilon_greedy_modsel_results_tuple[0]] = epsilon_greedy_modsel_results_tuple[1]

	for eps_res_tuple in epsilon_greedy_results_list:
		results_dictionary[eps_res_tuple[0]] = eps_res_tuple[1]


if PLOT_MAHALANOBIS:
	mahalanobis_modsel_results_tuple, mahalanobis_results_list = run_mahalanobis_experiments(dataset, alphas, modselalgo, num_experiments, baseline_model, num_batches, batch_size, decaying_epsilon, num_opt_steps = 1000, 
		opt_batch_size = 20, representation_layer_size = 10, restart_model_full_minimization = False)

	results_dictionary[mahalanobis_modsel_results_tuple[0]] = mahalanobis_modsel_results_tuple[1]
	for mahalanobis_res_tuple in mahalanobis_results_list:
		results_dictionary[mahalanobis_res_tuple[0]] = mahalanobis_res_tuple[1]	


Ts = np.arange(num_batches)+1
color_index = 0



pickle_results_filename = "results_modsel_{}_T{}_B{}.p".format(dataset, num_batches,batch_size )


pickle.dump(results_dictionary, 
    open("ModselResults/{}".format(pickle_results_filename), "wb"))



# IPython.embed()
# raise ValueError("asldkfm")

if PLOT_EPSILON:


	epsilon_modsel_results = results_dictionary["epsilon {}".format(modselalgo)]



	plot_modsel_probabilities("epsilon", dataset, num_batches, batch_size, modselalgo, 
		results_dictionary, epsilons, colors)



	plot_results("epsilon", dataset, "instantaneous_regrets", num_batches, batch_size, modselalgo, 
		results_dictionary, epsilons, colors, cummulative_plot = True )


	plot_results("epsilon", dataset, "instantaneous_accuracies", num_batches, batch_size, modselalgo, 
		results_dictionary, epsilons, colors, cummulative_plot = False )

	plot_results("epsilon", dataset, "num_negatives", num_batches, batch_size, modselalgo, 
		results_dictionary, epsilons, colors, cummulative_plot = False )

	plot_results("epsilon", dataset, "num_positives", num_batches, batch_size, modselalgo, 
		results_dictionary, epsilons, colors, cummulative_plot = False )

	plot_results("epsilon", dataset, "false_neg_rates", num_batches, batch_size, modselalgo, 
		results_dictionary, epsilons, colors, cummulative_plot = False )

	plot_results("epsilon", dataset, "false_positive_rates", num_batches, batch_size, modselalgo, 
		results_dictionary, epsilons, colors, cummulative_plot = False )







if PLOT_MAHALANOBIS:


	plot_modsel_probabilities("alpha", dataset, num_batches, batch_size, modselalgo, 
		results_dictionary, alphas, colors)


	# # Plot epsilon-greedy model selection
	# probs =[x["modselect_info"] for x in mahalanobis_modsel_results]



	plot_results("alpha", dataset, "instantaneous_regrets", num_batches, batch_size, modselalgo, 
		results_dictionary, alphas, colors, cummulative_plot = True )


	plot_results("alpha", dataset, "instantaneous_accuracies", num_batches, batch_size, modselalgo, 
		results_dictionary, alphas, colors, cummulative_plot = False )

	plot_results("alpha", dataset, "num_negatives", num_batches, batch_size, modselalgo, 
		results_dictionary, alphas, colors, cummulative_plot = False )

	plot_results("alpha", dataset, "num_positives", num_batches, batch_size, modselalgo, 
		results_dictionary, alphas, colors, cummulative_plot = False )

	plot_results("alpha", dataset, "false_neg_rates", num_batches, batch_size, modselalgo, 
		results_dictionary, alphas, colors, cummulative_plot = False )

	plot_results("alpha", dataset, "false_positive_rates", num_batches, batch_size, modselalgo, 
		results_dictionary, alphas, colors, cummulative_plot = False )





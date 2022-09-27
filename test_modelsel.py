import numpy as np
import matplotlib.pyplot as plt
import ray
import pickle
import IPython

from algorithmsmodsel import train_epsilon_greedy_modsel, train_mahalanobis_modsel
from algorithms import train_epsilon_greedy, train_mahalanobis, train_baseline
from algorithms_remote import train_epsilon_greedy_remote, train_epsilon_greedy_modsel_remote, train_baseline_remote, train_mahalanobis_remote, train_mahalanobis_modsel_remote


USE_RAY = True


def process_results(results_list):
    mean = np.mean(results_list, 0)
    standard_dev = np.std(results_list, 0)

 
    return mean, standard_dev



def run_train_baseline(dataset, num_experiments, batch_size = 32, num_timesteps = 1000, representation_layer_sizes = [10,10]):
	### BASELINE training
	if USE_RAY:
		baseline_results = [train_baseline_remote.remote(dataset, num_timesteps = num_timesteps, 
	    batch_size = batch_size, 
	    representation_layer_sizes = representation_layer_sizes) for _ in range(num_experiments)]
		baseline_results = ray.get(baseline_results)
	else:
		baseline_results = [train_baseline(dataset, num_timesteps = num_timesteps, 
	    batch_size = batch_size, 
	    representation_layer_sizes = representation_layer_sizes) for _ in range(num_experiments)]

	baseline_model = baseline_results[0][1]	

	return baseline_results, baseline_model


def run_epsilon_greedy_experiments(dataset, epsilons, modselalgo, num_experiments, baseline_model, 
	num_batches, batch_size, decaying_epsilon, num_opt_steps = 1000, 
	opt_batch_size = 20, representation_layer_sizes = [10, 10], restart_model_full_minimization = False):
	if USE_RAY:
		
		epsilon_greedy_modsel_results = [train_epsilon_greedy_modsel_remote.remote(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, 
	    representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, 
	    		epsilons = epsilons, restart_model_full_minimization = restart_model_full_minimization, 
	    		modselalgo = modselalgo) for _ in range(num_experiments)]
		epsilon_greedy_modsel_results = ray.get(epsilon_greedy_modsel_results)

	else:


		epsilon_greedy_modsel_results = [train_epsilon_greedy_modsel(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size,
	    representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True, decaying_epsilon = decaying_epsilon, 
	    	epsilons = epsilons, restart_model_full_minimization = restart_model_full_minimization, 
	    	modselalgo = modselalgo) for _ in range(num_experiments)]
		

	#results_dictionary["epsilon {}".format(modselalgo)] = epsilon_greedy_modsel_results


	epsilon_greedy_results_list = []

	### Run epsilon-greedy experiments
	for epsilon in epsilons:

		if USE_RAY:
			epsilon_greedy_results =  [train_epsilon_greedy_remote.remote(dataset, baseline_model, 
			    num_batches = num_batches, batch_size = batch_size, 
			    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size,
			    representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True, 
			    	decaying_epsilon = decaying_epsilon, epsilon = epsilon,
			    	restart_model_full_minimization = restart_model_full_minimization) for _ in range(num_experiments)]
			epsilon_greedy_results = ray.get(epsilon_greedy_results)
		else:



			epsilon_greedy_results = [train_epsilon_greedy(dataset, baseline_model, 
			    num_batches = num_batches, batch_size = batch_size, 
			    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size,
			    representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True, 
			    decaying_epsilon = decaying_epsilon, epsilon = epsilon,
			    restart_model_full_minimization = restart_model_full_minimization)]

		#results_dictionary["epsilon-{}".format(epsilon)] = epsilon_greedy_results

		epsilon_greedy_results_list.append(("epsilon-{}".format(epsilon), epsilon_greedy_results))

	return ("epsilon {}".format(modselalgo), epsilon_greedy_modsel_results), epsilon_greedy_results_list



def run_mahalanobis_experiments(dataset, alphas, modselalgo, num_experiments, baseline_model, num_batches, 
	batch_size, decaying_epsilon, num_opt_steps = 1000, 
	opt_batch_size = 20, representation_layer_sizes = [10, 10], restart_model_full_minimization = False):


	if USE_RAY:
		
		mahalanobis_modsel_results = [train_mahalanobis_modsel_remote.remote(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, 
	    representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True, alphas = alphas,
	    restart_model_full_minimization = False, modselalgo = modselalgo) for _ in range(num_experiments)]
		mahalanobis_modsel_results = ray.get(mahalanobis_modsel_results)

	else:


		mahalanobis_modsel_results = [train_mahalanobis_modsel(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, 
	    representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True, alphas = alphas,
	    restart_model_full_minimization = False, modselalgo = modselalgo) for _ in range(num_experiments)]
		

	results_dictionary["alpha {}".format(modselalgo)] = mahalanobis_modsel_results

	mahalanobis_results_list = []

	for alpha in alphas:

		if USE_RAY:


			mahalanobis_results = [ train_mahalanobis_remote.remote(dataset, baseline_model, 
	   				 num_batches = num_batches, batch_size = batch_size, 
	   				 num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, 
	   				 representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True,  alpha = alpha, 
	   				 lambda_reg = 1, restart_model_full_minimization = False) for _ in range(num_experiments)]

			mahalanobis_results = ray.get(mahalanobis_results)

		else:


			mahalanobis_results = [train_mahalanobis(dataset, baseline_model, 
	   				 num_batches = num_batches, batch_size = batch_size, 
	   				 num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, 
	   				 representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True,  alpha = alpha, 
	   				 lambda_reg = 1, restart_model_full_minimization = False) for _ in range(num_experiments)]

		mahalanobis_results_list.append(("alpha-{}".format(alpha), mahalanobis_results))
		#results_dictionary["alpha-{}".format(alpha)] = mahalanobis_results
	
	return ("alpha {}".format(modselalgo),mahalanobis_modsel_results ), mahalanobis_results_list





def plot_modsel_probabilities(algo_name, dataset, num_batches, batch_size, modselalgo, 
	results_dictionary, hyperparams, colors, representation_layer_sizes, averaging_window = 1):

	Ts = np.arange(num_batches)+1
	color_index = 0


	modsel_results = results_dictionary["{} {}".format(algo_name, modselalgo)]


	probs =np.array([x["modselect_info"] for x in modsel_results])

	color_index = 1

	#IPython.embed()
	mean_probs = np.mean(probs, 0)
	std_probs = np.std(probs, 0)

	for i in range(len(hyperparams)):
		plt.plot(Ts, mean_probs[:, i], color = colors[color_index], label = "{} {}".format(algo_name, hyperparams[i]))
		plt.fill_between(Ts, mean_probs[:, i] - .5*std_probs[:, i], mean_probs[:, i] + .5*std_probs[:, i], 
			color = colors[color_index], alpha = .2)

		color_index+=1

	repres_layers_name = get_architecture_name(representation_layer_sizes)

	plt.title("Probabilities evolution {} {} B{} N {}".format(modselalgo, dataset, batch_size, repres_layers_name))
	plt.xlabel("Number of batches")
	plt.legend(fontsize=8, loc="upper left")

	plt.savefig("./ModselResults/modsel_probabilities-{}_{}_{}_T{}_B{}_N_{}.png".format(modselalgo,algo_name, dataset,num_batches,batch_size, repres_layers_name))

	plt.close("all")
	





def plot_results(algo_name, dataset, results_type, num_batches, batch_size, modselalgo, 
	results_dictionary, hyperparams, colors, representation_layer_sizes, cummulative_plot = False, averaging_window = 1 ):


	Ts = (np.arange(num_batches/averaging_window)+1)*averaging_window
	color_index = 0


	if results_type != "instantaneous_regrets" and cummulative_plot == True:
		raise ValueError("Results type {} does not support cummulative plot".format(results_type))


	##### PLOTTING modsel results.
	modsel_results = results_dictionary["{} {}".format(algo_name, modselalgo)]

	color_index = 0


	if cummulative_plot:
		modsel_stats = np.array([np.cumsum(x[results_type]) for x in modsel_results])
	else:
		modsel_stats = np.array([x[results_type] for x in modsel_results])


	modsel_stat_mean = np.mean(modsel_stats,0)
	modsel_stat_std = np.std(modsel_stats,0)


	#IPython.embed()

	modsel_stat_mean = np.mean(modsel_stat_mean.reshape(int(num_batches/averaging_window), averaging_window), 1)
	modsel_stat_std = np.mean(modsel_stat_std.reshape(int(num_batches/averaging_window), averaging_window), 1)



	plt.plot(Ts, modsel_stat_mean, color = colors[color_index] ,  label = "{} {}".format(algo_name, modselalgo))
	plt.fill_between(Ts, modsel_stat_mean-.5*modsel_stat_std, 
		modsel_stat_mean+.5*modsel_stat_std, color = colors[color_index], alpha = .2)

	color_index += 1


	
	# Plot hyper sweep results

	for hyperparam in hyperparams:

		hyperparam_results = results_dictionary["{}-{}".format(algo_name, hyperparam)] 
		if cummulative_plot:
			hyperparam_stats = np.array([np.cumsum(x[results_type]) for x in hyperparam_results])
		else:
			hyperparam_stats = np.array([x[results_type] for x in hyperparam_results])





		hyperparam_results_mean = np.mean(hyperparam_stats,0)
		hyperparam_results_std = np.std(hyperparam_stats,0)




		hyperparam_results_mean = np.mean(hyperparam_results_mean.reshape(int(num_batches/averaging_window), averaging_window), 1)
		hyperparam_results_std = np.mean(hyperparam_results_std.reshape(int(num_batches/averaging_window), averaging_window), 1)








		#IPython.embed()
		


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
	

	repres_layers_name = get_architecture_name(representation_layer_sizes)


	plt.title("{} {} {} B{} N {}".format( label, modselalgo, dataset, batch_size, repres_layers_name))
	plt.xlabel("Number of batches")

	plt.ylabel(label)
	# plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
	plt.legend(fontsize=8, loc="upper left")


	plt.savefig("./ModselResults/modsel_{}_cum_{}-{}_{}_{}_T{}_B{}_N_{}.png".format(results_type, cummulative_plot, 
		algo_name, modselalgo,dataset, num_batches, batch_size, repres_layers_name))

	plt.close("all")







def get_architecture_name(representation_layer_sizes):

	### Get repres_layer_name 
	if len(representation_layer_sizes) == 0:
		repres_layers_name = "linear"
	else:
		repres_layers_name = "{}".format(representation_layer_sizes[0])
		for layer_size in representation_layer_sizes[1:]:
			repres_layers_name += "_{}".format(layer_size)

	return repres_layers_name






RUN_EPSILON = False
PLOT_EPSILON = False

RUN_MAHALANOBIS = True
PLOT_MAHALANOBIS = True


## What is the fractrion of rejected labels for each algorithm and each dataset. 
num_batches = 2000
averaging_window = 1
epsilon = .1
alpha = 10
epsilons = [.2, .1, .01, .05]#, .05]
alphas = [1/4.0, 1/2.0, 1, 2, 4, 8 ]#, .01, .001]
decaying_epsilon = False

batch_size = 10
num_experiments = 10

representation_layer_sizes = [10,10]



colors = ["blue", "red", "orange", "black", "violet", "orange", "green", "brown", "gray"]

modselalgos = ["EpochBalancing"]#"BalancingAnalytic", "BalancingSimple", "BalancingAnalyticHybrid" ,"Corral", "CorralAnytime"]
datasets = ["Adult"]#, "Crime", "German", "Bank"]

repres_layers_name = get_architecture_name(representation_layer_sizes)


results_dictionary = dict([])

for dataset in datasets:
	for modselalgo in modselalgos:


		if RUN_EPSILON or RUN_MAHALANOBIS:

			baseline_results, baseline_model = run_train_baseline(dataset, num_experiments)

			results_dictionary["baseline"] = [x[1] for x in baseline_results]


		### Run epsilon-greedy model selection experiments


		if RUN_EPSILON:

			epsilon_greedy_modsel_results_tuple, epsilon_greedy_results_list  =	run_epsilon_greedy_experiments(dataset, epsilons, modselalgo, num_experiments, baseline_model, num_batches, batch_size, decaying_epsilon, num_opt_steps = 1000, 
				opt_batch_size = 20, representation_layer_sizes = representation_layer_sizes, restart_model_full_minimization = False)

			results_dictionary[epsilon_greedy_modsel_results_tuple[0]] = epsilon_greedy_modsel_results_tuple[1]

			for eps_res_tuple in epsilon_greedy_results_list:
				results_dictionary[eps_res_tuple[0]] = eps_res_tuple[1]


		if RUN_MAHALANOBIS:
			mahalanobis_modsel_results_tuple, mahalanobis_results_list = run_mahalanobis_experiments(dataset, alphas, modselalgo, num_experiments, baseline_model, num_batches, batch_size, decaying_epsilon, num_opt_steps = 1000, 
				opt_batch_size = 20, representation_layer_sizes = representation_layer_sizes, restart_model_full_minimization = False)

			results_dictionary[mahalanobis_modsel_results_tuple[0]] = mahalanobis_modsel_results_tuple[1]
			for mahalanobis_res_tuple in mahalanobis_results_list:
				results_dictionary[mahalanobis_res_tuple[0]] = mahalanobis_res_tuple[1]	







		pickle_results_filename = "results_modsel_{}_T{}_B{}_N_{}.p".format(dataset, num_batches,batch_size,repres_layers_name )

		if RUN_EPSILON or RUN_MAHALANOBIS:
			pickle.dump(results_dictionary, 
			    open("ModselResults/{}".format(pickle_results_filename), "wb"))


		results_dictionary = pickle.load(open("ModselResults/{}".format(pickle_results_filename), "rb")) 


		Ts = np.arange(num_batches)+1
		color_index = 0

		if PLOT_EPSILON:


			epsilon_modsel_results = results_dictionary["epsilon {}".format(modselalgo)]



			plot_modsel_probabilities("epsilon", dataset, num_batches, batch_size, modselalgo, 
				results_dictionary, epsilons, colors, representation_layer_sizes = representation_layer_sizes)



			plot_results("epsilon", dataset, "instantaneous_regrets", num_batches, batch_size, modselalgo, 
				results_dictionary, epsilons, colors, representation_layer_sizes = representation_layer_sizes,
				 cummulative_plot = True , averaging_window = averaging_window)


			plot_results("epsilon", dataset, "instantaneous_accuracies", num_batches, batch_size, modselalgo, 
				results_dictionary, epsilons, colors, representation_layer_sizes = representation_layer_sizes,
				cummulative_plot = False, averaging_window = averaging_window )

			plot_results("epsilon", dataset, "num_negatives", num_batches, batch_size, modselalgo, 
				results_dictionary, epsilons, colors, representation_layer_sizes = representation_layer_sizes,
				 cummulative_plot = False, averaging_window = averaging_window )

			plot_results("epsilon", dataset, "num_positives", num_batches, batch_size, modselalgo, 
				results_dictionary, epsilons, colors, representation_layer_sizes = representation_layer_sizes,
				 cummulative_plot = False, averaging_window = averaging_window )

			plot_results("epsilon", dataset, "false_neg_rates", num_batches, batch_size, modselalgo, 
				results_dictionary, epsilons, colors, representation_layer_sizes = representation_layer_sizes,
				 cummulative_plot = False, averaging_window = averaging_window )

			plot_results("epsilon", dataset, "false_positive_rates", num_batches, batch_size, modselalgo, 
				results_dictionary, epsilons, colors, representation_layer_sizes = representation_layer_sizes,
				cummulative_plot = False, averaging_window = averaging_window )







		if PLOT_MAHALANOBIS:


			plot_modsel_probabilities("alpha", dataset, num_batches, batch_size, modselalgo, 
				results_dictionary, alphas, colors, representation_layer_sizes = representation_layer_sizes)


			plot_results("alpha", dataset, "instantaneous_regrets", num_batches, batch_size, modselalgo, 
				results_dictionary, alphas, colors, representation_layer_sizes = representation_layer_sizes,
				cummulative_plot = True , averaging_window = averaging_window)


			plot_results("alpha", dataset, "instantaneous_accuracies", num_batches, batch_size, modselalgo, 
				results_dictionary, alphas, colors, representation_layer_sizes = representation_layer_sizes,
				cummulative_plot = False, averaging_window = averaging_window )

			plot_results("alpha", dataset, "num_negatives", num_batches, batch_size, modselalgo, 
				results_dictionary, alphas, colors, representation_layer_sizes = representation_layer_sizes,
				cummulative_plot = False , averaging_window = averaging_window)

			plot_results("alpha", dataset, "num_positives", num_batches, batch_size, modselalgo, 
				results_dictionary, alphas, colors, representation_layer_sizes = representation_layer_sizes,
				cummulative_plot = False , averaging_window = averaging_window)

			plot_results("alpha", dataset, "false_neg_rates", num_batches, batch_size, modselalgo, 
				results_dictionary, alphas, colors, representation_layer_sizes = representation_layer_sizes,
				cummulative_plot = False , averaging_window = averaging_window)

			plot_results("alpha", dataset, "false_positive_rates", num_batches, batch_size, modselalgo, 
				results_dictionary, alphas, colors, representation_layer_sizes = representation_layer_sizes,
				cummulative_plot = False , averaging_window = averaging_window)





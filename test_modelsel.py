import numpy as np
import matplotlib.pyplot as plt
import ray
import pickle
import sys
import IPython
import os
from copy import deepcopy
import matplotlib.colors

from utilities import pickle_and_zip
from algorithmsmodsel import train_epsilon_greedy_modsel, train_mahalanobis_modsel, train_opt_reg_modsel
from algorithms import train_epsilon_greedy, train_mahalanobis, train_baseline, train_opt_reg
from algorithms_remote import train_epsilon_greedy_remote, train_epsilon_greedy_modsel_remote, train_baseline_remote, train_mahalanobis_remote, train_mahalanobis_modsel_remote, train_opt_reg_modsel_remote, train_opt_reg_remote


#USE_RAY = False
PLOT_ALL_STATS = False

def process_results(results_list):
    mean = np.mean(results_list, 0)
    standard_dev = np.std(results_list, 0)

 
    return mean, standard_dev


def get_pickle_filename_stub(dataset, num_batches,batch_size,repres_layers_name, split):
	if not split:
		filename_stub = "results_modsel_{}_T{}_B{}_N_{}".format(dataset, num_batches,batch_size,repres_layers_name )

	else:
		filename_stub = "results_modsel-split_{}_T{}_B{}_N_{}".format(dataset, num_batches,batch_size,repres_layers_name )

	return filename_stub





#def run_train_baseline(dataset, num_experiments, batch_size = 32, num_timesteps = 1000, representation_layer_sizes = [10,10]):
def run_train_baseline(dataset, num_experiments, batch_size = 32, num_timesteps = 10000, 
	representation_layer_sizes = [10,10]):
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
	opt_batch_size = 20, representation_layer_sizes = [10, 10], restart_model_full_minimization = False, split = False):
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

	return ("epsilon split{} {}".format(split, modselalgo), epsilon_greedy_modsel_results), epsilon_greedy_results_list



def run_modsel_mahalanobis_experiments(dataset, alphas, modselalgo, num_experiments, baseline_model, num_batches, 
	batch_size, num_opt_steps = 1000, 
	opt_batch_size = 20, representation_layer_sizes = [10, 10], restart_model_full_minimization = False, split = False):

	if USE_RAY:
		mahalanobis_modsel_results = [train_mahalanobis_modsel_remote.remote(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, 
	    representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True, alphas = alphas,
	    restart_model_full_minimization = restart_model_full_minimization, modselalgo = modselalgo,
	    split = split) for _ in range(num_experiments)]
		mahalanobis_modsel_results = ray.get(mahalanobis_modsel_results)

	else:
		mahalanobis_modsel_results = [train_mahalanobis_modsel(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, 
	    representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True, alphas = alphas,
	    restart_model_full_minimization = restart_model_full_minimization, modselalgo = modselalgo,
	    split = split) for _ in range(num_experiments)]
		
	return ("alpha split{} {}".format(split, modselalgo),mahalanobis_modsel_results )#, mahalanobis_results_list



def run_base_mahalanobis_experiments(dataset, alphas, num_experiments, baseline_model, num_batches, 
	batch_size, num_opt_steps = 1000, 
	opt_batch_size = 20, representation_layer_sizes = [10, 10], restart_model_full_minimization = False):

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
	
	return mahalanobis_results_list



def run_base_opt_reg_experiments(dataset, regs, num_experiments, baseline_model, num_batches, 
	batch_size, num_opt_steps = 1000, 
	opt_batch_size = 20, representation_layer_sizes = [10, 10], 
	restart_model_full_minimization = False, burn_in = 0):

	opt_reg_results_list = []

	for reg in regs:

		if USE_RAY:

			opt_reg_results = [train_opt_reg_remote.remote(dataset, baseline_model, num_batches, batch_size, 
    			num_opt_steps, opt_batch_size,
    			representation_layer_sizes = representation_layer_sizes, threshold = .5, reg = reg,
    			verbose = True,
    			restart_model_full_minimization = restart_model_full_minimization, burn_in = burn_in) for _ in range(num_experiments)]



			opt_reg_results = ray.get(opt_reg_results)

		else:


			opt_reg_results = [train_opt_reg(dataset, baseline_model, num_batches, batch_size, 
    			num_opt_steps, opt_batch_size,
    			representation_layer_sizes = representation_layer_sizes, threshold = .5, reg = reg,
    			verbose = True,
    			restart_model_full_minimization = restart_model_full_minimization, burn_in = burn_in) for _ in range(num_experiments)]

		opt_reg_results_list.append(("opt_reg-{}".format(reg), opt_reg_results))
	
	return opt_reg_results_list




def run_modsel_opt_reg_experiments(dataset, regs, modselalgo, num_experiments, baseline_model, num_batches, 
	batch_size, num_opt_steps = 1000, 
	opt_batch_size = 20, representation_layer_sizes = [10, 10], restart_model_full_minimization = False, 
	split = False, burn_in = 0):
	

	# IPython.embed()
	# raise ValueError("asdflkm")

	if USE_RAY:
		opt_reg_modsel_results = [train_opt_reg_modsel_remote.remote(dataset, baseline_model, num_batches, batch_size, 
   			num_opt_steps, opt_batch_size,
    		representation_layer_sizes = representation_layer_sizes, threshold = .5, regs = regs,
    		verbose = True,
    		restart_model_full_minimization = restart_model_full_minimization, modselalgo = modselalgo, 
    		split = split, burn_in = burn_in) for _ in range(num_experiments)]
		opt_reg_modsel_results = ray.get(opt_reg_modsel_results)

	else:
		opt_reg_modsel_results = [train_opt_reg_modsel(dataset, baseline_model, num_batches, batch_size, 
   			num_opt_steps, opt_batch_size,
    		representation_layer_sizes = representation_layer_sizes, threshold = .5, regs = regs,
    		verbose = True,
    		restart_model_full_minimization = restart_model_full_minimization, modselalgo = modselalgo, 
    		split = split, burn_in = burn_in) for _ in range(num_experiments)]
		
	return ("opt_reg split{} {}".format(split, modselalgo),opt_reg_modsel_results )#, mahalanobis_results_list



def plot_modsel_probabilities(algo_name, dataset, num_batches, batch_size, modselalgo, 
	results_dictionary, hyperparams, colors, representation_layer_sizes, averaging_window = 1,
	split = False):

	Ts = np.arange(num_batches)+1
	color_index = 0


	logging_dir = "./ModselResults/T{}".format(num_batches)
	if not os.path.exists(logging_dir):
		os.mkdir(logging_dir)


	logging_dir = "./ModselResults/T{}/{}".format(num_batches, dataset)
	if not os.path.exists(logging_dir):
		os.mkdir(logging_dir)




	modsel_results = results_dictionary["{} split{} {}".format(algo_name, split, modselalgo)]


	probs =np.array([x["modselect_info"] for x in modsel_results])

	#color_index = 1

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

	if not split:
		filename = "{}/modsel_probabilities-{}_{}_{}_T{}_B{}_N_{}.png".format(logging_dir,modselalgo,algo_name, dataset,num_batches,batch_size, repres_layers_name)
	else:
		filename = "{}/modsel_probabilities-split-{}_{}_{}_T{}_B{}_N_{}.png".format(logging_dir,modselalgo,algo_name, dataset,num_batches,batch_size, repres_layers_name)

	plt.savefig(filename)
	plt.close("all")
	



def plot_optimism_pessimism(algo_name,dataset, num_batches, batch_size, results_dictionary, 
	hyperparams, colors, representation_layer_sizes, averaging_window = 1):

	Ts = (np.arange(num_batches/averaging_window)+1)*averaging_window
	color_index = 0

	#results_type = "rewards"

	logging_dir = "./ModselResults/T{}".format(num_batches)
	if not os.path.exists(logging_dir):
		os.mkdir(logging_dir)


	logging_dir = "./ModselResults/T{}/{}".format(num_batches, dataset)
	if not os.path.exists(logging_dir):
		os.mkdir(logging_dir)



	for hyperparam in hyperparams:

		hyperparam_results = results_dictionary["{}-{}".format(algo_name, hyperparam)] 
		hyperparam_rewards = np.array([np.cumsum(x["rewards"]) for x in hyperparam_results])
		
		hyperparam_opt_rewards = np.array([np.cumsum(x["optimistic_reward_predictions"]) for x in hyperparam_results])

		hyperparam_pess_rewards = np.array([np.cumsum(x["pessimistic_reward_predictions"]) for x in hyperparam_results])





		hyperparam_rewards_mean = np.mean(hyperparam_rewards,0)
		#hyperparam_rewards_std = np.std(hyperparam_rewards,0)
		hyperparam_opt_rewards_mean = np.mean(hyperparam_opt_rewards, 0)
		hyperparam_pess_rewards_mean = np.mean(hyperparam_pess_rewards, 0)

		#IPython.embed()



		hyperparam_rewards_mean = np.mean(hyperparam_rewards_mean.reshape(int(num_batches/averaging_window), averaging_window), 1)
		#hyperparam_rewards_std = np.mean(hyperparam_rewards_std.reshape(int(num_batches/averaging_window), averaging_window), 1)

		hyperparam_opt_rewards_mean = np.mean(hyperparam_opt_rewards_mean.reshape(int(num_batches/averaging_window), averaging_window), 1)
		hyperparam_pess_rewards_mean = np.mean(hyperparam_pess_rewards_mean.reshape(int(num_batches/averaging_window), averaging_window), 1)





		#IPython.embed()
		


		plt.plot(Ts, hyperparam_rewards_mean, color = colors[color_index] ,  label = "{}-{}".format(algo_name,hyperparam))
		plt.plot(Ts, hyperparam_opt_rewards_mean, color = colors[color_index], linestyle = "dashed")
		plt.plot(Ts, hyperparam_pess_rewards_mean, color = colors[color_index], linestyle = "dotted")


		plt.fill_between(Ts, hyperparam_pess_rewards_mean, 
			hyperparam_opt_rewards_mean, color = colors[color_index], alpha = .2)


		color_index += 1


	label = "Cum rewards"
	

	repres_layers_name = get_architecture_name(representation_layer_sizes)


	plt.title("{} {} B{} N {}".format( label, dataset, batch_size, repres_layers_name))
	plt.xlabel("Number of batches")

	plt.ylabel(label)
	# plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
	plt.legend(fontsize=8, loc="upper left")

	filename = "{}/opt_pess_{}_{}_T{}_B{}_N_{}.png".format(logging_dir, algo_name,dataset, 
		num_batches, batch_size, repres_layers_name)

	plt.savefig(filename)
	plt.close("all")






def plot_results(algo_name, dataset, results_type, num_batches, batch_size, modselalgo, 
	results_dictionary, hyperparams, colors, representation_layer_sizes, cummulative_plot = False, 
	averaging_window = 1 , split=False):


	Ts = (np.arange(num_batches/averaging_window)+1)*averaging_window
	color_index = 0


	if results_type != "instantaneous_regrets" and cummulative_plot == True:
		raise ValueError("Results type {} does not support cummulative plot".format(results_type))



	logging_dir = "./ModselResults/T{}".format(num_batches)
	if not os.path.exists(logging_dir):
		os.mkdir(logging_dir)


	logging_dir = "./ModselResults/T{}/{}".format(num_batches, dataset)
	if not os.path.exists(logging_dir):
		os.mkdir(logging_dir)


	
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



		plt.plot(Ts, hyperparam_results_mean, color = colors[color_index] ,  label = "{}-{}".format(algo_name,hyperparam))
		plt.fill_between(Ts, hyperparam_results_mean-.5*hyperparam_results_std, 
			hyperparam_results_mean+.5*hyperparam_results_std, color = colors[color_index], alpha = .2)


		color_index += 1

	##### PLOTTING modsel results.
	modsel_results = results_dictionary["{} split{} {}".format(algo_name, split, modselalgo)]



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

	#color_index += 1




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

	if not split:
		filename = "{}/{}_cum_{}-{}_{}_{}_T{}_B{}_N_{}.png".format(logging_dir,results_type, cummulative_plot, 
			algo_name, modselalgo,dataset, num_batches, batch_size, repres_layers_name)

	else:

		filename = "{}/{}-split_cum_{}-{}_{}_{}_T{}_B{}_N_{}.png".format(logging_dir,results_type, cummulative_plot, 
			algo_name, modselalgo,dataset, num_batches, batch_size, repres_layers_name)

	plt.savefig(filename)
	plt.close("all")




def plot_contrast_modsel_results(algo_name, dataset, results_type, num_batches, batch_size, modsel_keys, 
	results_dictionary, colors, representation_layer_sizes, cummulative_plot = False, 
	averaging_window = 1 , split=False):


	Ts = (np.arange(num_batches/averaging_window)+1)*averaging_window
	color_index = 0


	if results_type != "instantaneous_regrets" and cummulative_plot == True:
		raise ValueError("Results type {} does not support cummulative plot".format(results_type))

	logging_dir = "./ModselResults/T{}".format(num_batches)
	if not os.path.exists(logging_dir):
		os.mkdir(logging_dir)


	logging_dir = "./ModselResults/T{}/{}".format(num_batches, dataset)
	if not os.path.exists(logging_dir):
		os.mkdir(logging_dir)


	for modsel_key in modsel_keys:

		##### PLOTTING modsel results.
		modsel_results = results_dictionary[modsel_key]



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

	if not split:
		filename = "{}/combined_{}_cum_{}-{}_{}_T{}_B{}_N_{}.png".format(logging_dir,results_type, cummulative_plot, 
			algo_name,dataset, num_batches, batch_size, repres_layers_name)

	else:

		filename = "{}/combined_{}-split_cum_{}-{}_{}_T{}_B{}_N_{}.png".format(logging_dir,results_type, cummulative_plot, 
			algo_name,dataset, num_batches, batch_size, repres_layers_name)

	plt.savefig(filename)
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



if __name__ == "__main__":


	

	algo_name = sys.argv[1]
	if algo_name not in ["mahalanobis", "epsilon", "opt_reg"]:
		raise ValueError("Algorithm name not in allowed algorithms {}".format(algo_name))

	num_batches = int(sys.argv[2])

	num_experiments = int(sys.argv[3])

	USE_RAY = sys.argv[4] == "True" #False
	if sys.argv[4] not in ["True", "False"]:
		raise ValueError("USE_RAY key not in [True, False]")
	# IPython.embed()
	# raise ValueError("asdlfkm")


	averaging_window = 1
	epsilon = .1
	alpha = 10
	epsilons = [.2, .1, .01, .05]
	alphas = [.05, .5, 1, 4]#[.000001, 1/4.0, 1/2.0, 1, 2, 4, 8 ]
	opt_regs = [.08, .16, 1, 10]
	decaying_epsilon = False

	num_opt_steps = 2000
	num_baseline_steps = 20000
	opt_batch_size = 20
	burn_in = 10

	#split = False
	restart_model_full_minimization = True

	batch_size = 10

	representation_layer_sizes = [100,10]



	modselalgos = [ 'BalancingDoResurrectClassic','BalancingDoResurrectDown', "BalancingDoResurrect", "BalancingDoubling", "Corral", "UCB", "EXP3"]
	datasets = ["Adult"]#, "German", "Bank", "Crime", "Adult-10_10", "German-10_10","Crime-10_10","Bank-10_10", ]# ["Adult-10_10", "German-10_10","Crime-10_10","Bank-10_10", "Adult", "German", "Bank", "Crime"]#, "German", "Bank", "Adult"]"Adult-10-10"]#,


	colors = plt.cm.viridis(range(len(modselalgos) + len(alphas)))

	#matplotlib.colors.TABLEAU_COLORS.keys()



	repres_layers_name = get_architecture_name(representation_layer_sizes)

	### 

	results_dictionary = dict([])



	for dataset in datasets:
		#IPython.embed()
		# if RUN_EPSILON or RUN_MAHALANOBIS or RUN_OPT_REG:

		baseline_results, baseline_model = run_train_baseline(dataset, num_experiments, 
				representation_layer_sizes = representation_layer_sizes, num_timesteps = num_baseline_steps)

		results_dictionary["baseline"] = [x[1] for x in baseline_results]



		if algo_name == "mahalanobis":


			mahalanobis_results_list = run_base_mahalanobis_experiments(dataset, alphas, num_experiments, baseline_model, num_batches, 
				batch_size, num_opt_steps = num_opt_steps, 
				opt_batch_size = 20, representation_layer_sizes = representation_layer_sizes, 
				restart_model_full_minimization = restart_model_full_minimization)

			for mahalanobis_res_tuple in mahalanobis_results_list:
				results_dictionary[mahalanobis_res_tuple[0]] = mahalanobis_res_tuple[1]	



		if algo_name == "opt_reg":


			opt_reg_results_list = run_base_opt_reg_experiments(dataset, opt_regs, num_experiments, baseline_model, num_batches, 
				batch_size, num_opt_steps = num_opt_steps, 
				opt_batch_size = 20, representation_layer_sizes = representation_layer_sizes, 
				restart_model_full_minimization = restart_model_full_minimization, burn_in = burn_in)

			for opt_reg_res_tuple in opt_reg_results_list:
				results_dictionary[opt_reg_res_tuple[0]] = opt_reg_res_tuple[1]	


		#IPython.embed()

		#results_dictionary_split = deepcopy(results_dictionary)
		
		for split in [True, False]:
			modsel_keys = []
			for modselalgo in modselalgos:

				### Run epsilon-greedy model selection experiments
				if algo_name == "epsilon":

					epsilon_greedy_modsel_results_tuple, epsilon_greedy_results_list  =	run_epsilon_greedy_experiments(dataset, epsilons, modselalgo, num_experiments, baseline_model, num_batches, batch_size, decaying_epsilon, num_opt_steps = 1000, 
						opt_batch_size = opt_batch_size, representation_layer_sizes = representation_layer_sizes, 
						restart_model_full_minimization = restart_model_full_minimization, split = split)

					results_dictionary[epsilon_greedy_modsel_results_tuple[0]] = epsilon_greedy_modsel_results_tuple[1]

					for eps_res_tuple in epsilon_greedy_results_list:
						#if split:
							results_dictionary[eps_res_tuple[0]] = eps_res_tuple[1]
						#else:
						#	results_dictionary[eps_res_tuple[0]] = eps_res_tuple[1]

					modsel_keys.append(eps_res_tuple[0])
				if algo_name == "mahalanobis":

					mahalanobis_modsel_results_tuple = run_modsel_mahalanobis_experiments(dataset, alphas, modselalgo, num_experiments, baseline_model, num_batches, 
											batch_size, num_opt_steps = num_opt_steps, 
											opt_batch_size = opt_batch_size, representation_layer_sizes = representation_layer_sizes, 
											restart_model_full_minimization = restart_model_full_minimization, split = split)


					results_dictionary[mahalanobis_modsel_results_tuple[0]] = mahalanobis_modsel_results_tuple[1]


					modsel_keys.append(mahalanobis_modsel_results_tuple[0])
				
				if algo_name == "opt_reg":

					opt_reg_modsel_results_tuple = run_modsel_opt_reg_experiments(dataset, opt_regs, modselalgo, num_experiments, baseline_model, num_batches, 
											batch_size, num_opt_steps = num_opt_steps, 
											opt_batch_size = opt_batch_size, representation_layer_sizes = representation_layer_sizes, 
											restart_model_full_minimization = restart_model_full_minimization, split = split, burn_in = burn_in)


					results_dictionary[opt_reg_modsel_results_tuple[0]] = opt_reg_modsel_results_tuple[1]

					modsel_keys.append(opt_reg_modsel_results_tuple[0])

				pickle_results_filename_stub = get_pickle_filename_stub(dataset, num_batches,batch_size,repres_layers_name, split)


				if algo_name == "epsilon" or algo_name == "mahalanobis" or algo_name == "opt_reg":
				
					pickle_and_zip(results_dictionary, "ModselResults/{}".format(pickle_results_filename_stub))
					


				#results_dictionary = pickle.load(open("ModselResults/{}".format(pickle_results_filename), "rb")) 





				Ts = np.arange(num_batches)+1
				color_index = 0


				if algo_name == "epsilon":
					algo_type_key = "epsilon"
					hyperparams = epsilons

				if algo_name == "mahalanobis":
					algo_type_key = "alpha"
					hyperparams = alphas

				if algo_name == "opt_reg":
					algo_type_key = "opt_reg"
					hyperparams = opt_regs

				plot_modsel_probabilities(algo_type_key, dataset, num_batches, batch_size, modselalgo, 
						results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes, split = split)

				plot_results(algo_type_key, dataset, "instantaneous_regrets", num_batches, batch_size, modselalgo, 
						results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
						 cummulative_plot = True , averaging_window = averaging_window, split = split)

				plot_results(algo_type_key, dataset, "instantaneous_accuracies", num_batches, batch_size, modselalgo, 
						results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
						cummulative_plot = False, averaging_window = averaging_window, split = split)

				if PLOT_ALL_STATS:
					plot_results(algo_type_key, dataset, "num_negatives", num_batches, batch_size, modselalgo, 
							results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
							 cummulative_plot = False, averaging_window = averaging_window, split = split)

					plot_results(algo_type_key, dataset, "num_positives", num_batches, batch_size, modselalgo, 
							results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
							 cummulative_plot = False, averaging_window = averaging_window, split = split)

					plot_results(algo_type_key, dataset, "false_neg_rates", num_batches, batch_size, modselalgo, 
							results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
							 cummulative_plot = False, averaging_window = averaging_window, split = split)

					plot_results(algo_type_key, dataset, "false_positive_rates", num_batches, batch_size, modselalgo, 
							results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
							cummulative_plot = False, averaging_window = averaging_window, split = split)


				plot_optimism_pessimism(algo_type_key, dataset, num_batches, batch_size, results_dictionary, 
					hyperparams, colors, representation_layer_sizes, averaging_window = averaging_window)

			IPython.embed()

			# ## plot all the model selection results together
			plot_contrast_modsel_results(algo_type_key, dataset, "instantaneous_regrets", num_batches, batch_size, modsel_keys, 
				results_dictionary, colors, representation_layer_sizes, cummulative_plot = True, 
				averaging_window = averaging_window , split=split)




import numpy as np
import matplotlib.pyplot as plt
import ray
import pickle
import sys
import IPython
import os
from copy import deepcopy
import matplotlib.colors

from utilities import pickle_and_zip, unzip_and_load_pickle
from algorithmsmodsel import train_epsilon_greedy_modsel, train_mahalanobis_modsel, train_opt_reg_modsel
from algorithms import train_epsilon_greedy, train_mahalanobis, train_baseline, train_opt_reg
from algorithms_remote import train_epsilon_greedy_remote, train_epsilon_greedy_modsel_remote, train_baseline_remote, train_mahalanobis_remote, train_mahalanobis_modsel_remote, train_opt_reg_modsel_remote, train_opt_reg_remote

from modsel_plot_tools import plot_modsel_probabilities, plot_optimism_pessimism, get_results_label, plot_base, plot_results, plot_contrast_modsel_results, get_architecture_name, plot_all


def process_results(results_list):
    mean = np.mean(results_list, 0)
    standard_dev = np.std(results_list, 0)

 
    return mean, standard_dev


def get_pickle_modsel_filename_stub(experiment_name, dataset, algo_name, modselalgo, num_batches,batch_size,repres_layers_name, split):
	if not split:
		filename_stub = "{}_results_{}_{}_{}_T{}_B{}_N_{}".format(experiment_name, dataset, algo_name, modselalgo, num_batches,batch_size,repres_layers_name )

	else:
		filename_stub = "{}_results-split_{}_{}_{}_T{}_B{}_N_{}".format(experiment_name, dataset, algo_name, modselalgo, num_batches,batch_size,repres_layers_name )

	return filename_stub

def get_pickle_base_filename_stub(experiment_name, dataset, algo_name, num_batches,batch_size,repres_layers_name):
	
	filename_stub = "{}_results_bases_{}_{}_T{}_B{}_N_{}".format(experiment_name, dataset, algo_name, num_batches,batch_size,repres_layers_name )


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
	opt_batch_size = 20, representation_layer_sizes = [10, 10], restart_model_full_minimization = False, split = False, retraining_frequency = 1, burn_in = -1):

	if USE_RAY:
		mahalanobis_modsel_results = [train_mahalanobis_modsel_remote.remote(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, 
	    representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True, alphas = alphas,
	    restart_model_full_minimization = restart_model_full_minimization, modselalgo = modselalgo,
	    split = split, retraining_frequency = retraining_frequency, burn_in = burn_in) for _ in range(num_experiments)]
		mahalanobis_modsel_results = ray.get(mahalanobis_modsel_results)

	else:
		mahalanobis_modsel_results = [train_mahalanobis_modsel(dataset, baseline_model, 
	    num_batches = num_batches, batch_size = batch_size, 
	    num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, 
	    representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True, alphas = alphas,
	    restart_model_full_minimization = restart_model_full_minimization, modselalgo = modselalgo,
	    split = split, retraining_frequency = retraining_frequency, burn_in = burn_in) for _ in range(num_experiments)]
		
	return ("alpha split{} {}".format(split, modselalgo),mahalanobis_modsel_results )#, mahalanobis_results_list



def run_base_mahalanobis_experiments(dataset, alphas, num_experiments, baseline_model, num_batches, 
	batch_size, num_opt_steps = 1000, 
	opt_batch_size = 20, representation_layer_sizes = [10, 10], restart_model_full_minimization = False, retraining_frequency = 1, burn_in = -1):

	mahalanobis_results_list = []

	reduced_alpha_sequence = list(set(alphas))

	for alpha in reduced_alpha_sequence:

		if USE_RAY:


			mahalanobis_results = [ train_mahalanobis_remote.remote(dataset, baseline_model, 
	   				 num_batches = num_batches, batch_size = batch_size, 
	   				 num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, 
	   				 representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True,  alpha = alpha, 
	   				 lambda_reg = 1, restart_model_full_minimization = False, retraining_frequency = retraining_frequency, burn_in = burn_in) for _ in range(num_experiments)]

			mahalanobis_results = ray.get(mahalanobis_results)

		else:

			#IPython.embed()
			mahalanobis_results = [train_mahalanobis(dataset, baseline_model, 
	   				 num_batches = num_batches, batch_size = batch_size, 
	   				 num_opt_steps = num_opt_steps, opt_batch_size = opt_batch_size, 
	   				 representation_layer_sizes = representation_layer_sizes, threshold = .5, verbose = True,  alpha = alpha, 
	   				 lambda_reg = 1, restart_model_full_minimization = False, retraining_frequency = retraining_frequency, burn_in = burn_in) for _ in range(num_experiments)]

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





if __name__ == "__main__":


	
	### Algorithm name
	algo_name = sys.argv[1]
	if algo_name not in ["mahalanobis", "epsilon", "opt_reg"]:
		raise ValueError("Algorithm name not in allowed algorithms {}".format(algo_name))

	### Number of batches
	num_batches = int(sys.argv[2])

	### Number of experiments
	num_experiments = int(sys.argv[3])


	### List of datasets
	datasets = str(sys.argv[4]).split(",")

	### Neural architecture
	representation_layer_sizes = str(sys.argv[5]).split("_")
	representation_layer_sizes = [int(a) for a in representation_layer_sizes]


	### USE Ray
	USE_RAY = sys.argv[6] == "True" #False
	if sys.argv[6] not in ["True", "False"]:
		raise ValueError("USE_RAY key not in [True, False]")


	### Run exps
	RUN_EXPS = sys.argv[7] == "True" #False
	if sys.argv[7] not in ["True", "False"]:
		raise ValueError("RUN_EXPS key not in [True, False]")

	experiment_name = sys.argv[8]


	#IPython.embed()

	### Import GLOBAL parameters
	from parameters_uci import *
	#IPython.embed()

	### set experiment specific parameters
	experiment_specific_params = experiment_parameter_map[experiment_name]
	alphas = experiment_specific_params["alphas"]
	epsilons = experiment_specific_params["epsilons"]
	opt_regs = experiment_specific_params["opt_regs"]


	#IPython.embed()

	## TODO: this should ideally be larger than max(num_experiments, len(hyperparams))
	colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
	#IPython.embed()


	repres_layers_name = get_architecture_name(representation_layer_sizes)
	results_dictionary = dict([])

	path = os.getcwd()
	base_data_dir = "{}/ModselResults".format(path)


	if algo_name == "epsilon":
		algo_type_key = "epsilon"
		hyperparams = epsilons

	if algo_name == "mahalanobis":
		algo_type_key = "alpha"
		hyperparams = alphas

	if algo_name == "opt_reg":
		algo_type_key = "opt_reg"
		hyperparams = opt_regs


	for dataset in datasets:

		### Get data file name
		base_algorithms_file_name = get_pickle_base_filename_stub(experiment_name, dataset, algo_name, num_batches,batch_size,repres_layers_name)


		if RUN_EXPS:

			baseline_results, baseline_model = run_train_baseline(dataset, num_experiments, 
					representation_layer_sizes = representation_layer_sizes, num_timesteps = num_baseline_steps)

			results_dictionary["baseline"] = [x[1] for x in baseline_results]

			if algo_name == "mahalanobis":

				mahalanobis_results_list = run_base_mahalanobis_experiments(dataset, alphas, num_experiments, 
					baseline_model, 
					num_batches, 
					batch_size, 
					num_opt_steps = num_opt_steps, 
					opt_batch_size = 20, 
					representation_layer_sizes = representation_layer_sizes, 
					restart_model_full_minimization = restart_model_full_minimization,
					retraining_frequency = retraining_frequency, 
					burn_in = burn_in)

				for mahalanobis_res_tuple in mahalanobis_results_list:
					results_dictionary[mahalanobis_res_tuple[0]] = mahalanobis_res_tuple[1]	

			if algo_name == "opt_reg":

				opt_reg_results_list = run_base_opt_reg_experiments(dataset, opt_regs, num_experiments, baseline_model, num_batches, 
					batch_size, num_opt_steps = num_opt_steps, 
					opt_batch_size = 20, representation_layer_sizes = representation_layer_sizes, 
					restart_model_full_minimization = restart_model_full_minimization, burn_in = burn_in)

				for opt_reg_res_tuple in opt_reg_results_list:
					results_dictionary[opt_reg_res_tuple[0]] = opt_reg_res_tuple[1]	

			#### Save the base algorithms data
			pickle_and_zip(results_dictionary, base_algorithms_file_name, base_data_dir, is_zip_file = True)


		else: 

			results_dictionary = {**results_dictionary, **unzip_and_load_pickle(base_data_dir, base_algorithms_file_name, is_zip_file = True)}


		for hyperparam in hyperparams:
			plot_base(algo_type_key, dataset, "instantaneous_regrets", num_batches, batch_size, results_dictionary, 
				hyperparam, averaging_window, representation_layer_sizes, cummulative_plot = True)


		for split in [True, False]:
			modsel_keys = []
			for modselalgo in modselalgos:
				
				pickle_modsel_results_filename_stub = get_pickle_modsel_filename_stub(experiment_name,dataset, algo_name, modselalgo, num_batches,batch_size,repres_layers_name, split)

				if RUN_EXPS:

					### Run epsilon-greedy model selection experiments
					if algo_name == "epsilon":

						raise ValueError("Not properly implemented")

						epsilon_greedy_modsel_results_tuple, epsilon_greedy_results_list  =	run_epsilon_greedy_experiments(dataset, epsilons, modselalgo, num_experiments, baseline_model, num_batches, batch_size, decaying_epsilon, num_opt_steps = 1000, 
							opt_batch_size = opt_batch_size, representation_layer_sizes = representation_layer_sizes, 
							restart_model_full_minimization = restart_model_full_minimization, split = split)

						results_dictionary[epsilon_greedy_modsel_results_tuple[0]] = epsilon_greedy_modsel_results_tuple[1]

						for eps_res_tuple in epsilon_greedy_results_list:						
								results_dictionary[eps_res_tuple[0]] = eps_res_tuple[1]

						modsel_keys.append(eps_res_tuple[0])

						


					if algo_name == "mahalanobis":

						mahalanobis_modsel_results_tuple = run_modsel_mahalanobis_experiments(dataset, alphas, modselalgo, num_experiments, baseline_model, num_batches, 
												batch_size, num_opt_steps = num_opt_steps, 
												opt_batch_size = opt_batch_size, representation_layer_sizes = representation_layer_sizes, 
												restart_model_full_minimization = restart_model_full_minimization, split = split, retraining_frequency = retraining_frequency)

						results_dictionary[mahalanobis_modsel_results_tuple[0]] = mahalanobis_modsel_results_tuple[1]


						modsel_keys.append(mahalanobis_modsel_results_tuple[0])

						modsel_result = (mahalanobis_modsel_results_tuple[0], mahalanobis_modsel_results_tuple[1])

					
					if algo_name == "opt_reg":

						opt_reg_modsel_results_tuple = run_modsel_opt_reg_experiments(dataset, opt_regs, modselalgo, num_experiments, baseline_model, num_batches, 
												batch_size, num_opt_steps = num_opt_steps, 
												opt_batch_size = opt_batch_size, representation_layer_sizes = representation_layer_sizes, 
												restart_model_full_minimization = restart_model_full_minimization, split = split, burn_in = burn_in)

						results_dictionary[opt_reg_modsel_results_tuple[0]] = opt_reg_modsel_results_tuple[1]
						modsel_keys.append(opt_reg_modsel_results_tuple[0])
						modsel_result = (opt_reg_modsel_results_tuple[0], opt_reg_modsel_results_tuple[1])
					

					#pickle_and_zip(modsel_result, "./ModselResults/{}".format(pickle_modsel_results_filename_stub))
					pickle_and_zip(modsel_result, pickle_modsel_results_filename_stub, base_data_dir, is_zip_file = True)

				else:

					modsel_result = unzip_and_load_pickle(base_data_dir, pickle_modsel_results_filename_stub, is_zip_file = True)
					modsel_keys.append(modsel_result[0])

					results_dictionary[modsel_result[0]] = modsel_result[1]



				plot_all(experiment_name,dataset, results_dictionary, num_batches, batch_size, split, hyperparams,
					algo_type_key, modselalgo, colors, representation_layer_sizes, 
					averaging_window, modsel_keys)



			plot_contrast_modsel_results(experiment_name,algo_type_key, dataset, "instantaneous_regrets", num_batches, batch_size, modselalgos, modsel_keys, 
				results_dictionary, colors, representation_layer_sizes, cummulative_plot = True, 
				averaging_window = averaging_window , split=split, sqrt_scaled = True)




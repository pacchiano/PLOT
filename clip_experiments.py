import numpy as np
import matplotlib.pyplot as plt
import ray
import pickle
import sys
import IPython


from algorithmsmodsel import train_epsilon_greedy_modsel, train_mahalanobis_modsel, train_opt_reg_modsel
from algorithms import train_epsilon_greedy, train_mahalanobis, train_baseline, train_opt_reg
from algorithms_remote import train_epsilon_greedy_remote, train_epsilon_greedy_modsel_remote, train_baseline_remote, train_mahalanobis_remote, train_mahalanobis_modsel_remote, train_opt_reg_modsel_remote, train_opt_reg_remote


from datasets import get_dataset
USE_RAY = True

def run_train_baseline(dataset, num_experiments, batch_size = 32, num_timesteps = 10000, 
	representation_layer_sizes = [10,10]):
	### BASELINE training
	if USE_RAY:
		baseline_results = [train_baseline_remote.remote(dataset, num_timesteps = num_timesteps, 
	    batch_size = batch_size, 
	    representation_layer_sizes = representation_layer_sizes, mode = "regression") for _ in range(num_experiments)]
		baseline_results = ray.get(baseline_results)
	else:
		baseline_results = [train_baseline(dataset, num_timesteps = num_timesteps, 
	    batch_size = batch_size, 
	    representation_layer_sizes = representation_layer_sizes, mode = "regression") for _ in range(num_experiments)]

	baseline_model = baseline_results[0][1]	

	return baseline_results, baseline_model



def run_opt_reg_experiments(dataset, regs, num_experiments, baseline_model, num_batches, 
	batch_size, num_opt_steps = 1000, 
	opt_batch_size = 20, representation_layer_sizes = [10, 10], 
	restart_model_full_minimization = False, threshold = .5, burn_in = 0, 
	model_selection_algos = ["Corral"], split = False):

	opt_reg_results_list = []

	for reg in regs:

		if USE_RAY:

			opt_reg_results = [train_opt_reg_remote.remote(dataset, baseline_model, num_batches, batch_size, 
    			num_opt_steps, opt_batch_size,
    			representation_layer_sizes = representation_layer_sizes, threshold = threshold, reg = reg,
    			verbose = True,
    			restart_model_full_minimization = restart_model_full_minimization, burn_in = burn_in) for _ in range(num_experiments)]



			opt_reg_results = ray.get(opt_reg_results)

		else:


			opt_reg_results = [train_opt_reg(dataset, baseline_model, num_batches, batch_size, 
    			num_opt_steps, opt_batch_size,
    			representation_layer_sizes = representation_layer_sizes, threshold = threshold, reg = reg,
    			verbose = True,
    			restart_model_full_minimization = restart_model_full_minimization, burn_in = burn_in) for _ in range(num_experiments)]

		opt_reg_results_list.append(("opt_reg-{}".format(reg), opt_reg_results))
	
	##### Model Selection Run
	for modsel_algo in model_selection_algos:

		if USE_RAY:


			opt_reg_results = [train_opt_reg_modsel_remote.remote(dataset, baseline_model, num_batches, batch_size, 
    				num_opt_steps, opt_batch_size,
    				representation_layer_sizes = representation_layer_sizes, threshold = threshold, regs = regs,
    				verbose = True,
    				restart_model_full_minimization = restart_model_full_minimization, modselalgo = modsel_algo, 
    				split = split, burn_in  = burn_in) for _ in range(num_experiments)]


			# opt_reg_results = [train_opt_reg_remote.remote(dataset, baseline_model, num_batches, batch_size, 
    		# 	num_opt_steps, opt_batch_size,
    		# 	representation_layer_sizes = representation_layer_sizes, threshold = threshold, reg = reg,
    		# 	verbose = True,
    		# 	restart_model_full_minimization = restart_model_full_minimization, burn_in = burn_in) for _ in range(num_experiments)]



			opt_reg_results = ray.get(opt_reg_results)

		else:
			opt_reg_results = [ train_opt_reg_modsel(dataset, baseline_model, num_batches, batch_size, 
    				num_opt_steps, opt_batch_size,
    				representation_layer_sizes = representation_layer_sizes, threshold = threshold, regs = regs,
    				verbose = True,
    				restart_model_full_minimization = restart_model_full_minimization, modselalgo = modsel_algo, 
    				split = split, burn_in  = burn_in) for _ in range(num_experiments)]


			# opt_reg_results = [train_opt_reg(dataset, baseline_model, num_batches, batch_size, 
    		# 	num_opt_steps, opt_batch_size,
    		# 	representation_layer_sizes = representation_layer_sizes, threshold = threshold, reg = reg,
    		# 	verbose = True,
    		# 	restart_model_full_minimization = restart_model_full_minimization, burn_in = burn_in) for _ in range(num_experiments)]

		opt_reg_results_list.append(("opt_reg-{}".format(modsel_algo), opt_reg_results))
	

	return opt_reg_results_list


if __name__ == "__main__":
	threshold = float(sys.argv[1])
	num_batches = int(sys.argv[2])
	split = sys.argv[3] == "True"
	if sys.argv[3] not in ["False", "True"]:
		raise ValueError("Third split argument not in [False, True]")

	num_experiments = 10
	regs=  [0, .01, .03, .05]
	colors = ["red", "blue", "green", "black", "violet", "orange", "yellow", "gray"]
	#num_batches = 1000
	baseline_batches = 10000
	baseline_batch_size = 32
	batch_size = 1
	num_opt_steps = 1000
	#threshold = .7
	model_selection_algos = ["BalancingDoubling"]#"Corral", "UCB", "EXP3", "BalancingSharp"]
	burn_in = 10
	if len(model_selection_algos) + len(regs) > len(colors):
		raise ValueError("Num experiment params is > num colors")

	baseline_results, baseline_model = run_train_baseline("CLIP", num_experiments, batch_size = baseline_batch_size, 
		num_timesteps = baseline_batches, representation_layer_sizes = [10,10])


	opt_reg_results_list = run_opt_reg_experiments("CLIP", regs, num_experiments, baseline_model, num_batches, 
	batch_size, num_opt_steps = num_opt_steps, 
	opt_batch_size = 20, representation_layer_sizes = [10, 10], 
	restart_model_full_minimization = False, threshold = threshold, burn_in = burn_in, 
	model_selection_algos = model_selection_algos, split = split)

	### Plot regrets
	for j in range(len(opt_reg_results_list)):

		results = np.zeros((num_experiments, num_batches))
		for i in range(num_experiments):
			results[i,:]= np.cumsum(opt_reg_results_list[j][1][i]['instantaneous_regrets'])
			
			label = opt_reg_results_list[j][0]
			#IPython.embed()
		plt.plot(np.arange(num_batches)+1, results.mean(0), color= colors[j], label = label)
		plt.fill_between(np.arange(num_batches)+1, results.mean(0) - .5*results.std(0),
			results.mean(0) + .5*results.std(0),  color= colors[j], alpha = .2)
	plt.title("Regrets")
	plt.xlabel("timesteps")
	plt.ylabel("regret")		

	plt.legend(fontsize=8, loc="upper left")	
	plt.savefig("./CLIP/regrets_{}_T{}_B{}_split_{}.png".format(threshold, num_batches, batch_size, split))
	plt.close("all")

	### Plot cum num positives
	for j in range(len(opt_reg_results_list)):

		results = np.zeros((num_experiments, num_batches))
		for i in range(num_experiments):
			results[i,:]= np.cumsum(opt_reg_results_list[j][1][i]['num_positives'])
			label = opt_reg_results_list[j][0]
		#IPython.embed()
		# run_opt_reg_experiments("CLIP", [0,1], 1)
		plt.plot(np.arange(num_batches)+1, results.mean(0), color= colors[j], label = label)
		plt.fill_between(np.arange(num_batches)+1, results.mean(0) - .5*results.std(0),
			results.mean(0) + .5*results.std(0),  color= colors[j], alpha = .2)

	plt.title("Num Positives")
	plt.xlabel("timesteps")
	plt.ylabel("num positives")
	plt.legend(fontsize=8, loc="upper left")	
	plt.savefig("./CLIP/num_positives_{}_T{}_B{}_split_{}.png".format(threshold, num_batches, batch_size, split))
	plt.close("all")



	##


	pickle.dump(opt_reg_results_list, 
				    open("./CLIP/data_{}_T{}_B{}_split_{}.p".format(threshold, num_batches, batch_size, split), "wb"))



	#train_dataset, test_dataset = pickle.load(open("./clip_datasets.p", "rb"))
	#IPython.embed()
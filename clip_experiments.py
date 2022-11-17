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
USE_RAY = False

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
	restart_model_full_minimization = False, threshold = .5, burn_in = 0):

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
	
	return opt_reg_results_list


if __name__ == "__main__":
	threshold = float(sys.argv[1])
	num_batches = int(sys.argv[2])

	num_experiments = 2
	regs=  [0, .1, .3, .5]
	colors = ["red", "blue", "green", "black"]
	#num_batches = 1000
	baseline_batches = 1000
	baseline_batch_size = 32
	batch_size = 32
	num_opt_steps = 1000
	#threshold = .7


	burn_in = 10

	baseline_results, baseline_model = run_train_baseline("CLIP", num_experiments, batch_size = baseline_batch_size, 
		num_timesteps = baseline_batches, representation_layer_sizes = [10,10])


	opt_reg_results_list = run_opt_reg_experiments("CLIP", regs, num_experiments, baseline_model, num_batches, 
	batch_size, num_opt_steps = num_opt_steps, 
	opt_batch_size = 20, representation_layer_sizes = [10, 10], 
	restart_model_full_minimization = False, threshold = threshold, burn_in = burn_in)

	### Plot regrets
	for reg, j in zip(regs, range(len(regs))):

		results = np.zeros((num_experiments, num_batches))
		for i in range(num_experiments):
			results[i,:]= np.cumsum(opt_reg_results_list[j][1][i]['instantaneous_regrets'])
			
		#IPython.embed()
		# run_opt_reg_experiments("CLIP", [0,1], 1)
		plt.plot(np.arange(num_batches)+1, results.mean(0), color= colors[j], label = "opt_reg {}".format(reg))
		plt.fill_between(np.arange(num_batches)+1, results.mean(0) - .5*results.std(0),
			results.mean(0) + .5*results.std(0),  color= colors[j], alpha = .2)
	plt.title("Regrets")
	plt.xlabel("timesteps")
	plt.ylabel("regret")		

	plt.legend(fontsize=8, loc="upper left")	
	plt.savefig("./CLIP/CLIP_regrets_{}.png".format(threshold))
	plt.close("all")

	### Plot cum num positives
	for reg, j in zip(regs, range(len(regs))):

		results = np.zeros((num_experiments, num_batches))
		for i in range(num_experiments):
			results[i,:]= np.cumsum(opt_reg_results_list[j][1][i]['num_positives'])
			
		#IPython.embed()
		# run_opt_reg_experiments("CLIP", [0,1], 1)
		plt.plot(np.arange(num_batches)+1, results.mean(0), color= colors[j], label = "opt_reg {}".format(reg))
		plt.fill_between(np.arange(num_batches)+1, results.mean(0) - .5*results.std(0),
			results.mean(0) + .5*results.std(0),  color= colors[j], alpha = .2)

	plt.title("Num Positives")
	plt.xlabel("timesteps")
	plt.ylabel("num positives")
	plt.legend(fontsize=8, loc="upper left")	
	plt.savefig("./CLIP/CLIP_num_positives_{}.png".format(threshold))
	plt.close("all")



	#train_dataset, test_dataset = pickle.load(open("./clip_datasets.p", "rb"))
	IPython.embed()
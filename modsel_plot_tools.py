from copy import deepcopy
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import IPython
import os

from utilities import pickle_and_zip, unzip_and_load_pickle

PLOT_ALL_STATS = False

def plot_modsel_probabilities(experiment_name, algo_name, dataset, num_batches, batch_size, modselalgo, 
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

	
	plt.xlabel("Number of batches")
	plt.legend(fontsize=8, loc="upper left")

	if not split:
		filename = "{}/{}_modsel_probabilities-{}_{}_{}_T{}_B{}_N_{}.png".format(logging_dir, experiment_name,modselalgo,algo_name, dataset,num_batches,batch_size, repres_layers_name)
		plt.title("Probs {} {} B{} N {}".format(modselalgo, dataset, batch_size, repres_layers_name))
	else:
		filename = "{}/{}_modsel_probabilities-split-{}_{}_{}_T{}_B{}_N_{}.png".format(logging_dir, experiment_name,modselalgo,algo_name, dataset,num_batches,batch_size, repres_layers_name)
		plt.title("Probs split {} {} B{} N {}".format(modselalgo, dataset, batch_size, repres_layers_name))
	plt.savefig(filename)
	plt.close("all")
	



def plot_optimism_pessimism(experiment_name, algo_name,dataset, num_batches, batch_size, results_dictionary, 
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


	reduced_hyperparams = list(set(hyperparams))


	for hyperparam in reduced_hyperparams:

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

	filename = "{}/{}_opt_pess_{}_{}_T{}_B{}_N_{}.png".format(logging_dir, experiment_name,algo_name,dataset, 
		num_batches, batch_size, repres_layers_name)

	plt.savefig(filename)
	plt.close("all")

def get_results_label(results_type, cummulative_plot):
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

	return label

def plot_base(algo_name, dataset, results_type, num_batches, batch_size, 
	results_dictionary, hyperparam, averaging_window, representation_layer_sizes, 
	cummulative_plot = True, sqrt_scaled = True):
	
	Ts = (np.arange(num_batches/averaging_window)+1)*averaging_window

	#IPython.embed()

	if results_type != "instantaneous_regrets" and cummulative_plot == True:
		raise ValueError("Results type {} does not support cummulative plot".format(results_type))


	repres_layers_name = get_architecture_name(representation_layer_sizes)


	color_index = 0
	
	logging_dir = "./ModselResults/T{}".format(num_batches)
	if not os.path.exists(logging_dir):
		os.mkdir(logging_dir)


	logging_dir = "./ModselResults/T{}/{}".format(num_batches, dataset)
	if not os.path.exists(logging_dir):
		os.mkdir(logging_dir)


	if sqrt_scaled and not cummulative_plot:
		raise ValueError("Sqrt scaled on and cummulative plot off. This is not allowed")


	# Plot hyper sweep results

	#for hyperparam in hyperparams:

	hyperparam_results = results_dictionary["{}-{}".format(algo_name, hyperparam)] 
	if cummulative_plot:
		hyperparam_stats = np.array([np.cumsum(x[results_type]) for x in hyperparam_results])

	else:
		hyperparam_stats = np.array([x[results_type] for x in hyperparam_results])


	num_experiments = hyperparam_stats.shape[0]
	#IPython.embed()

	### get mean stuff.
	hyperparam_results_mean = np.mean(hyperparam_stats,0)
	hyperparam_results_std = np.std(hyperparam_stats,0)

	if sqrt_scaled:
		#IPython.embed()
		hyperparam_results_mean *= 1.0/np.sqrt(Ts*np.log(Ts+1))
		hyperparam_results_std *= 1.0/np.sqrt(Ts*np.log(Ts+1))

	plt.plot(Ts, hyperparam_results_mean, color = "black",  label = "mean", linestyle = "dashed", linewidth = 5)
	plt.fill_between(Ts, hyperparam_results_mean-.5*hyperparam_results_std, 
			hyperparam_results_mean+.5*hyperparam_results_std, color = "black", alpha = .1)

	#color_index += 1
	#IPython.embed()

	if len(colors) < num_experiments:
		raise ValueError("Number of colors < number of experiments. Base plot will fail.")

	for i in range(num_experiments):
		plot_data = hyperparam_stats[i,:]

		if sqrt_scaled:
			#IPython.embed()
			plot_data *= 1.0/np.sqrt(Ts*np.log(Ts+1))
		


		plt.plot(Ts, plot_data, color = 'blue')
		color_index += 1
	plt.xlabel("Number of batches")

	plt.ylabel(get_results_label(results_type, cummulative_plot))
	# plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
	#plt.legend(fontsize=8, loc="upper left")
	plt.title("Regrets {} {}".format(algo_name, hyperparam))
	filename = "{}/bases_{}_cum_{}-{}-{}_{}_T{}_B{}_N_{}.png".format(logging_dir,results_type, cummulative_plot, 
			algo_name, hyperparam, dataset, num_batches, batch_size, repres_layers_name)
	plt.savefig(filename)
	plt.close("all")





def plot_results(experiment_name, algo_name, dataset, results_type, num_batches, batch_size, modselalgo, 
	results_dictionary, hyperparams, colors, representation_layer_sizes, cummulative_plot = False, 
	averaging_window = 1 , split=False, sqrt_scaled = False):


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


	if sqrt_scaled and not cummulative_plot:
		raise ValueError("Sqrt scaled on and cummulative plot off. This is not allowed")


	# Plot hyper sweep results

	reduced_hyperparams = list(set(hyperparams))

	for hyperparam in reduced_hyperparams:

		hyperparam_results = results_dictionary["{}-{}".format(algo_name, hyperparam)] 
		if cummulative_plot:
			hyperparam_stats = np.array([np.cumsum(x[results_type]) for x in hyperparam_results])



		else:
			hyperparam_stats = np.array([x[results_type] for x in hyperparam_results])





		hyperparam_results_mean = np.mean(hyperparam_stats,0)
		hyperparam_results_std = np.std(hyperparam_stats,0)

		if sqrt_scaled:
			#IPython.embed()
			hyperparam_results_mean *= 1.0/np.sqrt(Ts*np.log(Ts+1))
			hyperparam_results_std *= 1.0/np.sqrt(Ts*np.log(Ts+1))

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
	
	if sqrt_scaled:
			modsel_stat_mean *= 1.0/np.sqrt(Ts*np.log(Ts+1))
			modsel_stat_std *= 1.0/np.sqrt(Ts*np.log(Ts+1))



	plt.plot(Ts, modsel_stat_mean, color = colors[color_index] ,  label = "{} {}".format(algo_name, modselalgo))
	plt.fill_between(Ts, modsel_stat_mean-.5*modsel_stat_std, 
		modsel_stat_mean+.5*modsel_stat_std, color = colors[color_index], alpha = .2)

	#color_index += 1


	label = get_results_label(results_type, cummulative_plot)


	repres_layers_name = get_architecture_name(representation_layer_sizes)


		
	
	plt.xlabel("Number of batches")

	plt.ylabel(label)
	# plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
	plt.legend(fontsize=8, loc="upper left")

	if not split:
		filename = "{}/{}_{}_cum_{}-{}_{}_{}_T{}_B{}_N_{}.png".format(logging_dir, experiment_name, results_type, cummulative_plot, 
			algo_name, modselalgo,dataset, num_batches, batch_size, repres_layers_name)
		plt.title("{} {} {} B{} N {}".format( label, modselalgo, dataset, batch_size, repres_layers_name))	
	else:

		filename = "{}/{}_{}-split_cum_{}-{}_{}_{}_T{}_B{}_N_{}.png".format(logging_dir,experiment_name, results_type, cummulative_plot, 
			algo_name, modselalgo,dataset, num_batches, batch_size, repres_layers_name)
		plt.title("{} {} split {} B{} N {}".format( label, modselalgo, dataset, batch_size, repres_layers_name))
	
	plt.savefig(filename)
	plt.close("all")




def plot_contrast_modsel_results(experiment_name, algo_name, dataset, results_type, num_batches, batch_size, 
	modselalgos, modsel_keys, 
	results_dictionary, colors, representation_layer_sizes, cummulative_plot = False, 
	averaging_window = 1 , split=False, sqrt_scaled = False):


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

	if sqrt_scaled and not cummulative_plot:
		raise ValueError("Sqrt scaled on and cummulative plot off. This is not allowed")


	#IPython.embed()

	for modsel_key, modselalgo in zip(modsel_keys, modselalgos):

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
		
		if sqrt_scaled:
				modsel_stat_mean *= 1.0/np.sqrt(Ts*np.log(Ts+1))
				modsel_stat_std *= 1.0/np.sqrt(Ts*np.log(Ts+1))


		plt.plot(Ts, modsel_stat_mean, color = colors[color_index] ,  label = "{} {}".format(algo_name, modselalgo))
		plt.fill_between(Ts, modsel_stat_mean-.5*modsel_stat_std, 
			modsel_stat_mean+.5*modsel_stat_std, color = colors[color_index], alpha = .2)



		color_index += 1
	label = get_results_label(results_type, cummulative_plot)
	

	repres_layers_name = get_architecture_name(representation_layer_sizes)


	plt.xlabel("Number of batches")

	plt.ylabel(label)
	# plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
	plt.legend(fontsize=8, loc="upper left")

	if not split:
		filename = "{}/{}_combined_{}_cum_{}-{}_{}_T{}_B{}_N_{}.png".format(logging_dir,experiment_name,results_type, cummulative_plot, 
			algo_name,dataset, num_batches, batch_size, repres_layers_name)
		plt.title("{} {} B{} N {}".format( label, dataset, batch_size, repres_layers_name))

	else:

		filename = "{}/{}_combined_{}-split_cum_{}-{}_{}_T{}_B{}_N_{}.png".format(logging_dir,experiment_name,results_type, cummulative_plot, 
			algo_name,dataset, num_batches, batch_size, repres_layers_name)
		plt.title("{} split {} B{} N {}".format( label, dataset, batch_size, repres_layers_name))

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


def plot_all(experiment_name,dataset, results_dictionary, num_batches, batch_size, split, hyperparams,
	algo_type_key, modselalgo, colors, representation_layer_sizes, 
	averaging_window, modsel_keys):


	plot_modsel_probabilities(experiment_name,algo_type_key, dataset, num_batches, batch_size, modselalgo, 
			results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes, split = split)

	plot_results(experiment_name,algo_type_key, dataset, "instantaneous_regrets", num_batches, batch_size, modselalgo, 
			results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
			 cummulative_plot = True , averaging_window = averaging_window, split = split, sqrt_scaled = True)

	plot_results(experiment_name,algo_type_key, dataset, "instantaneous_accuracies", num_batches, batch_size, modselalgo, 
			results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
			cummulative_plot = False, averaging_window = averaging_window, split = split)

	if PLOT_ALL_STATS:
		plot_results(experiment_name,algo_type_key, dataset, "num_negatives", num_batches, batch_size, modselalgo, 
				results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
				 cummulative_plot = False, averaging_window = averaging_window, split = split)

		plot_results(experiment_name,algo_type_key, dataset, "num_positives", num_batches, batch_size, modselalgo, 
				results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
				 cummulative_plot = False, averaging_window = averaging_window, split = split)

		plot_results(experiment_name,algo_type_key, dataset, "false_neg_rates", num_batches, batch_size, modselalgo, 
				results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
				 cummulative_plot = False, averaging_window = averaging_window, split = split)

		plot_results(experiment_name,algo_type_key, dataset, "false_positive_rates", num_batches, batch_size, modselalgo, 
				results_dictionary, hyperparams, colors, representation_layer_sizes = representation_layer_sizes,
				cummulative_plot = False, averaging_window = averaging_window, split = split)


	plot_optimism_pessimism(experiment_name,algo_type_key, dataset, num_batches, batch_size, results_dictionary, 
		hyperparams, colors, representation_layer_sizes, averaging_window = averaging_window)





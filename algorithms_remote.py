import ray

from algorithmsmodsel import train_epsilon_greedy_modsel, train_mahalanobis_modsel, train_opt_reg_modsel
from algorithms import train_epsilon_greedy, train_mahalanobis, train_baseline, train_opt_reg


@ray.remote
def train_baseline_remote(dataset, num_timesteps = 1000, batch_size = 32, 
    representation_layer_sizes = [10,10]):
    return train_baseline(dataset=dataset, num_timesteps=num_timesteps, batch_size=batch_size, 
    representation_layer_sizes=representation_layer_sizes)


# @ray.remote
# def train_baseline_remote(dataset, num_timesteps = 1000, batch_size = 32, 
#     MLP = True, representation_layer_size = 10):
# 	return train_baseline(dataset=dataset, num_timesteps=num_timesteps, batch_size=batch_size, 
#     MLP=MLP, representation_layer_size=representation_layer_size)


@ray.remote
def train_epsilon_greedy_remote(dataset, baseline_model, 
    num_batches = 20, batch_size = 32, 
    num_opt_steps = 1000, opt_batch_size = 20, 
    representation_layer_sizes = [10, 10], threshold = .5, verbose = True, decaying_epsilon = False, epsilon = .1,
    restart_model_full_minimization = False):

	return train_epsilon_greedy(dataset=dataset, baseline_model=baseline_model, 
    num_batches=num_batches, batch_size=batch_size , 
    num_opt_steps=num_opt_steps, opt_batch_size=opt_batch_size ,
    representation_layer_sizes=representation_layer_sizes , threshold=threshold , verbose=verbose , decaying_epsilon=decaying_epsilon,
    epsilon = epsilon, restart_model_full_minimization=restart_model_full_minimization )


# @ray.remote
# def train_epsilon_greedy_remote(dataset, baseline_model, 
#     num_batches = 20, batch_size = 32, 
#     num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
#     representation_layer_size = 10, threshold = .5, verbose = True, decaying_epsilon = False, epsilon = .1,
#     restart_model_full_minimization = False):

#     return train_epsilon_greedy(dataset=dataset, baseline_model=baseline_model, 
#     num_batches=num_batches, batch_size=batch_size , 
#     num_opt_steps=num_opt_steps, opt_batch_size=opt_batch_size , MLP=MLP , 
#     representation_layer_size=representation_layer_size , threshold=threshold , verbose=verbose , decaying_epsilon=decaying_epsilon,
#     epsilon = epsilon, restart_model_full_minimization=restart_model_full_minimization )




@ray.remote
def train_epsilon_greedy_modsel_remote(dataset, baseline_model, 
    num_batches = 20, batch_size = 32, 
    num_opt_steps = 1000, opt_batch_size = 20,
    representation_layer_sizes = [10, 10], threshold = .5, verbose = True, decaying_epsilon = False, epsilons = [.01, .1, .2],
    restart_model_full_minimization = False, modselalgo = "Corral"):

	return train_epsilon_greedy_modsel(dataset=dataset, baseline_model=baseline_model, 
    num_batches=num_batches, batch_size=batch_size , 
    num_opt_steps=num_opt_steps , opt_batch_size=opt_batch_size ,
    representation_layer_sizes=representation_layer_sizes, threshold=threshold , verbose=verbose , 
    decaying_epsilon=decaying_epsilon , epsilons=epsilons ,
    restart_model_full_minimization=restart_model_full_minimization , modselalgo=modselalgo )



@ray.remote
def train_mahalanobis_remote(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size,
    representation_layer_sizes = [10,10], threshold = .5, alpha = 1, lambda_reg = 1,
    verbose = False, 
    restart_model_full_minimization = False):


    return train_mahalanobis(dataset=dataset, baseline_model=baseline_model, num_batches=num_batches, batch_size=batch_size, 
    num_opt_steps=num_opt_steps, opt_batch_size=opt_batch_size, 
    representation_layer_sizes = representation_layer_sizes, threshold = threshold, 
    alpha = alpha, lambda_reg = lambda_reg,
    verbose = verbose, 
    restart_model_full_minimization = restart_model_full_minimization)




@ray.remote
def train_mahalanobis_modsel_remote(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size, 
    representation_layer_sizes = [10, 10], threshold = .5, alphas = [1, .1, .01], lambda_reg = 1,
    verbose = False, 
    restart_model_full_minimization = False, modselalgo = "Corral", split = False):



	return train_mahalanobis_modsel(dataset=dataset, baseline_model=baseline_model, num_batches=num_batches, batch_size=batch_size, 
    num_opt_steps=num_opt_steps, opt_batch_size=opt_batch_size, 
    representation_layer_sizes = representation_layer_sizes, threshold = threshold, alphas = alphas, lambda_reg = lambda_reg,
    verbose = verbose, 
    restart_model_full_minimization = restart_model_full_minimization, modselalgo = modselalgo, split = split)


@ray.remote
def train_opt_reg_remote(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size,
    representation_layer_sizes = [10, 10], threshold = .5, reg = 1,
    verbose = False,
    restart_model_full_minimization = False):

    return train_opt_reg(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size,
    representation_layer_sizes = representation_layer_sizes, threshold = threshold, reg = reg,
    verbose = verbose,
    restart_model_full_minimization = restart_model_full_minimization)





@ray.remote
def train_opt_reg_modsel_remote(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size,
    representation_layer_sizes = [10, 10], threshold = .5, regs = [1, .1, .01],
    verbose = False,
    restart_model_full_minimization = False, modselalgo = "Corral", split = False):

    return train_opt_reg_modsel(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size,
    representation_layer_sizes = representation_layer_sizes, threshold = threshold, regs = regs,
    verbose = verbose,
    restart_model_full_minimization = restart_model_full_minimization, modselalgo = modselalgo, split = split)



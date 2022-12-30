num_opt_steps = 2000
num_baseline_steps = 20000
opt_batch_size = 20
burn_in = 10
retraining_frequency = 40
averaging_window = 1
restart_model_full_minimization = True
batch_size = 10


modselalgos = [ 'BalancingDoResurrectClassic','BalancingDoResurrectDown', 
	"BalancingDoResurrect", "BalancingDoubling", "Corral", "UCB", "EXP3"]



decaying_epsilon = False

#epsilon = .1
#alpha = 10
#epsilons = 
#alphas = #[.000001, 1/4.0, 1/2.0, 1, 2, 4, 8 ]

# Experiment Specific Parameter Map
experiment_parameter_map = dict([])
experiment_parameter_map["classic"] = dict([("alphas", [.05, .5, 1, 4]), ("epsilons",[.2, .1, .01, .05] ), ("opt_regs", [.08, .16, 1, 10])  ])
experiment_parameter_map["self"] =  dict([("alphas", [.5, .5, .5, .5,.5]), ("epsilons",[ .1,.1,.1,.1,.1] ), ("opt_regs", [.16, .16,.16,.16,.16])  ])
experiment_parameter_map["self-greedy"] =  dict([("alphas", [.05, .05, .05, .05,.05]), ("epsilons",[ .01,.01,.01,.01,.01] ), ("opt_regs", [.08, .08, .08, .08, .08])  ])


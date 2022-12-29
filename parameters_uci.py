num_opt_steps = 2000
num_baseline_steps = 20000
opt_batch_size = 20
burn_in = 10


retraining_frequency = 40

averaging_window = 1
epsilon = .1
alpha = 10
epsilons = [.2, .1, .01, .05]
alphas = [.05, .5, 1, 4]#[.000001, 1/4.0, 1/2.0, 1, 2, 4, 8 ]
opt_regs = [.08, .16, 1, 10]
decaying_epsilon = False

restart_model_full_minimization = True
batch_size = 10

modselalgos = [ 'BalancingDoResurrectClassic','BalancingDoResurrectDown', 
	"BalancingDoResurrect", "BalancingDoubling", "Corral", "UCB", "EXP3"]

import torch
from evotorch.tools import dtype_of, device_of
from test_f_env import *

def eval_func(network: torch.nn.Module):
	hor, a, b = 50, 1, 1
	tolerance = 0.2
	env = test_f_env_AHT(hor,1,1)
	episodes = 100
	av_aer, av_ler, av_fit = 0, 0, 0
	for episode in range(1,episodes+1):
		state = env.reset()
		done = False
		while not done:
			state = state.reshape(1,8)
			net_out = network(torch.from_numpy(state).float())
			# print(net_out)
			state, fitness, done, info = env.step(int(torch.argmax(net_out)))
		av_ler += env.ler
		av_aer += env.aer
	av_ler, av_aer = av_ler/episodes, av_aer/episodes
	if av_ler > tolerance:
		return -av_ler
	return av_aer-av_ler

from evotorch.neuroevolution import NEProblem
from collections import OrderedDict

def policy():
	return torch.nn.Sequential(OrderedDict([
          ('lin1', torch.nn.Linear(8,100)),
          ('relu1', torch.nn.ReLU()),
          ('lin2', torch.nn.Linear(100, 100)),
          ('relu2', torch.nn.ReLU()),
          ('lin3', torch.nn.Linear(100, 100)),
          ('relu3', torch.nn.ReLU()),
          ('out', torch.nn.Linear(100, 3)),
          ('softm', torch.nn.Softmax(dim=None))
        ]))

problem = NEProblem(
    # The objective sense -- we wish to maximize the sign_prediction_score
    objective_sense="max",
    # The network is a Linear layer mapping 3 inputs to 1 output
    network=policy,
    # Networks will be evaluated according to sign_prediction_score
    network_eval_func=eval_func,
    initial_bounds=(-5,5),
    eval_dtype=np.float64(), 
	)
print(problem.parameterize_net(problem.make_zeros(problem.solution_length)))

from evotorch.logging import StdOutLogger
from evotorch.logging import PandasLogger
from evotorch.algorithms import Cosyne

searcher = Cosyne(
    problem,
    num_elites = 2,
    popsize=100,  
    tournament_size = 4,
    mutation_stdev = 0.3,
    mutation_probability = 0.5,
    permute_all = True, 
)

logger = StdOutLogger(searcher, interval=1)
p_logger = PandasLogger(searcher)
searcher.run(10)
trained_network = problem.parameterize_net(searcher.status["pop_best"])

PATH = "./models/GANN/torch_policy"
# torch.save(trained_network, PATH)

import matplotlib.pyplot as plt
p_logger.to_dataframe().mean_eval.plot()
plt.title('Population mean evaluation')
plt.ylabel('Evaluation')
plt.xlabel('Generation')
plt.legend()
plt.grid()
plt.show()

policy = torch.load(PATH)
policy.eval()


hor = 50
a = 1
b = 1
episodes = 1000
timespace = np.arange(0,hor,1)
env = test_f_env_AHT(hor, a, b)
err = [[0]*hor,[0]*hor]

for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    while not done:
        err[0][env.t] += env.ler/episodes 
        err[1][env.t] += env.aer/episodes
        state = state.reshape(1,8)
        action = policy(torch.from_numpy(state).float())
        state, fitness, done, info = env.step(int(torch.argmax(action)))
    print(episode)

# fig =plt.gcf()
# fig.set_dpi(300)
# fig.set_size_inches(3.5, 2.625)
plt.title('Error Averaged over {} episodes'.format(episodes))
plt.ylabel('Error Probability')
plt.xlabel('Horizon')
plt.plot(timespace, err[0], '-g', label = "Legit")
plt.plot(timespace, err[1], '-b', label = "Adversary")
plt.legend()
plt.grid()
plt.show()   
import numpy as np
import pygad
import pygad.nn
import pygad.gann
import matplotlib.pyplot as plt
from test_f_env import *

def fitness_func(solution, sol_idx):
	global GANN_instance
	hor, a, b = 50, 1, 1
	tolerance = 0.2
	env = test_f_env_AHT(hor,1,1)
	episodes = 100
	av_aer, av_ler = 0, 0
	for episode in range(1,episodes+1):
		state = env.reset()
		done = False
		while not done:
			state = state.reshape(1,8)
			action = pygad.nn.predict(last_layer = GANN_instance.population_networks[sol_idx], data_inputs = state, problem_type = "classification")
			state, fitness, done, info = env.step(action[0])
		av_ler += env.ler 
		av_aer += env.aer 
	fitness = 0
	av_ler, av_aer = av_ler/episodes, av_aer/episodes
	if av_ler > tolerance:
		fitness = -av_ler
	else:
		fitness = av_aer
	return fitness

def callback_generation(ga_instance):
    global GANN_instance

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks, population_vectors=ga_instance.population)
    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

GANN_instance = pygad.gann.GANN(num_solutions = 100,
							 	num_neurons_input = 8,
							 	num_neurons_output = 3,
							 	num_neurons_hidden_layers = [100,100],
							 	output_activation = "softmax",
							 	hidden_activations = ["relu","relu"])

population_vectors = pygad.gann.population_as_vectors(population_networks = GANN_instance.population_networks)

initial_population = population_vectors.copy()
num_parents_mating = 4
num_generations = 60
mutation_percent_genes = 3
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
keep_parents = 1
init_range_low = -5
init_range_high = 5

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       mutation_percent_genes=mutation_percent_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

# ga_instance.run()
# ga_instance.plot_fitness()

# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# filename = 'models/GANN/genetic'
# ga_instance.save(filename = filename)

filename = 'models/GANN/genetic'
ga_instance = pygad.load(filename = filename)
solution, solution_fitness, solution_idx = ga_instance.best_solution()

episodes = 100
hor = 50
test_env = test_f_env_AHT(hor,1,1)
ga_av_err = [[0]*hor,[0]*hor]
timespace = np.arange(0,hor,1)

for episode in range(1,episodes+1):
	state = test_env.reset()
	done = False
	while not done:
		state = state.reshape(1,8)
		ga_av_err[0][test_env.t] += test_env.ler/episodes
		ga_av_err[1][test_env.t] += test_env.aer/episodes
		action = pygad.nn.predict(last_layer=GANN_instance.population_networks[solution_idx],
						 		  data_inputs=state,
						 		  problem_type="classification")
		state, r, done, info = test_env.step(np.argmax(action))

plt.title('Error Averaged over {} episodes'.format(episodes))
plt.ylabel('Error Probability')
plt.xlabel('Steps')
plt.plot(timespace, ga_av_err[0], '-b', label = 'Legit')
plt.plot(timespace, ga_av_err[1], '-g', label = 'Adversary')
plt.legend()
plt.grid()
plt.show()

print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
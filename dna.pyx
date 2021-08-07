import numpy as np
cimport numpy as np
cimport cython as cy
from libc.math cimport exp
import time
np.set_printoptions(suppress=True)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef double[::1] MV1D
ctypedef double[:, ::1] MV2D
ctypedef double[:, :, ::1] MV3D

@cy.cdivision(True)
cdef double expit_(double x):
    return 1 / (1 + exp(-x))

cdef double expit(double x):
    if x:
        return expit_(x)
    return x

@cy.boundscheck(False)
@cy.wraparound(False)
cdef object softmax(np.ndarray[DTYPE_t, ndim=1] lov):
    cdef double lov_sum
    lov_arr = np.exp(lov)
    lov_arr_sum = np.sum(lov_arr)
    return lov_arr / lov_arr_sum

@cy.boundscheck(False)
@cy.wraparound(False)
cdef bint chance(double p):
    return np.random.random() <= p

@cy.boundscheck(False)
@cy.wraparound(False)
cdef double safe_weight():
    cdef double weight = np.random.random()
    while not(weight):
        weight += np.random.random()
    if chance(0.5):
        return -weight
    return weight

@cy.boundscheck(False)
@cy.wraparound(False)
cdef MV1D expit_array(MV1D arr):
    cdef int N = arr.shape[0]
    cdef MV1D new_arr = np.empty(N)
    cdef int i
    for i in range(N):
        new_arr[i] = expit(arr[i])
    return new_arr

@cy.boundscheck(False)
@cy.wraparound(False)
cdef double compat_score(MV3D net):
    cdef MV2D omega = net[1]
    cdef MV2D omega_to_eval = np.tril(omega)
    cdef MV1D net_ev = np.abs(np.linalg.eigvals(omega_to_eval))
    return np.dot(net_ev, net_ev.T)

@cy.boundscheck(False)
@cy.wraparound(False)
cdef double weight_score(MV3D net):
    cdef MV2D omega = net[1]
    cdef MV2D omega_to_eval = np.tril(omega)
    return np.sum(np.abs(omega_to_eval))


# How many networks do we start with?
cdef int POPULATION_SIZE = 150

# Max weight perturbation per mutation
cdef int PERTURB_WEIGHT_DELTA = 25

# Chance we perturb weights by making them stronger during uniform perturbation
cdef double CHANCE_PERTURB_HIGHER = 0.5

# Chance we perturb weights uniformly (rather than reassigning them randomly)
cdef double CHANCE_PERTURB_UNIFORMLY = 0.9

# Target value for network to compute
cdef double TRUE_VALUE = 0.112358

# Error threshold
cdef double GOOD_ENOUGH = 0.0001

#####################
# Network Constants #
#####################
# 24 input nodes
# 18 hidden layer 0 nodes
# 18 hidden layer 1 nodes
# 12 hidden layer 2 nodes
# 1 output node
# 73 nodes total, 5 layers total
cdef int input_layer_min_idx = 0
cdef int input_layer_max_idx = 24
cdef int h0_min_idx = 24
cdef int h0_max_idx = 42
cdef int h1_min_idx = 42
cdef int h1_max_idx = 60
cdef int h2_min_idx = 60
cdef int h2_max_idx = 72
cdef Py_ssize_t output_layer_idx = 72
cdef int N_LAYERS = 5


@cy.boundscheck(False)
@cy.wraparound(False)
@cy.cdivision(True)
cdef double return_weight_perturbation():
    return np.random.randint(1, (PERTURB_WEIGHT_DELTA + 1)) / 100

@cy.boundscheck(False)
@cy.wraparound(False)
cdef double perturb_weight_uniformly(double weight):
    cdef double perturbation = return_weight_perturbation()
    if chance(CHANCE_PERTURB_HIGHER):
        return min(1.0, weight + perturbation)
    return max(-1.0, weight - perturbation)

@cy.boundscheck(False)
@cy.wraparound(False)
cdef double mutate_weight(double weight):
    if chance(CHANCE_PERTURB_UNIFORMLY):
        return perturb_weight_uniformly(weight)
    return safe_weight()


@cy.boundscheck(False)
@cy.wraparound(False)
cdef MV3D clone_myself(MV3D source_tensor, bint with_mutation):
    cdef MV3D result = source_tensor.copy()
    if with_mutation:
        result = mutate_connections(result)
    return result

@cy.boundscheck(False)
@cy.wraparound(False)
cpdef MV3D clone_wrapper(MV3D source_tensor, bint with_mutation):
    return clone_myself(source_tensor, with_mutation)


@cy.boundscheck(False)
@cy.wraparound(False)
cdef MV3D mutate_connections(MV3D tensor):
    cdef Ni = 73
    cdef Nj = 73
    cdef Py_ssize_t i, j
    for i in range(Ni):
        for j in range(Nj):
            tensor[1, i, j] = mutate_weight(tensor[1, i, j])
    return tensor

@cy.boundscheck(False)
@cy.wraparound(False)
cdef MV3D compose_matrices(MV3D m0, MV3D m1, bint with_mutation):
    cdef MV3D offspring = clone_myself(m0, False)
    cdef int min_x_idx = 0
    cdef int max_x_idx = offspring.shape[1]
    cdef int min_y_idx = 0
    cdef int max_y_idx = offspring.shape[2]
    cdef double value_to_copy
    cdef Py_ssize_t curr_x_idx, curr_y_idx
    for curr_x_idx in range(min_x_idx, max_x_idx):
        for curr_y_idx in range(min_y_idx, max_y_idx):
            if chance(0.5):
                offspring[1, curr_x_idx, curr_y_idx] = m1[1, curr_x_idx, curr_y_idx]
    if with_mutation:
        offspring = mutate_connections(offspring)
    return offspring

@cy.boundscheck(False)
@cy.wraparound(False)
cdef void store_input_vector(MV1D input_vector, MV3D net):
    net[0, 0, input_layer_min_idx:input_layer_max_idx] = input_vector

@cy.boundscheck(False)
@cy.wraparound(False)
cdef void store_h0_vector(MV1D h0_vector, MV3D net):
    net[0, 0, h0_min_idx:h0_max_idx] = h0_vector

@cy.boundscheck(False)
@cy.wraparound(False)
cdef void store_h1_vector(MV1D h1_vector, MV3D net):
    net[0, 0, h1_min_idx:h1_max_idx] = h1_vector

@cy.boundscheck(False)
@cy.wraparound(False)
cdef void store_h2_vector(MV1D h2_vector, MV3D net):
    net[0, 0, h2_min_idx:h2_max_idx] = h2_vector

@cy.boundscheck(False)
@cy.wraparound(False)
cdef np.ndarray[double, ndim=1] compute_h0_activations(MV3D net):
    cdef np.ndarray[double, ndim=1] x = np.asarray(net[0, 0, input_layer_min_idx:input_layer_max_idx])
    cdef np.ndarray[double, ndim=2] W = np.asarray(net[1, h0_min_idx:h0_max_idx, input_layer_min_idx:input_layer_max_idx])
    cdef np.ndarray[double, ndim=1] h0_vector = np.dot(x, W.T)
    return h0_vector

@cy.boundscheck(False)
@cy.wraparound(False)
cdef np.ndarray[double, ndim=1] compute_h1_activations(MV3D net):
    cdef np.ndarray[double, ndim=1] x = np.asarray(net[0, 0, input_layer_min_idx:h0_max_idx])
    cdef np.ndarray[double, ndim=2] W = np.asarray(net[1, h1_min_idx:h1_max_idx, input_layer_min_idx:h0_max_idx])
    cdef np.ndarray[double, ndim=1] h1_vector = np.dot(x, W.T)
    return h1_vector

@cy.boundscheck(False)
@cy.wraparound(False)
cdef np.ndarray[double, ndim=1] compute_h2_activations(MV3D net):
    cdef np.ndarray[double, ndim=1] x = np.asarray(net[0, 0, input_layer_min_idx:h1_max_idx])
    cdef np.ndarray[double, ndim=2] W = np.asarray(net[1, h2_min_idx:h2_max_idx, input_layer_min_idx:h1_max_idx])
    cdef np.ndarray[double, ndim=1] h2_vector = np.dot(x, W.T)
    return h2_vector

@cy.boundscheck(False)
@cy.wraparound(False)
cdef double compute_output_activations(MV3D net):
    cdef np.ndarray[double, ndim=1] x = np.asarray(net[0, 0, input_layer_min_idx:h2_max_idx])
    cdef np.ndarray[double, ndim=1] W = np.asarray(net[1, output_layer_idx, input_layer_min_idx:h2_max_idx])
    cdef double output_activation = np.dot(x, W.T)
    return output_activation

@cy.boundscheck(False)
@cy.wraparound(False)
cdef MV3D init_tensor():
    cdef int sample_size = 5329
    cdef np.ndarray[DTYPE_t, ndim=3] tensor
    cdef np.ndarray[DTYPE_t, ndim=2] tensor0 = np.zeros((73,73), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] randoms = np.random.random_sample(size=sample_size)
    cdef np.ndarray[DTYPE_t, ndim=2] randoms_
    randoms = 2 * randoms
    randoms = randoms - 1
    randoms = np.asarray(randoms)
    randoms_ = randoms.reshape((73,73))
    tensor = np.stack((tensor0, randoms_), axis=0)
    return tensor

@cy.boundscheck(False)
@cy.wraparound(False)
cdef MV3D init_tensor_with_input(MV1D inp_vec):
    cdef MV3D tensor = init_tensor()
    store_input_vector(input_vector=inp_vec, net=tensor)
    return tensor

@cy.boundscheck(False)
@cy.wraparound(False)
cpdef MV3D test_init_tensor():
    return init_tensor()

@cy.boundscheck(False)
@cy.wraparound(False)
cpdef MV3D test_init_tensor_with_input():
    cdef MV1D iv = np.random.random_sample(size=24)
    return init_tensor_with_input(inp_vec=iv)

@cy.boundscheck(False)
@cy.wraparound(False)
cdef double feedforward(MV1D input_vector, MV3D net):
    # Apply the non-linearity to the input vector
    cdef MV1D input_vector_ = expit_array(input_vector)
    # Store it
    store_input_vector(input_vector_, net)
    # Compute h0_activations
    cdef MV1D h0_vector = compute_h0_activations(net)
    # Apply the non-linearity to the h0_vector
    cdef MV1D h0_vector_ = expit_array(h0_vector)
    # Store it
    store_h0_vector(h0_vector_, net)
    # Compute h1 activations
    cdef MV1D h1_vector = compute_h1_activations(net)
    # Apply the non-linearity to the h1_vector
    cdef MV1D h1_vector_ = expit_array(h1_vector)
    # Store it
    store_h1_vector(h1_vector_, net)
    # Compute h2 activations
    cdef MV1D h2_vector = compute_h2_activations(net)
    # Apply the non-linearity to the h2_vector
    cdef MV1D h2_vector_ = expit_array(h2_vector)
    # Store it
    store_h2_vector(h2_vector_, net)
    # Compute the output activations
    cdef double output_activation = compute_output_activations(net)
    # Apply the non-linearity to the output_vector
    cdef double output_activation_ = expit(output_activation)
    # Return it
    return output_activation_

cdef MV1D test_input_vector0 = np.array([np.random.randint(6) for i in range(input_layer_max_idx)], dtype=np.float64)


@cy.boundscheck(False)
@cy.wraparound(False)
cdef list solve_species(list species_list):
    cdef list species_list_copy = list(species_list)
    cdef list results = []
    cdef list true_results = []
    cdef list exemplars = []
    cdef list s0, s1, s2, s3, s4, sorted_species_list
    cdef int N_networks, i0, i1, i2, i3

    sorted_species_list = sorted(species_list_copy, key=lambda network: compat_score(network), reverse=True)
    N_networks = len(sorted_species_list)
    i0 = int(np.floor(N_networks / 5))
    i1 = i0 * 2
    i2 = i0 * 3
    i3 = i0 * 4
    s0 = sorted_species_list[:i0]
    s1 = sorted_species_list[i0:i1]
    s2 = sorted_species_list[i1:i2]
    s3 = sorted_species_list[i2:i3]
    s4 = sorted_species_list[i3:]
    np.random.shuffle(s0)
    np.random.shuffle(s1)
    np.random.shuffle(s2)
    np.random.shuffle(s3)
    np.random.shuffle(s4)
    exemplars.append(s0[0])
    exemplars.append(s1[0])
    exemplars.append(s2[0])
    exemplars.append(s3[0])
    exemplars.append(s4[0])
    results.append(s0)
    results.append(s1)
    results.append(s2)
    results.append(s3)
    results.append(s4)
    true_results.append(results)
    true_results.append(exemplars)
    return true_results


cdef class Simulation:
    cdef public list networks, exemplars, species_list
    cdef public object winner
    cdef public double winner_fitness, winner_attempt
    cdef dict __dict__

    def __init__(self):
        self.networks = []
        self.exemplars = []
        self.species_list = []
        self.winner = None
        self.winner_fitness = -666.666
        self.winner_attempt = -666.666

    cdef void speciate(self, MV3D network):
        self.exemplars.append(network)
        self.species_list.append([network])

    @cy.boundscheck(False)
    @cy.wraparound(False)
    cdef void statistics(self):
        cdef MV3D this_network
        cdef Py_ssize_t net_idx
        cdef int N_network = len(self.networks)
        cdef double M, SD, max_, min_
        cdef np.ndarray[DTYPE_t, ndim=1] fitnesses = np.zeros((N_network))
        for net_idx in range(N_network):
            this_network = self.networks[net_idx]
            fitnesses[net_idx] = this_network[0,72,0]
        M = np.mean(fitnesses)
        SD = np.std(fitnesses)
        max_ = np.max(fitnesses)
        min_ = np.min(fitnesses)
        print("M: {} SD: {} Max: {} Min: {}".format(M, SD, max_, min_))

    @cy.boundscheck(False)
    @cy.wraparound(False)
    cdef void solve_species(self):
        temp_result = solve_species(self.networks)
        self.species_list = list(temp_result[0])
        self.exemplars = list(temp_result[1])

    @cy.boundscheck(False)
    @cy.wraparound(False)
    cdef list sort_species(self, list list_of_networks):
        return sorted(list_of_networks, key=lambda network: network[0, 72, 0], reverse=True)

    @cy.boundscheck(False)
    @cy.wraparound(False)
    cdef MV3D deduce_champion(self, list list_of_networks):
        cdef MV3D best_network = list_of_networks[0]
        cdef MV3D current_network
        cdef double best_fitness = best_network[0, 72, 0]
        cdef double current_fitness
        cdef list champions = []
        cdef Py_ssize_t species_idx
        cdef int N_network = len(list_of_networks)
        for species_idx in range(N_network):
            current_network = list_of_networks[species_idx]
            current_fitness = current_network[0, 72, 0]
            if (current_fitness == best_fitness):
                champions.append(current_network)
        np.random.shuffle(champions)
        return champions[0]

    @cy.boundscheck(False)
    @cy.wraparound(False)
    cdef double train_network(self, MV3D network_to_evaluate):
        cdef double attempt = feedforward(network_to_evaluate[0, 0, input_layer_min_idx:input_layer_max_idx], network_to_evaluate)
        cdef double delta = np.abs(TRUE_VALUE - attempt)
        if (delta <= GOOD_ENOUGH):
            self.winner = network_to_evaluate
            self.winner_attempt = attempt
        return 5.0 - delta

    @cy.boundscheck(False)
    @cy.wraparound(False)
    cdef double mean_fitness(self, list list_of_networks):
        cdef int N_network = len(list_of_networks)
        cdef np.ndarray[DTYPE_t, ndim=1] mean_array = np.zeros(N_network, dtype=DTYPE)
        cdef double fitness_sum = 0.0
        cdef double current_fitness
        cdef Py_ssize_t network_idx
        cdef MV3D current_network
        for network_idx in range(N_network):
            current_network = list_of_networks[network_idx]
            current_fitness = current_network[0, 72, 0]
            fitness_sum = fitness_sum + current_fitness
        return fitness_sum / N_network

    @cy.boundscheck(False)
    @cy.wraparound(False)
    cdef void evaluate_fitness(self, MV3D network_to_evaluate):
        cdef double fitness = self.train_network(network_to_evaluate)
        network_to_evaluate[0, 72, 0] = fitness
        if not(fitness is None):
            if (fitness <= GOOD_ENOUGH):
                self.winner = network_to_evaluate
                self.winner_fitness = fitness

    @cy.boundscheck(False)
    @cy.wraparound(False)
    cdef void evaluate_fitnesses(self):
        cdef Py_ssize_t species_idx
        cdef int N_networks = len(self.networks)
        for species_idx in range(N_networks):
            self.evaluate_fitness(self.networks[species_idx])

    @cy.boundscheck(False)
    @cy.wraparound(False)
    cdef void generate_generation(self):
        cdef object self_replicating_species, mating_species, current_mating_species
        cdef list new_generation, sorted_species, champions
        cdef np.ndarray[DTYPE_t, ndim=1] mean_fitnesses
        cdef list self_replicators, species_that_can_mate, species_index_list, mating_pairs
        cdef int n_champions, n_to_make, n_to_self_replicate, n_to_breed, n_samples, N_new_generation
        cdef int mating_species_idx
        cdef int species_idx = 0
        cdef int N_networks = len(self.networks)
        cdef int N_species = len(self.species_list)
        cdef int N_species_that_can_mate
        cdef bint sample_with_replacement = True
        cdef int N_mating_species, N_self_replicating_species, N_network_in_this_species, N_current_mating_species
        cdef int N_network_in_current_mating_species, N_mating_pairs
        cdef list current_mating_species_list_copy

        if (self.winner is None):
            new_generation = []
            self.evaluate_fitnesses()
            self.statistics()
            self.solve_species()

            sorted_species = []
            for species_idx in range(N_species):
                sorted_species.append(self.sort_species(self.species_list[species_idx]))

            champions = []
            for species_idx in range(N_species):
                champions.append(self.deduce_champion(self.species_list[species_idx]))

            n_champions = len(champions)
            n_to_make = POPULATION_SIZE - n_champions

            n_to_self_replicate = int(np.floor((n_to_make / 4)))
            n_to_breed = n_to_make - n_to_self_replicate

            N_species = len(self.species_list)
            mean_fitnesses = np.zeros(N_species, dtype=DTYPE)
            for species_idx in range(N_species):
                mean_fitnesses[species_idx] = self.mean_fitness(self.species_list[species_idx])

            clone_sample_pdist = softmax(mean_fitnesses)
            n_samples = n_to_self_replicate
            sample_with_replacement = True
            species_index_list = list(range(len(self.species_list)))
            self_replicating_species = np.random.choice(a=species_index_list,
                                                        size=n_samples,
                                                        replace=sample_with_replacement,
                                                        p=clone_sample_pdist)
            self_replicators = []
            N_self_replicating_species = len(self_replicating_species)
            for species_idx in range(N_self_replicating_species):
                species = self.species_list[self_replicating_species[species_idx]]
                np.random.shuffle(species)
                self_replicators.append(species[0])

            species_that_can_mate = list(filter(lambda species: (len(species) > 1), self.species_list))
            N_species_that_can_mate = len(species_that_can_mate)
            mean_fitnesses = np.zeros(N_species_that_can_mate, dtype=DTYPE)
            for species_idx in range(N_species_that_can_mate):
                mean_fitnesses[species_idx] = self.mean_fitness(species_that_can_mate[species_idx])

            mating_sample_pdist = softmax(mean_fitnesses)
            n_samples = n_to_breed
            sample_with_replacement = True
            species_index_list = list(range(len(species_that_can_mate)))
            mating_species = np.random.choice(a=species_index_list,
                                              size=n_samples,
                                              replace=sample_with_replacement,
                                              p=mating_sample_pdist)

            mating_pairs = []
            N_current_mating_species = len(mating_species)
            for species_idx in range(N_current_mating_species):
                current_mating_species_idx = mating_species[species_idx]
                current_mating_species = species_that_can_mate[current_mating_species_idx]
                N_network_in_current_mating_species = len(current_mating_species)
                current_mating_species_list_copy = []
                for s in range(N_network_in_current_mating_species):
                    current_mating_species_list_copy.append(current_mating_species[s])
                np.random.shuffle(current_mating_species_list_copy)
                mating_pairs.append((current_mating_species_list_copy[0], current_mating_species_list_copy[1]))

            # Add the champions to the new generation as mutation-free clones
            for species_idx in range(n_champions):
                new_generation.append(clone_myself(champions[species_idx], False))

            # Add the clones to the new generation
            for species_idx in range(N_self_replicating_species):
                new_generation.append(clone_myself(self_replicators[species_idx], True))

            # Add the offspring of the mating pairs to the new generation
            N_mating_pairs = len(mating_pairs)
            for species_idx in range(N_mating_pairs):
                mating_pair = mating_pairs[species_idx]
                new_generation.append(compose_matrices(mating_pair[0], mating_pair[1], True))

            self.networks = []
            N_new_generation = len(new_generation)
            for species_idx in range(N_new_generation):
                self.networks.append(new_generation[species_idx])


        else:
            print("We have a winner.")

    cpdef void gen_gen_wrapper(self):
        self.generate_generation()


SIM = Simulation()
cdef int sim_i
cdef int sim_N = POPULATION_SIZE
for sim_i in range(sim_N):
    SIM.networks.append(init_tensor_with_input(inp_vec=test_input_vector0))

@cy.boundscheck(False)
@cy.wraparound(False)
cpdef void test():
    cdef int iter_idx
    cdef int N_iter = 100
    cdef double start_time = time.time()
    for iter_idx in range(N_iter):
        if (SIM.winner is None):
            print("Generation: {} [{}]".format(iter_idx, time.time()-start_time))
            SIM.gen_gen_wrapper()
        else:
            break
    print("Generation: {} [{}]".format(iter_idx, time.time()-start_time))
    print(SIM.winner_attempt)


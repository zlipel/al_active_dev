from __future__ import annotations

import numpy as np
import random
import sys  
import matplotlib.pyplot as plt

class geneticalgorithm_batch():

    """
    fairly simply GA class
    """
    ### note right now we always use int; this is probably just overkill... should change it.
    def __init__(self, function=None, variable_type='int', \
                 function_timeout=100,\
                 algorithm_parameters={'max_num_iteration': 100,\
                                       'population_size':96,\
                                       'mutation_probability':0.5,\
                                       'elit_ratio':0.01,\
                                       'crossover_probability':0.5,\
                                       'deletion_probability':0.2,\
                                       'growth_probability':0.5,\
                                       'parents_portion':0.3,\
                                       'crossover_type':'uniform',\
                                       'max_iteration_without_improv':10,\
                                       'maxLen': 160, \
                                       'minLen': 20},\
                 convergence_curve=True,\
                 progress_bar=True):
        
        self.__name__ = geneticalgorithm_batch
        assert (callable(function)), "function must be callable"     
        self.f = function

        # Population and GA parameters
        self.param = algorithm_parameters
        self.pop_s = int(self.param['population_size'])
        self.mutation_prob = self.param['mutation_probability']
        self.crossover_prob = self.param['crossover_probability']
        self.deletion_prob = self.param['deletion_probability']
        self.growth_prob = self.param['growth_probability']
        self.maxLen = int(self.param['maxLen'])
        self.minLen = int(self.param['minLen'])
        
        self.iterate = int(self.param['max_num_iteration'])
        self.par_s = int(self.param['parents_portion'] * self.pop_s)
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s += 1

        self.new_gen_size = self.pop_s - self.par_s 
        # Make sure that these add up to the initial population size
        assert self.pop_s == self.new_gen_size + self.par_s, "Sizes do not add up to the initial population size."
    
        self.num_elit = int(self.pop_s * self.param['elit_ratio'])
        if self.num_elit < 1:
            self.num_elit = 1

        self.funtimeout = float(function_timeout)
        self.convergence_curve = convergence_curve
        self.progress_bar = progress_bar

        self.stop_mniwi = False
        if self.param['max_iteration_without_improv'] is None:
            self.mniwi = self.iterate + 1
        else:
            self.mniwi = int(self.param['max_iteration_without_improv'])
        
    def run(self, init_pop=None):
        # Initial Population
        #print('Starting GA...')
        pop = init_pop if init_pop is not None else self.initialize_population()
        pop = [ind[:] for ind in pop]  # Shallow copy of the population

        #print(len(pop))

        # Batch evaluation of the initial population
        sequences = [ind[:] for ind in pop]  # Shallow copy for sequences
        fitness_values = self.sim_batch(sequences) 
        #print(len(fitness_values))

        for i in range(self.pop_s):
            pop[i].append(fitness_values[i])  # Append fitness values to individuals
        
        # Report variables
        self.report = []
        self.best_function = min(fitness_values)  # Use the minimum fitness value
        self.best_variable = sequences[np.argmin(fitness_values)]  # Corresponding sequence
    
        t = 1
        counter = 0
        while t <= self.iterate:
            if self.progress_bar:
                self.progress(t, self.iterate, status="GA is running...")
    
            # Sort population by fitness
            pop_sort = sorted(pop, key=lambda x: x[-1] if x[-1] is not None else float('inf'))
    
            # Update best solution
            if pop_sort[0][-1] < self.best_function:  # First element is the 'best' after sorting
                counter = 0
                self.best_function = pop_sort[0][-1]
                self.best_variable = pop_sort[0][:-1]  # Shallow copy of the sequence
            else:
                counter += 1
    
            self.report.append(pop_sort[0][-1])
    
            # Apply Genetic Algorithm operations
            new_generation = []
            # for _ in range(int(0.7 * self.pop_s) // 2):
            for _ in range(self.new_gen_size // 2):
                # Select two parents
                parent1, parent2 = self.select_parents(pop_sort)
    
                # Apply genetic operations
                child1_seq, child2_seq = self.apply_moves(parent1[:-1][:], parent2[:-1][:])  # Shallow copy of sequence parts

                # No need to calculate fitness now, will batch evaluate later
                new_generation.append(child1_seq + [None])
                new_generation.append(child2_seq + [None])
            
            # If new_gen_size is odd, add one more child to reach the target size
            if len(new_generation) < self.new_gen_size:
                parent1, parent2 = self.select_parents(pop_sort)
                child1_seq, _ = self.apply_moves(parent1[:-1], parent2[:-1])  # Use only one child sequence
                new_generation.append(child1_seq + [None])
            
            # Replace population with new generation and top 30% of parents
            # pop = new_generation[:int(0.7 * self.pop_s)] + pop_sort[:self.par_s]
            pop = new_generation[:self.new_gen_size] + pop_sort[:self.par_s]

            # Verify population size
            #assert len(pop) == self.pop_s, f"Expected population size {self.pop_s}, but got {len(pop)}"
            #print(len(pop))
            
            # Batch evaluate fitness for the new generation
            fitness_values = self.sim_batch([ind[:-1] for ind in pop])
            #print(len(fitness_values))

            for i in range(self.pop_s):
                pop[i][-1] = fitness_values[i]  # Assign fitness value to individuals
    
            # Stop condition: if no improvement for too long
            if counter > self.mniwi:
                self.stop_mniwi = True
                break
    
            t += 1

        pop_sort = sorted(pop, key=lambda x: x[-1] if x[-1] is not None else float('inf'))
    
        # Update best solution
        if pop_sort[0][-1] < self.best_function:  # First element is the 'best' after sorting
            self.best_function = pop_sort[0][-1]
            self.best_variable = pop_sort[0][:-1]  # Shallow copy of the sequence

        self.report.append(pop_sort[0][-1])
    
        # Final Reporting
        #self.final_report(pop)

        
    def initialize_population(self):
        # Initialize random sequences for the population
        pop = []
        for _ in range(self.pop_s):
            N = np.random.randint(self.minLen, self.maxLen+1)  # Sequence length between 20 and 50
            seq = np.random.randint(0, 20, size=N).tolist()  # Random sequence of amino acids
            pop.append(seq)
        return pop

    def select_parents(self, pop_sort):
        # Select two parents from the top 30%
        top_parents = pop_sort[:int(self.par_s)] 
        parent1 = random.choice(top_parents)
        parent2 = random.choice(top_parents)
        return parent1, parent2
    
    def apply_moves(self, x, y):
        # Apply crossover, deletion, growth, and mutation moves based on probabilities
        if np.random.random() < self.crossover_prob:
            x, y = self.crossover(x, y)
        if np.random.random() < self.deletion_prob:
            x = self.deletion(x)
        if np.random.random() < self.deletion_prob:
            y = self.deletion(y)
        if np.random.random() < self.growth_prob:
            x = self.growth(x)
        if np.random.random() < self.growth_prob:
            y = self.growth(y)
        if np.random.random() < self.mutation_prob:
            x = self.mutate(x)
        if np.random.random() < self.mutation_prob:
            y = self.mutate(y)
        return x, y

    def crossover(self, x, y):
        N, M = len(x), len(y)
        s1, s2 = sorted(np.random.randint(1, min(N, M), size=2))  # Ensuring s1 < s2
        s3 = np.random.randint(1, max(N, M) - (s2 - s1))
        
        # Perform crossover
        if N < M:
            sub_x = x[s1:s2]
            sub_y = y[s3:s3 + (s2 - s1)]
            
            # Swap the sub-sequences
            x[s1:s2], y[s3:s3 + (s2 - s1)] = sub_y, sub_x
        else:
            sub_x = x[s3:s3 + (s2 - s1)]
            sub_y = y[s1:s2]
            
            # Swap the sub-sequences
            x[s3:s3 + (s2 - s1)], y[s1:s2] = sub_y, sub_x
        
        return x, y

    
    def deletion(self, seq, minLen=None):
        if minLen is None:
            minLen = self.minLen
        N = len(seq)
        if N > minLen:
            ldel = np.random.randint(0, N - minLen + 1)  # Length of deletion
            s = np.random.randint(0, N - ldel)  # Start of deletion
            seq = seq[:s] + seq[s + ldel:]  # Remove sub-sequence
        return seq

    def growth(self, seq, Lmax=None):
        if Lmax is None:
            Lmax = self.maxLen
        N = len(seq)
        if N < Lmax:
            lgro = np.random.randint(0, Lmax - N + 1)   # 0 .. Lmax-N
            if lgro == 0:
                return seq

            s = np.random.randint(0, N)                 # 0 .. N-1
            subseq = seq[:lgro]                          # replicate indices [1..lgro] (1-indexed)
            seq = seq[:s+1] + subseq + seq[s+1:]        # insert between s and s+1
        return seq

    
    def mutate(self, seq):
        N = len(seq)
        s = np.random.randint(0, N)  # Random mutation index
        old_residue = seq[s]
        choices = list(range(20)) # since we are using 'int'
        choices.remove(old_residue)  # Remove the current residue
        seq[s] = np.random.choice(choices) # unique mutation
        return seq
    
###############################################################################     
    def evaluate(self):
        return self.f(self.temp)
    
###############################################################################    
    def sim_batch(self, sequences):
        """
        Batch simulation of fitness function.
        """
        return self.f(sequences)
        
    def final_report(self, pop):
        best = min(pop, key=lambda x: x[-1])
        print(f"The best solution found: {best[:-1]}")
        print(f"Objective function: {best[-1]}")
        if self.convergence_curve:
            plt.plot(self.report)
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            plt.show()

    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))
        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write(f'\r{bar} {percents}% {status}')
        sys.stdout.flush()


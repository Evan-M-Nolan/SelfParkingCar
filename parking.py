from collections import namedtuple
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import math
from scipy import interpolate
from typing import List

Individual = List[int]
newChromosome = List[int]
stateHistory = namedtuple('state', ['cost','fitness', 'x','y','angle','velocity'])

#penis lol
#from joe

max_pop = 201
mutation_rate = .005
binary_param_size = 7
max_generations = 500

min_heading = -0.524
max_heading = 0.524
heading_range = max_heading - min_heading

min_accel = -5
max_accel = 5
accel_range = max_accel - min_accel


time_to_park_car = 10
time_step = 0.1
# List of the time steps to use for interpolation
all_time_steps_for_x = np.arange(0, time_to_park_car, time_step)
amt_control_variables = 2
total_entries = time_to_park_car * amt_control_variables
# Creates a new chromosome for a single time step

total_chromosomes = 10
length_of_binary_genome = amt_control_variables * binary_param_size * total_chromosomes
initial_state =  [0, 8, 0, 0]

final_state = np.array([0, 0, 0, 0])
K = 350
cost_for_being_out_of_bounds = 1
stop_criteria = 0.1

total_bits = binary_param_size * (total_entries - 2)  #subtract 2 for the initial 2 states

#used for the max value of a binary chromosome
max = pow(2, binary_param_size)-1

# Creates a new gray coded binary sequence
def newChromosome():
    chromosome = np.random.randint(0,2, length_of_binary_genome).tolist()
    return chromosome

# Take in an array of 7 values for a value sequence of a gene to convert it to decimal
def binaryToDecimal(binary_array):
    decimal = 0
    for i in range(len(binary_array) -1, -1, -1):
        decimal = decimal + binary_array[i] * pow(2, len(binary_array)- (i + 1))
    if binary_array[0] == 0:
        decimal *= -1
    return decimal

#flips a binary encoding
def flip(c):
    return 1 if(c == 0) else 0;

    #grey coding for speed
def graytoBinary(gray):
    binary = [];
    binary.append(gray[0])

    for i in range(1, len(gray)):
        if (gray[i] == 0):
            binary.append(binary[i - 1])
        else:
            binary.append(flip(binary[i - 1]))
    return binary;


#gets the heading
def convertHeading(binary_chromosome):
    oldVal = abs(binaryToDecimal(binary_chromosome))
    return ((oldVal * heading_range) / max) + min_heading

# Converts acceleration from the chromosome value
def convertAcceleration(binary_chromosome):
    oldVal = abs(binaryToDecimal(binary_chromosome))
    return ((oldVal * accel_range) / max) + min_accel

#gets the cost based on the final state
def costFunction(local_final_state, cf):
    if cf == 0:
        cost = np.linalg.norm( local_final_state - final_state)
    else:
        cost = K + cf
    return cost
def generateNewPopulation():
    population = []
    for i in range(0, max_pop, 1):
        population.append(newChromosome())
    return population

# a function for determining out of boundness
def isOutOfBounds(x, y):
    if x <= -4 and y > 3:
        return False
    elif (x >-4 and x< 4) and y > -1:
        return False
    elif x>= 4 and y > 3:
        return False
    return True

#Gets the y value given an out of bounds x
def getYValue(x):
    if x <= -4:
        return 3
    elif x > -4 and x < 4:
        return -1
    elif x >= 4:
        return 3
    return 0

def crossOver(pop1 :Individual, pop2 : Individual):
    # For every chromsome in each population
    cross_over_pivot = np.random.randint(0, len(pop1))
    second_pivot = np.random.randint(cross_over_pivot, len(pop1))
    new_child1 = pop2[0:cross_over_pivot] + pop1[cross_over_pivot:second_pivot] + pop2[second_pivot:]
    new_child2 = pop1[0:cross_over_pivot] + pop2[cross_over_pivot:second_pivot] + pop1[second_pivot:]
    new_c_1 = []
    new_c_2 = []

    for i in range(len(new_child1)):
        new_c_1.insert(i, random.choices(
            [new_child1[i], (new_child1[i] + 1 % 2)],
            weights=[(1 - mutation_rate), mutation_rate]
        )[0]
                          )
        new_c_2.insert(i, random.choices(
            [new_child2[i], (new_child2[i] + 1 % 2)],
            weights=[1 - mutation_rate, mutation_rate]
        )[0])
    return [new_c_1, new_c_2]
# Do crossover breeding
def newPopulationCrossover(chance_array, pops):
    new_chromeosomes = []
    for i in range(math.floor(max_pop/2)):

        new_indexs = np.random.choice(
            max_pop,
            2,
            p=chance_array
        )
        pop1 = pops[new_indexs[0]]
        pop2 = pops[new_indexs[1]]
        new_sequences = crossOver(pop1, pop2)
        new_chromeosomes.append(new_sequences[0])
        new_chromeosomes.append(new_sequences[1])

    return new_chromeosomes
def getHighRezParams(individual: Individual):
    chromesome = np.array_split(individual, total_entries)
    optimization_parameter = []
    acceleration_history = []
    heading_history = []
    for i in range(0, len(chromesome), 2):

        heading = convertHeading(chromesome[i])
        acceleration = convertAcceleration(chromesome[i+1])

        acceleration_history.append(acceleration)
        heading_history.append(heading)
        optimization_parameter.append(heading)
        optimization_parameter.append(acceleration)

    time = np.linspace(0, time_to_park_car, num=time_to_park_car, endpoint = True)
    s = interpolate.CubicSpline(time, acceleration_history, bc_type='natural')
    h = interpolate.CubicSpline(time, heading_history, bc_type='natural')
    x_new = np.linspace(0, 10, 100)
    highRezAccel = s(x_new)
    highRezHeading = h(x_new)
    return highRezAccel, highRezHeading, optimization_parameter

def convertChromesomeSequence(individual) -> stateHistory:
    x = []
    y = []

    current_x = initial_state[0]
    current_y = initial_state[1]
    current_velocity = initial_state[2]
    current_heading = initial_state[3]
    acceleration_new, heading_new, optimization_parameter = getHighRezParams(individual)
    local_cost = 0
    for timeStep in range(0, len(all_time_steps_for_x)):
        current_velocity += acceleration_new[timeStep] * time_step
        current_heading += heading_new[timeStep] * time_step
        current_x += current_velocity * math.cos(current_heading) * time_step 
        current_y += current_velocity * math.sin(current_heading) * time_step

        if isOutOfBounds(current_x, current_y):
            local_cost += math.pow(getYValue(current_x) - current_y, 2 ) * time_step
        x.append( current_x)
        y.append( current_y)
    local_final_state = np.array([current_x, current_y, current_heading, current_velocity])
    cost = costFunction(local_final_state, local_cost)
    state_history = stateHistory(cost,1/(cost+1),x,y,heading_new,acceleration_new)
    return {
        "cost": cost,
        "fitness": 1 / (cost+1),
        "x": x,
        "y": y,
        "acceleration": acceleration_new,
        "heading": heading_new,
        "optimization_vector": optimization_parameter
    }
# Basically converts an entire
def doPopulationGeneration(pops, generation):
    fitness_of_pops = []
    total_combined_fitness = 0
    most_fit_pop_index = 0
    for i in range(len(pops)):
        pop = pops[i]
        values = convertChromesomeSequence(pop) 
        total_combined_fitness += values["fitness"]
        fitness_of_pops.insert(i, values)
        if values["fitness"] > fitness_of_pops[most_fit_pop_index]["fitness"]:
            most_fit_pop_index = i
    print("Generation " +str(generation) +" : J = " + str(fitness_of_pops[most_fit_pop_index]["cost"]) )


    if fitness_of_pops[most_fit_pop_index]["cost"] <= stop_criteria or generation >= max_generations:
        return None, fitness_of_pops[most_fit_pop_index]

    chance_array = []
    for i in range(len(fitness_of_pops)):
        pop = fitness_of_pops[i]
        chance_array.insert(i, pop["fitness"] / total_combined_fitness)

    new_generation = newPopulationCrossover(chance_array, pops)
    new_generation.append(pops[most_fit_pop_index])
    return new_generation, fitness_of_pops[most_fit_pop_index]

#Prints to the file and does the graphs
def printer_grapher(solution):
    print("")
    print("Final state values:")
    print("x_f = " + str(solution["x"][-1]))
    print("y_f = " + str(solution["y"][-1]))
    print("alpha_f = " + str(solution["heading"][-1]))
    print("v_f = " + str(solution["acceleration"][-1]))

    figure, axis = plt.subplots(5, 1, figsize=(4, 2 * 5), tight_layout= True)

    axis[0].set_xlim([-15, 15])
    axis[0].set_ylim([-10, 15])
    axis[0].plot([-15, -4], [3, 3], 'k-', lw=1)
    axis[0].plot([-4, -4], [-1, 3], 'k-', lw=1)
    axis[0].plot([-4, 4], [-1, -1], 'k-', lw=1)
    axis[0].plot([4, 4], [-1, 3], 'k-', lw=1)
    axis[0].plot([15, 4], [3, 3], 'k-', lw=1)
    axis[0].plot(solution["x"], solution["y"])
    axis[0].plot([-15, -4], [3, 3], 'k-', lw=1)
    axis[0].plot([-4, -4], [-1, 3], 'k-', lw=1)
    axis[0].plot([-4, 4], [-1, -1], 'k-', lw=1)
    axis[0].plot([4, 4], [-1, 3], 'k-', lw=1)
    axis[0].plot([15, 4], [3, 3], 'k-', lw=1)
    plt.setp(axis[0], xlabel="x ", ylabel="y ")

    axis[1].plot(all_time_steps_for_x, solution["acceleration"])
    plt.setp(axis[1], xlabel="Time (s)", ylabel="Acceleration (ft/s^2)")

    axis[2].plot(all_time_steps_for_x, solution["heading"])
    plt.setp(axis[2], xlabel="Time (s)", ylabel="heading (rad/s^2)")

    axis[3].plot(all_time_steps_for_x, solution["x"])
    plt.setp(axis[3], xlabel="Time (s)", ylabel="x (ft)")

    axis[4].plot(all_time_steps_for_x, solution["y"])
    plt.setp(axis[4], xlabel="Time (s)", ylabel="y (ft)")

    plt.show()



def main():
    start_time = time.time()
    generations = 0
    current_population = generateNewPopulation()
    while (current_population != None and time.time() -start_time < 420):
        current_population, most_fit = doPopulationGeneration(current_population, generations)
        generations += 1

    printer_grapher(most_fit)
    file = open('control.dat', 'w')
    for val in most_fit["optimization_vector"]:
        file.write(str(val)+"\n")
    file.close()

if __name__ == "__main__":
    main()

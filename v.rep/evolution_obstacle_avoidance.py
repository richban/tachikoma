import random
import vrep
import numpy as np
import time
from deap import base, creator, tools, algorithms
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import networkx
import argparse


from robot import EvolvedRobot
from eaplots import plot_single_run

# VREP
PORT_NUM = 19997
OP_MODE = vrep.simx_opmode_oneshot_wait
RUNTIME = 24

# GENOME TYPE
MINMAX = 5
MIN = -3
MAX = 3

# EVOLUTION
POPULATION = 2
N_GENERATIONS = 2
# CXPB  is the probability with which two individuals
# are crossed
#
# MUTPB is the probability for mutating an individual
CXPB = 0.5
MUTPB = 0.2

def evolution_obstacle_avoidance():
    print('Evolutionary program started!')
    # Just in case, close all opened connections
    vrep.simxFinish(-1)

    client_id = vrep.simxStart(
        '127.0.0.1',
        PORT_NUM,
        True,
        True,
        5000,
        5)  # Connect to V-REP

    if client_id == -1:
        print('Failed connecting to remote API server')
        print('Program ended')
        return

    print('Connected to remote API server')


    robot = EvolvedRobot([], client_id=client_id, id=None, op_mode=OP_MODE)

    # Creating the appropriate type of the problem
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Deap Initialization
    toolbox = base.Toolbox()

    history = tools.History()

    # Attribute generator random
    toolbox.register("attr_int", random.randint, MIN, MAX)
    # Structure initializers; instantiate an individual or population
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_int,
        n=robot.chromosome_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("map", map)

    def eval_robot(individual):

        if (vrep.simxStartSimulation(client_id, OP_MODE) == -1):
            print('Failed to start the simulation\n')
            print('Program ended\n')
            return

        print("Starting simulation")

        individual = EvolvedRobot(
            individual,
            client_id=client_id,
            id=None,
            op_mode=OP_MODE)

        start_position = None
        fitness_t = np.array([])

        # collistion detection initialization
        errorCode, collision_handle = vrep.simxGetCollisionHandle(
            client_id, "robot_collision", vrep.simx_opmode_blocking)
        collision = False
        first_collision_check = True

        now = datetime.now()

        while not collision and datetime.now() - now < timedelta(seconds=RUNTIME):

            if start_position is None:
                start_position = individual.position

            individual.loop()

            # print(individual)

            fitness_t = np.append(fitness_t,
                ((individual.norm_wheel_speeds[0] +
                 individual.norm_wheel_speeds[1]) / 2) *
                (1 - (np.sqrt(np.absolute(
                    individual.norm_wheel_speeds[0] -
                    individual.norm_wheel_speeds[1])))) *
                (np.absolute(np.amin(individual.sensor_activation - 1))))


            # collision detection
            if first_collision_check:
                collision_mode = vrep.simx_opmode_streaming
            else:
                collision_mode = vrep.simx_opmode_buffer

            collisionDetected, collision = vrep.simxReadCollision(
                client_id, collision_handle, collision_mode)
            first_collision_check = False

        # Fitness
        fitness = [np.sum(fitness_t)]

        print(
            "Finished simulation. Went from [%f,%f] to [%f,%f] with fitness: %f" %
            (start_position[0],
             start_position[1],
             individual.position[0],
             individual.position[1],
             fitness[0]))

        if (vrep.simxStopSimulation(client_id, OP_MODE) == -1):
            print('Failed to stop the simulation\n')
            print('Program ended\n')
            return

        time.sleep(1)

        return fitness

    # Register genetic operators
    # register the goal / fitness function
    toolbox.register("evaluate", eval_robot)
    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    # register a mutation operator with a probability to
    # flip each attribute/gene
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTPB)
    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Decorate the variation operators
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    # instantiate the population
    # create an initial population of N individuals
    pop = toolbox.population(n=POPULATION)
    history.update(pop)

    # object that contain the best individuals
    hof = tools.HallOfFame(20)
    # maintain stats of the evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # very basic evolutianry algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=N_GENERATIONS,
                                   stats=stats, halloffame=hof, verbose=True)

    graph = networkx.DiGraph(history.genealogy_tree)
    graph = graph.reverse()     # Make the grah top-down
    colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    networkx.draw(graph, node_color=colors)
    plt.savefig('../images/genealogy_tree.pdf')


    # Evolution records as a chronological list of dictionaries
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_avgs = log.select("avg")
    fit_maxs = log.select("max")

    save_date = datetime.now().strftime('%m-%d-%H-%M')

    plot_single_run(
        gen,
        fit_mins,
        fit_avgs,
        fit_maxs,
        ratio=0.35,
        save='../images/'+save_date+'-evolved-obstacle.pdf')

    if (vrep.simxFinish(client_id) == -1):
        print('Evolutionary program failed to exit\n')
        return

evolution_obstacle_avoidance()

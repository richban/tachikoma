import random
import vrep
import numpy as np
from deap import base, creator, tools, algorithms
from datetime import datetime, timedelta

from robot import Robot, EvolvedRobot, avoid_obstacles

MINMAX = 5
PORT_NUM = 20010
POPULATION = 300
RUNTIME = 3
OP_MODE = vrep.simx_opmode_oneshot_wait


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

    robot = EvolvedRobot([], client_id=client_id, id=None, op_mode=OP_MODE)

    # Creating the appropriate type of the problem
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Deap Initialization
    toolbox = base.Toolbox()
    # Attribute generator random integer between MINMAX
    toolbox.register("attr_int", random.randint, -MINMAX, MINMAX)
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
        print("Starting simulation: %s" % str(individual))

        vrep.simxStartSimulation(client_id, OP_MODE)

        individual = EvolvedRobot(
            individual,
            client_id=client_id,
            id=None,
            op_mode=OP_MODE)

        now = datetime.now()
        start_position = None

        while datetime.now() - now < timedelta(seconds=RUNTIME):
            if start_position is None:
                start_position = individual.position

            individual.loop()

        # Fitness
        fitness = [np.array(individual.position)[0] -
                   np.array(start_position)[0] -
                   abs(np.array(individual.position)[1] -
                       np.array(start_position)[1]), ]

        print(
            "Finished simulation. Went from [%f,%f] to [%f,%f] with fitness: %f" %
            (start_position[0],
             start_position[1],
             individual.position[0],
             individual.position[1],
             fitness[0]))

        vrep.simxStopSimulation(client_id, OP_MODE)

        return fitness

    # Register genetic operators
    # register the goal / fitness function
    toolbox.register("evaluate", eval_robot)
    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)

    # instantiate the population
    # create an initial population of 300 individuals
    pop = toolbox.population(n=POPULATION)
    # object that contain the best individuals
    hof = tools.HallOfFame(1)
    # maintain stats of the evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # very basic evolutianry algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
                                   stats=stats, halloffame=hof, verbose=True)

    # Evolution records as a chronological list of dictionaries
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_avgs = log.select("avg")
    fit_maxs = log.select("max")

    vrep.simxFinish(client_id)


evolution_obstacle_avoidance()

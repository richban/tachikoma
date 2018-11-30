import random
import vrep
import numpy as np
import time
import os
from deap import base, creator, tools, algorithms
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import networkx
from robot import EvolvedRobot
from eaplots import plot_single_run
from helpers import f_wheel_center, f_straight_movements, f_pain
from argparse import ArgumentParser
import math
import pickle

# VREP
PORT_NUM = 19997
DEBUG = False
OP_MODE = vrep.simx_opmode_oneshot_wait
PATH = './data/ea/' + datetime.now().strftime('%Y-%m-%d') + '/'
MIN = 0.0
MAX = 3.0

if not os.path.exists(PATH):
    os.makedirs(PATH)


def evolution_obstacle_avoidance(args):
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

    # Evolution
    RUNTIME = args.time
    POPULATION = args.pop
    N_GENERATIONS = args.n_gen
    # CXPB  is the probability with which two individuals are crossed
    CXPB = args.cxpb # 0.1
    # MUTPB is the probability for mutating an individual
    MUTPB = args.mutpb # 0.2

    # save the config
    dump_config(POPULATION, N_GENERATIONS, RUNTIME, CXPB, MUTPB)

    # Creating the appropriate type of the problem
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    # Deap Initialization
    toolbox = base.Toolbox()

    history = tools.History()

    # Attribute generator random
    toolbox.register('attr_float', random.uniform, MIN, MAX)
    # Structure initializers; instantiate an individual or population
    toolbox.register(
        'individual',
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=robot.chromosome_size)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    def mapN(function, iterable, n=4):
        i = [iterable[i*n:i*n+n] for i in range(int(len(iterable) / n))]
        mapped = map(function, i)
        ext_map = []
        for m in mapped: ext_map.extend(m)
        print(ext_map)
        return ext_map

    toolbox.register("map", mapN)

    def eval_robot(individual):

        if (vrep.simxStartSimulation(client_id, OP_MODE) == -1):
            print('Failed to start the simulation\n')
            print('Program ended\n')
            return


        individuals = [
            EvolvedRobot(individual[0], id=None, client_id=client_id, op_mode=OP_MODE),
            EvolvedRobot(individual[1], id=0, client_id=client_id, op_mode=OP_MODE),
            EvolvedRobot(individual[2], id=1, client_id=client_id, op_mode=OP_MODE),
            EvolvedRobot(individual[3], id=2, client_id=client_id, op_mode=OP_MODE),
        ]

        start_position = None
        fitness_aff = [[1.0,] for i in range(4)]

        collision_handlers = []
        first_collision_check = []
        collisions = []
        collision_mode = [None, None, None, None]

        for i, ind in enumerate(individuals):
            errorCode, collision_handler = vrep.simxGetCollisionHandle(client_id, 'robot_collision_{}'.format(i), vrep.simx_opmode_oneshot_wait)
            collision_handlers.append(collision_handler)
            collisions.append(False)
            first_collision_check.append(True)

        now = datetime.now()

        if DEBUG: individual.logger.info('Chromosome {}'.format(individual.chromosome))

        while datetime.now() - now < timedelta(seconds=RUNTIME):

            for i, ind in enumerate(individuals):
                ind.loop()

            # Collision Mode
            for i, ind in enumerate(individuals):
                if first_collision_check[i]:
                    collision_mode[i] = vrep.simx_opmode_streaming
                else:
                    collision_mode[i] = vrep.simx_opmode_buffer

            for i, ind in enumerate(individuals):
                if not collisions[i]:
                    collisionDetected, collision = vrep.simxReadCollision(client_id, collision_handlers[i], collision_mode[i])
                    collisions[i] = collision
                    first_collision_check[i] = False
                else:
                    print(ind.id)

            for i, ind in enumerate(individuals):
                if not collisions[i]:
                    # Fitness function; each feature;
                    # V - wheel center
                    V = f_wheel_center(ind.norm_wheel_speeds[0], ind.norm_wheel_speeds[1])
                    if DEBUG: ind.logger.info('f_wheel_center {}'.format(V))

                    # pleasure - straight movements
                    pleasure = f_straight_movements(ind.norm_wheel_speeds[0], ind.norm_wheel_speeds[1])
                    if DEBUG: ind.logger.info('f_straight_movements {}'.format(pleasure))

                    # pain - closer to an obstacle more pain
                    pain = f_pain(ind.sensor_activation)
                    if DEBUG: ind.logger.info('f_pain {}'.format(pain))

                    #  fitness_t at time stamp
                    fitness_t = V * pleasure * pain
                    fitness_aff[i] += [fitness_t, ]
                else:
                    fitness_aff[i] += [0.0, ]


        fitness = [sum(f) for f in fitness_aff]
        
        print('%s with fitness: %s ' % (str(id), str(fitness)))

        if (vrep.simxStopSimulation(client_id, OP_MODE) == -1):
            print('Failed to stop the simulation\n')
            print('Program ended\n')
            return

        time.sleep(1)

        return fitness

    # Register genetic operators
    # register the goal / fitness function
    toolbox.register('evaluate', eval_robot)
    # register the crossover operator
    toolbox.register('mate', tools.cxTwoPoint)
    # register a mutation operator with a probability to
    # flip each attribute/gene
    toolbox.register('mutate', tools.mutFlipBit, indpb=MUTPB)
    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register('select', tools.selTournament, tournsize=3)

    # Decorate the variation operators
    toolbox.decorate('mate', history.decorator)
    toolbox.decorate('mutate', history.decorator)

    # instantiate the population
    # create an initial population of N individuals
    pop = toolbox.population(n=POPULATION)
    history.update(pop)

    # object that contain the best individuals
    hof = tools.HallOfFame(20)
    # maintain stats of the evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    # very basic evolutianry algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=N_GENERATIONS,
                                   stats=stats, halloffame=hof, verbose=True)

    # plot the best individuals genealogy
    gen_best = history.getGenealogy(hof[0])
    graph = networkx.DiGraph(gen_best).reverse()
    colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    networkx.draw(graph, node_color=colors, node_size=100)
    plt.savefig(PATH+'genealogy_tree.pdf')

    # log Statistics
    with open(PATH+'ea_statistics.txt', 'w') as s:
        s.write(log.__str__())

    # save the best genome
    with open(PATH+'best', 'wb') as fp:
        pickle.dump(hof, fp)

    # Evolution records as a chronological list of dictionaries
    gen = log.select('gen')
    fit_mins = log.select('min')
    fit_avgs = log.select('avg')
    fit_maxs = log.select('max')

    save_date = datetime.now().strftime('%m-%d-%H-%M')

    plot_single_run(
        gen,
        fit_mins,
        fit_avgs,
        fit_maxs,
        ratio=0.35,
        save=PATH+'evolved-obstacle.pdf')

    if (vrep.simxFinish(client_id) == -1):
        print('Evolutionary program failed to exit\n')
        return


def dump_config(pop, n_gen, time, cxpb, mutpb):
    with open(PATH+'ea_config.txt', 'a') as f:
        f.write('Poluation size: {0}\nNumber of generations: {1}\n'
            'Simulation Time: {2}\nCrossover: {3}\nMutation: {4}\n' \
            .format(pop, n_gen, time, cxpb, mutpb))

if __name__ == '__main__':
    parser = ArgumentParser(description='Help me throughout the evolution')
    parser.add_argument('--pop', type=int, help='population size')
    parser.add_argument('--n_gen', type=int, help='number of generations')
    parser.add_argument('--time', type=int, help='running time of one epoch')
    parser.add_argument('--cxpb', type=float, help='the probability with which two individuals are crossed')
    parser.add_argument('--mutpb', type=float, help='the probability for mutating an individual')
    args = parser.parse_args()

    evolution_obstacle_avoidance(args)

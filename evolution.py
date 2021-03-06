import random
import vrep
import numpy as np
import time
import uuid
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
import settings
from functools import partial

settings.init()

if not os.path.exists(settings.PATH_EA):
    os.makedirs(settings.PATH_EA)

def eval_robot(robot, chromosome):

    # Enable the synchronous mode
    vrep.simxSynchronous(settings.CLIENT_ID, True)

    if (vrep.simxStartSimulation(settings.CLIENT_ID, settings.OP_MODE) == -1):
        print('Failed to start the simulation\n')
        print('Program ended\n')
        return

    robot.chromosome = chromosome
    individual = robot
    
    start_position = None
    fitness_agg = np.array([])

    # collistion detection initialization
    errorCode, collision_handle = vrep.simxGetCollisionHandle(
        settings.CLIENT_ID, 'robot_collision', vrep.simx_opmode_oneshot_wait)
    collision = False

    now = datetime.now()
    id = uuid.uuid1()

    if start_position is None:
        start_position = individual.position

    distance_acc = 0.0
    previous = np.array(start_position)

    collisionDetected, collision = vrep.simxReadCollision(
        settings.CLIENT_ID, collision_handle, vrep.simx_opmode_streaming)


    if settings.DEBUG: individual.logger.info('Chromosome {}'.format(individual.chromosome))

    while not collision and datetime.now() - now < timedelta(seconds=settings.RUNTIME):

        # The first simulation step waits for a trigger before being executed
        # start_time = time.time()
        vrep.simxSynchronousTrigger(settings.CLIENT_ID)

        collisionDetected, collision = vrep.simxReadCollision(
            settings.CLIENT_ID, collision_handle, vrep.simx_opmode_buffer)

        individual.loop()

        # # Traveled distance calculation
        # current = np.array(individual.position)
        # distance = math.sqrt(((current[0] - previous[0])**2) + ((current[1] - previous[1])**2))
        # distance_acc += distance
        # previous = current

        # After this call, the first simulation step is finished
        vrep.simxGetPingTime(settings.CLIENT_ID)

        # Fitness function; each feature;
        # V - wheel center
        V = f_wheel_center(individual.norm_wheel_speeds[0], individual.norm_wheel_speeds[1])
        if settings.DEBUG: individual.logger.info('f_wheel_center {}'.format(V))

        # pleasure - straight movements
        pleasure = f_straight_movements(individual.norm_wheel_speeds[0], individual.norm_wheel_speeds[1])
        if settings.DEBUG: individual.logger.info('f_straight_movements {}'.format(pleasure))

        # pain - closer to an obstacle more pain
        pain = f_pain(individual.sensor_activation)
        if settings.DEBUG: individual.logger.info('f_pain {}'.format(pain))

        #  fitness_t at time stamp
        fitness_t = V * pleasure * pain
        fitness_agg = np.append(fitness_agg, fitness_t)
        
        # elapsed_time = time.time() - start_time

        # dump individuals data
        if settings.DEBUG:
            with open(settings.PATH_EA + str(id) + '_fitness.txt', 'a') as f:
                f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8}, {9}\n'.format(id, individual.wheel_speeds[0],
                individual.wheel_speeds[1], individual.norm_wheel_speeds[0], individual.norm_wheel_speeds[1], V, pleasure, pain, fitness_t, distance_acc))


    # aggregate fitness function
    # fitness_aff = [distance_acc]

    # behavarioral fitness function
    fitness_bff = [np.sum(fitness_agg)]

    # tailored fitness function
    fitness = fitness_bff[0] # * fitness_aff[0]

    # Now send some data to V-REP in a non-blocking fashion:
    vrep.simxAddStatusbarMessage(settings.CLIENT_ID, 'fitness: {}'.format(fitness), vrep.simx_opmode_oneshot)

    # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    vrep.simxGetPingTime(settings.CLIENT_ID)

    print('%s fitness: %f | fitness_bff %f | fitness_aff %f' % (str(id), fitness, fitness_bff[0], 0.0)) # , fitness_aff[0]))

    # save individual as object
    if settings.DEBUG:
        if fitness > 30:
            individual.save_robot(settings.PATH_EA + str(id) + '_robot')

    if (vrep.simxStopSimulation(settings.CLIENT_ID, settings.OP_MODE) == -1):
        print('Failed to stop the simulation\n')
        print('Program ended\n')
        return

    time.sleep(1)

    return [fitness]

def evolution_obstacle_avoidance():
    print('Evolutionary program started!')
    # Just in case, close all opened connections
    vrep.simxFinish(-1)

    settings.CLIENT_ID = vrep.simxStart(
        '127.0.0.1',
        settings.PORT_NUM,
        True,
        True,
        5000,
        5)  # Connect to V-REP

    if settings.CLIENT_ID == -1:
        print('Failed connecting to remote API server')
        print('Program ended')
        return

    print('Connected to remote API server')

    robot = EvolvedRobot(
        None,
        client_id=settings.CLIENT_ID,
        id=None,
        op_mode=settings.OP_MODE)

    dump_config(settings.POPULATION,
                settings.N_GENERATIONS,
                settings.RUNTIME,
                settings.CXPB,
                settings.MUTPB)

    # Creating the appropriate type of the problem
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    # Deap Initialization
    toolbox = base.Toolbox()

    history = tools.History()

    # Attribute generator random
    toolbox.register('attr_float', random.uniform, settings.MIN, settings.MAX)
    # Structure initializers; instantiate an individual or population
    toolbox.register(
        'individual',
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=robot.chromosome_size)

    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('map', map)

    # Register genetic operators
    # register the goal / fitness function
    toolbox.register('evaluate', partial(eval_robot, robot))
    # register the crossover operator
    toolbox.register('mate', tools.cxTwoPoint)
    # register a mutation operator with a probability to
    # flip each attribute/gene
    toolbox.register('mutate', tools.mutFlipBit, indpb=settings.MUTPB)
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
    pop = toolbox.population(n=settings.POPULATION)
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
    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=settings.CXPB,
                                   mutpb=settings.MUTPB,
                                   ngen=settings.N_GENERATIONS,
                                   stats=stats,
                                   halloffame=hof,
                                   verbose=True)

    # plot the best individuals genealogy
    gen_best = history.getGenealogy(hof[0])
    graph = networkx.DiGraph(gen_best).reverse()
    colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    networkx.draw(graph, node_color=colors, node_size=100)
    plt.savefig(settings.PATH_EA + 'genealogy_tree.pdf')

    # log Statistics
    with open(settings.PATH_EA + 'ea_statistics.txt', 'w') as s:
        s.write(log.__str__())

    # save the best genome
    with open(settings.PATH_EA + 'best.pkl', 'wb') as fp:
        pickle.dump(hof, fp)

    # Evolution records as a chronological list of dictionaries
    gen = log.select('gen')
    fit_mins = log.select('min')
    fit_avgs = log.select('avg')
    fit_maxs = log.select('max')
    
    plot_single_run(
        gen,
        fit_mins,
        fit_avgs,
        fit_maxs,
        ratio=0.35,
        save=settings.PATH_EA + 'evolved-obstacle.pdf')

    if (vrep.simxFinish(settings.CLIENT_ID) == -1):
        print('Evolutionary program failed to exit\n')
        return

def dump_config(pop, n_gen, time, cxpb, mutpb):
    with open(settings.PATH_EA + 'ea_config.txt', 'w') as f:
        f.write('Poluation size: {0}\nNumber of generations: {1}\n'
            'Simulation Time: {2}\nCrossover: {3}\nMutation: {4}\n' \
            .format(pop, n_gen, time, cxpb, mutpb))

if __name__ == '__main__':
    # Start Evolution
    evolution_obstacle_avoidance()

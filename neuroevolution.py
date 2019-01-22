from __future__ import print_function
import os
import neat
import vrep
import numpy as np
from datetime import datetime, timedelta
import time
import uuid
import visualize
from robot import EvolvedRobot
from helpers import f_wheel_center, f_straight_movements, f_pain, scale
import math
from argparse import ArgumentParser
import configparser
import settings


settings.init()

if not os.path.exists(settings.PATH_NE):
    os.makedirs(settings.PATH_NE)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        # Enable the synchronous mode
        vrep.simxSynchronous(settings.CLIENT_ID, True)

        if (vrep.simxStartSimulation(settings.CLIENT_ID, vrep.simx_opmode_oneshot) == -1):
            print('Failed to start the simulation\n')
            print('Program ended\n')
            return

        individual = EvolvedRobot(
            genome,
            client_id=settings.CLIENT_ID,
            id=None,
            op_mode=settings.OP_MODE)

        start_position = None
        # collistion detection initialization
        errorCode, collision_handle = vrep.simxGetCollisionHandle(
            settings.CLIENT_ID, 'robot_collision', vrep.simx_opmode_blocking)
        collision = False
        first_collision_check = True

        now = datetime.now()
        fitness_agg = np.array([])
        scaled_output = np.array([])
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        id = uuid.uuid1()

        if start_position is None:
            start_position = individual.position

        distance_acc = 0.0
        pp = np.array(start_position)
        p = np.array([])

        collisionDetected, collision = vrep.simxReadCollision(
            settings.CLIENT_ID, collision_handle, vrep.simx_opmode_streaming)

        while not collision and datetime.now() - now < timedelta(seconds=settings.RUNTIME):

            # The first simulation step waits for a trigger before being executed
            vrep.simxSynchronousTrigger(settings.CLIENT_ID)

            collisionDetected, collision = vrep.simxReadCollision(
                settings.CLIENT_ID, collision_handle, vrep.simx_opmode_buffer)

            individual.neuro_loop()

            # Traveled distance calculation
            # p = np.array(individual.position)
            # d = math.sqrt(((p[0] - pp[0])**2) + ((p[1] -pp[1])**2))
            # distance_acc += d
            # pp = p

            output = net.activate(individual.sensor_activation)
            # normalize motor wheel wheel_speeds [0.0, 2.0] - robot
            scaled_output = np.array([scale(xi, 0.0, 2.0) for xi in output])

            if settings.DEBUG: individual.logger.info('Wheels {}'.format(scaled_output))

            individual.set_motors(*list(scaled_output))

            # After this call, the first simulation step is finished
            vrep.simxGetPingTime(settings.CLIENT_ID)
            # Now we can safely read all streamed values

            # Fitness function; each feature;
            # V - wheel center
            V = f_wheel_center(output[0], output[1])
            if settings.DEBUG: individual.logger.info('f_wheel_center {}'.format(V))

            # pleasure - straight movements
            pleasure = f_straight_movements(output[0], output[1])
            if settings.DEBUG: individual.logger.info('f_straight_movements {}'.format(pleasure))

            # pain - closer to an obstacle more pain
            pain = f_pain(individual.sensor_activation)
            if settings.DEBUG: individual.logger.info('f_pain {}'.format(pain))

            #  fitness_t at time stamp
            fitness_t = V * pleasure * pain
            fitness_agg = np.append(fitness_agg, fitness_t)


            # dump individuals data
            if settings.DEBUG:
                with open(settings.PATH_NE + str(id) + '_fitness.txt', 'a') as f:
                    f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8}\n'.format(id, scaled_output[0], scaled_output[1], output[0], output[1], V, pleasure, pain, fitness_t))


        # errorCode, distance = vrep.simxGetFloatSignal(CLIENT_ID, 'distance', vrep.simx_opmode_blocking)
        # aggregate fitness function - traveled distance
        # fitness_aff = [distance_acc]

        # behavarioral fitness function
        fitness_bff = [np.sum(fitness_agg)]

        # tailored fitness function
        fitness = fitness_bff[0] # * fitness_aff[0]

        # Now send some data to V-REP in a non-blocking fashion:
        vrep.simxAddStatusbarMessage(settings.CLIENT_ID, 'fitness: {}'.format(fitness), vrep.simx_opmode_oneshot)

        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        vrep.simxGetPingTime(settings.CLIENT_ID)

        # print('%s with fitness: %f and distance %f' % (str(genome_id), fitness, fitness_aff[0]))

        if (vrep.simxStopSimulation(settings.CLIENT_ID, settings.OP_MODE) == -1):
            print('Failed to stop the simulation\n')
            print('Program ended\n')
            return

        time.sleep(1)
        genome.fitness = fitness

def run(config_file, args):
    print('Neuroevolutionary program started!')
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

    settings.N_GENERATIONS = args.n_gen
    settings.RUNTIME = args.time
    settings.DEBUG = True
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # save the current configuration
    config.save(settings.PATH_NE + 'config.ini')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    # Run for up to N_GENERATIONS generations.
    winner = p.run(eval_genomes, settings.N_GENERATIONS)

    # Write run statistics to file.
    stats.save_genome_fitness(filename=settings.PATH_NE+'fitnesss_history.csv')
    stats.save_species_count(filename=settings.PATH_NE+'speciation.csv')
    stats.save_species_fitness(filename=settings.PATH_NE+'species_fitness.csv')

    # log the winner network
    with open(settings.PATH_NE + 'winner_network.txt', 'w') as s:
        s.write('\nBest genome:\n{!s}'.format(winner))
        s.write('\nBest genomes:\n{!s}'.format(print(stats.best_genomes(5))))

    # save the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', -5: 'E',
                  -6: 'F', -7: 'G', -8: 'H', -9: 'I', -10: 'J',
                  -11: 'K', -12: 'L', -13: 'M', -14: 'N', -15: 'O',
                  -16: 'P', 0: 'LEFT', 1: 'RIGHT', }

    visualize.draw_net(config, winner, True, node_names=node_names, filename=settings.PATH_NE+'network')

    visualize.plot_stats(stats, ylog=False, view=False, filename=settings.PATH_NE+'feedforward-fitness.svg')
    visualize.plot_species(stats, view=False, filename=settings.PATH_NE+'feedforward-speciation.svg')

    visualize.draw_net(config, winner, view=False, node_names=node_names,
                           filename=settings.PATH_NE+'winner-feedforward.gv')
    visualize.draw_net(config, winner, view=False, node_names=node_names,
                        filename=settings.PATH_NE+'winner-feedforward-enabled.gv', show_disabled=False)
    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename=settings.PATH_NE+'winner-feedforward-enabled-pruned.gv', show_disabled=False, prune_unused=False)


if __name__ == '__main__':
    # Determine path to configuration file.
    parser = ArgumentParser(description='Help me throughout the evolution')
    parser.add_argument('--n_gen', type=int, help='number of generations')
    parser.add_argument('--time', type=int, help='running time of one epoch')
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')

    run(config_path, args)

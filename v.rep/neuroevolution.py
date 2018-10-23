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

OP_MODE = vrep.simx_opmode_oneshot_wait
PORT_NUM = 19997
RUNTIME = 20
N_GENERATIONS = 20
WHEEL_SPEED_SCALE = 16
DEBUG = False
global client_id

PATH = './data/neat/' + datetime.now().strftime("%Y-%m-%d") + '/'


if not os.path.exists(PATH):
    os.makedirs(PATH)


def run(config_file):
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

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:

            if (vrep.simxStartSimulation(client_id, OP_MODE) == -1):
                print('Failed to start the simulation\n')
                print('Program ended\n')
                return

            individual = EvolvedRobot(
                genome,
                client_id=client_id,
                id=None,
                op_mode=OP_MODE)

            start_position = None
            # collistion detection initialization
            errorCode, collision_handle = vrep.simxGetCollisionHandle(
                client_id, "robot_collision", vrep.simx_opmode_oneshot_wait)
            collision = False
            first_collision_check = True

            now = datetime.now()
            fitness_agg = np.array([])
            scaled_output = np.array([])

            net = neat.nn.FeedForwardNetwork.create(genome, config)
            id = uuid.uuid1()

            while not collision and datetime.now() - now < timedelta(seconds=RUNTIME):
                if start_position is None:
                    start_position = individual.position

                individual.neuro_loop()

                if first_collision_check:
                    collision_mode = vrep.simx_opmode_streaming
                else:
                    collision_mode = vrep.simx_opmode_buffer

                collisionDetected, collision = vrep.simxReadCollision(
                    client_id, collision_handle, collision_mode)
                first_collision_check = False

                output = net.activate(individual.sensor_activation)
                scaled_output = np.array([scale(xi, -2, 2) for xi in output])

                if DEBUG: individual.logger.info(f'Wheels {scaled_output}')

                individual.set_motors(*list(scaled_output))

                # Fitness function; each feature;
                # V - wheel center
                V = f_wheel_center(output[0], output[1])
                if DEBUG: individual.logger.info(f'f_wheel_center {V}')

                # pleasure - straight movements
                pleasure = f_straight_movements(output[0], output[1])
                if DEBUG: individual.logger.info(f'f_straight_movements {pleasure}')

                # pain - closer to an obstacle more pain
                pain = f_pain(individual.sensor_activation)
                if DEBUG: individual.logger.info(f'f_pain {pain}')

                #  fitness_t at time stamp
                fitness_t = V * pleasure * pain
                fitness_agg = np.append(fitness_agg, fitness_t)

                with open(PATH + str(id) + "_fitness.txt", "a") as f:
                    f.write(
                        f"{str(id)},{output[0]},{output[1]},{scaled_output[0]},{scaled_output[1]},{V},{pleasure},{pain},{fitness_t} \n")


            # aggregate fitness function - euclidean distance
            fitness_aff = [np.sqrt(abs(np.array(individual.position)[0] -
                            np.array(start_position)[0])**2 +
                            abs(np.array(individual.position)[1] -
                            np.array(start_position)[1])**2), ]

            # behavarioral fitness function
            fitness_bff = [np.sum(fitness_agg)]

            # tailored fitness function
            fitness = fitness_bff[0] # * fitness_aff[0]

            # Now send some data to V-REP in a non-blocking fashion:
            vrep.simxAddStatusbarMessage(client_id, f"fitness: {fitness}", vrep.simx_opmode_oneshot)

            # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
            vrep.simxGetPingTime(client_id)

            print("%s with fitness: %f and distance %f" % (str(id), fitness, fitness_aff[0]))

            if (vrep.simxStopSimulation(client_id, OP_MODE) == -1):
                print('Failed to stop the simulation\n')
                print('Program ended\n')
                return

            time.sleep(1)
            genome.fitness = fitness

    # Run for up to N_GENERATIONS generations.
    winner = p.run(eval_genomes, N_GENERATIONS)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', -5: 'E',
                  -6: 'F', -7: 'G', -8: 'H', -9: 'I', -10: 'J',
                  -11: 'K', -12: 'L', -13: 'M', -14: 'N', -15: 'O',
                  -16: 'P', 0: 'LEFT', 1: 'RIGHT', }
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')
    run(config_path)

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
from neuroevolution import eval_genomes
import settings


def run(config_file, args):
    print('Neuroevolutionary program started!')
    # Just in case, close all opened connections
    vrep.simxFinish(-1)

    settings.init()

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
    settings.DEBUG = False
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    restored_population = neat.Checkpointer.restore_checkpoint(args.checkpoint)
    restored_population.run(eval_genomes, settings.N_GENERATIONS)



if __name__ == '__main__':
    # Determine path to configuration file.
    parser = ArgumentParser(description='Help me throughout the evolution')
    parser.add_argument('--n_gen', type=int, help='number of generations')
    parser.add_argument('--time', type=int, help='running time of one epoch')
    parser.add_argument('--checkpoint', type=str, help='checkpoint to restore')
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')

    run(config_path, args)

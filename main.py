import pyximport; pyximport.install()

from torch import multiprocessing as mp

from Coach import Coach
from NNetWrapper import NNetWrapper as nn
from connect4.Connect4Game import Connect4Game as Game
from utils import *

args = dotdict({
    'run_name': 'connect4_resnet',
    'workers': mp.cpu_count() - 1,
    'startIter': 1,
    'numIters': 50,
    'process_batch_size': 128,
    'train_batch_size': 512,
    'train_steps_per_iteration': 100,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 3072,
    'numItersForTrainExamplesHistory': 20,
    'symmetricSamples': False,
    'numMCTSSims': 25,
    'tempThreshold': 10,
    'temp': 1,
    'compareWithRandom': True,
    'arenaCompareRandom': 500,
    'arenaCompare': 50,
    'arenaTemp': 0.1,
    'arenaMCTS': True,
    'randomCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 3,
    'expertValueWeight': dotdict({
        'start': 0,
        'end': 0.5,
        'iterations': 35
    }),
    'cpuct': 2,
    'checkpoint': 'checkpoint',
    'data': 'data',
})

if __name__ == "__main__":
    g = Game()
    nnet = nn(g)
    c = Coach(g, nnet, args)
    c.learn()

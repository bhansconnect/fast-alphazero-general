from torch import multiprocessing as mp

from Coach import Coach
from othello.NNet import NNetWrapper as nn
from othello.OthelloGame import OthelloGame as Game
from utils import *

args = dotdict({
    'workers': mp.cpu_count() - 1,
    'numIters': 100,
    'process_batch_size': 64,
    'train_batch_size': 512,
    #should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 768,
    'numItersForTrainExamplesHistory': 10,
    'symmetricSamples': False,
    'updateThreshold': 0.6,
    'arenaCompare': 50,
    'arenaTemp': 0.1,
    'numMCTSSims': 30,
    'tempThreshold': 15,
    'compareWithRandom': True,
    'randomCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 5,
    'expertValueWeight': dotdict({
        'start': 0,
        'end': 1,
        'iterations': 20
    }),
    'cpuct': 1,
    'checkpoint': 'checkpoint',
    'data': 'data',
    'load_model': False,
    'load_folder_file': ('./checkpoint/', 'iteration-best.pkl'),
})

if __name__ == "__main__":
    g = Game(6)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    c.learn()

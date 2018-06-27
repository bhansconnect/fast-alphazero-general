from torch import multiprocessing as mp

from Coach import Coach
from othello.NNet import NNetWrapper as nn
from othello.OthelloGame import OthelloGame as Game
from utils import *

args = dotdict({
    # Because of batching, expect about workers * train_batch_size * num_moves_in_a_game samples to generate at the same time
    'workers': mp.cpu_count() - 1,
    'process_batch_size': 256,
    'train_batch_size': 64,
    'numIters': 1000,
    'gamesPerIteration': 24000,
    'numMCTSSims': 30,
    'numItersForTrainExamplesHistory': 1,
    'checkpoint': 'checkpoint',
    'data': 'data',
    'arenaCompare': 500,
    'load_model': False,
    'load_folder_file': ('./checkpoint/', 'iteration-best.pkl'),
    'updateThreshold': 0.6,
    'tempThreshold': 15,
    'cpuct': 1,
    'arena': dotdict({
        'cpuct': 1,
        'temp': 0.1,
        'numMCTSSims': 30,
    })
})

if __name__ == "__main__":
    g = Game(6)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    c.learn()

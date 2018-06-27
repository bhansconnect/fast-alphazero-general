from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.NNet import NNetWrapper as nn
from torch import multiprocessing as mp
from utils import *

args = dotdict({
    'workers': mp.cpu_count(),
    'process_batch_size': 256,
    'train_batch_size': 64,
    'numIters': 10,
    'gamesPerIteration': 25000,
    'numMCTSSims': 100,
    'numItersForTrainExamplesHistory': 1,
    'checkpoint': 'checkpoint',
    'data': 'data',
    'arenaCompare': 100,
    'load_model': False,
    'load_folder_file': ('./checkpoint/','iteration-best.pkl'),
    'updateThreshold': 0.6,
    'tempThreshold': 40,
    'cpuct': 1,
    'arena': dotdict({
        'cpuct': 1,
        'temp': 0.1,
        'numMCTSSims': 30,
    })
})

if __name__=="__main__":
    g = Game(6)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    c.learn()

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.NNet import NNetWrapper as nn
from torch import multiprocessing as mp
from utils import *

args = dotdict({
    'workers': mp.cpu_count(),
    'batch_size': 128,
    'numIters': 2,
    'gamesPerIteration': 512,
    'numMCTSSims': 25,
    'numItersForTrainExamplesHistory': 10,
    'checkpoint': 'checkpoint',
    'arenaCompare': 40,
    'load_model': False,
    'load_folder_file': ('/checkpoint/','best.pkl'),

    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'cpuct': 1,

})

if __name__=="__main__":
    g = Game(6)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    c.learn()

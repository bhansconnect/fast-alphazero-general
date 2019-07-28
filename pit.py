import pyximport; pyximport.install()

import Arena
from MCTS import MCTS
from GenericPlayers import *
from othello.OthelloGame import OthelloGame, display
from othello.OthelloPlayers import *
from othello.NNet import NNetWrapper as NNet
import numpy as np
from utils import *


"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
if __name__ == '__main__':

    g = OthelloGame(6)

    # all players
    rp = RandomPlayer(g).play
    gp = GreedyOthelloPlayer(g).play
    hp = HumanOthelloPlayer(g).play

    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./checkpoint/', 'iteration-0050.pkl')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1)

    def n1p(x, turn):
        if turn <= 2:
            mcts1.reset()
        temp = 1 if turn <= 10 else 0
        policy = mcts1.getActionProb(x, temp=temp)
        if sum(policy) > 1:
            print('Multiple optins ->', policy)
        return np.random.choice(len(policy), p=policy)

    #n2 = NNet(g)
    #n2.load_checkpoint('./roundrobin/', 'iteration-0075.pkl')
    #args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    #mcts2 = MCTS(g, n2, args2)

    # def n2p(x, turn):
    #    if turn <= 2:
    #        mcts2.reset()
    #    temp = 1 if turn <= 10 else 0
    #    policy = mcts2.getActionProb(x, temp=temp)
    #    return np.random.choice(len(policy), p=policy)

    arena = Arena.Arena(n1p, hp, g, display=display)
    print(arena.playGames(2, verbose=True))

from pathlib import Path
from glob import glob
from utils import *
from othello.NNet import NNetWrapper as nn
from othello.OthelloGame import OthelloGame as Game
from tensorboardX import SummaryWriter
from GenericPlayers import *
from MCTS import MCTS
from Arena import Arena
import numpy as np
import choix

"""
use this script to play any all agents against a single agent and graph win rate.
"""

args = dotdict({
    'arenaCompare': 40,
    'arenaTemp': 0.1,
    # use zero if no montecarlo
    'numMCTSSims': 25,
    'playRandom': False,
    'cpuct': 1,
})

if __name__ == '__main__':

    benchmark_agent = "roundrobin/6x100x25_best.pth.tar"

    writer = SummaryWriter()
    if not Path('checkpoint').exists():
        Path('checkpoint').mkdir()
    print('Beginning comparison')
    networks = sorted(glob('checkpoint/*'))
    model_count = len(networks) + int(args.playRandom)

    if model_count <= 1:
        print(
            "Too few models for round robin. Please add models to the roundrobin/ directory")
        exit()

    total_games = model_count * args.arenaCompare
    print(
        f'Comparing {model_count} different models in {total_games} total games')

    g = Game(6)
    nnet1 = nn(g)
    nnet2 = nn(g)

    nnet1.load_checkpoint(folder="", filename=benchmark_agent)
    short_name = Path(benchmark_agent).stem

    if args.numMCTSSims <= 0:
        p1 = NNPlayer(g, nnet1, args.arenaTemp).play
    else:
        mcts1 = MCTS(g, nnet1, args)

        def p1(x): return np.argmax(
            mcts1.getActionProb(x, temp=args.arenaTemp))

    for i in range(model_count - 1):
        file = Path(networks[i])
        print(f'{short_name} vs {file.stem}')

        nnet2.load_checkpoint(folder='checkpoint', filename=file.name)
        if args.numMCTSSims <= 0:
            p2 = NNPlayer(g, nnet2, args.arenaTemp).play
        else:
            mcts2 = MCTS(g, nnet2, args)

            def p2(x): return np.argmax(
                mcts2.getActionProb(x, temp=args.arenaTemp))

        arena = Arena(p1, p2, g)
        p1wins, p2wins, draws = arena.playGames(args.arenaCompare)
        writer.add_scalar(
            f'Win Rate vs {short_name}', (p2wins + 0.5*draws)/args.arenaCompare, i)
        print(f'wins: {p1wins}, ties: {draws}, losses:{p2wins}')

import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board, turn):
        valids = self.game.getValidMoves(board, 1)
        valids = valids / np.sum(valids)
        a = np.random.choice(self.game.getActionSize(), p=valids)
        return a


class NNPlayer:
    def __init__(self, game, nn, temp=1, tempThreshold=0):
        self.game = game
        self.nn = nn
        self.temp = temp
        self.tempThreshold = tempThreshold

    def play(self, board, turn):
        policy, _ = self.nn.predict(board)
        valids = self.game.getValidMoves(board, 1)
        options = policy * valids
        temp = 1 if turn <= self.tempThreshold else self.temp
        if temp == 0:
            bestA = np.argmax(options)
            probs = [0] * len(options)
            probs[bestA] = 1
        else:
            probs = [x ** (1. / temp) for x in options]
            probs /= np.sum(probs)

        choice = np.random.choice(
            np.arange(self.game.getActionSize()), p=probs)

        if valids[choice] == 0:
            print()
            print(temp)
            print(valids)
            print(policy)
            print(probs)
            assert valids[choice] > 0

        return choice

import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        valids = valids / np.sum(valids)
        a = np.random.choice(self.game.getActionSize(), p=valids)
        return a


class NNPlayer:
    def __init__(self, game, nn, temp=1):
        self.game = game
        self.nn = nn
        self.temp = temp

    def play(self, board):
        policy, _ = self.nn.predict(board)
        valids = self.game.getValidMoves(board, 1)
        options = policy * valids
        if self.temp == 0:
            bestA = np.argmax(options)
            probs = [0] * len(options)
            probs[bestA] = 1
        else:
            probs = [x ** (1. / self.temp) for x in options]
            probs /= np.sum(probs)

        return np.random.choice(np.arange(self.game.getActionSize()), p=probs)

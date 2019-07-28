# cython: language_level=3
import sys
import numpy as np

sys.path.append('..')
from Game import Game
from .Connect4Logic import Board

DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WIN_LENGTH = 4

class Connect4Game(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height=None, width=None, win_length=None):
        Game.__init__(self)
        self.height = height or DEFAULT_HEIGHT
        self.width = width or DEFAULT_WIDTH
        self.win_length = win_length or DEFAULT_WIN_LENGTH

    def getInitBoard(self):
        b = Board(self.height, self.width, self.win_length)
        return np.asarray(b.pieces)

    def getBoardSize(self):
        return (self.height, self.width)

    def getActionSize(self):
        return self.width

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = Board(self.height, self.width, self.win_length)
        b.pieces = np.copy(board)
        b.add_stone(action, player)
        return (np.asarray(b.pieces), -player)

    def getValidMoves(self, board, player):
        "Any zero value in top row in a valid move"
        b = Board(self.height, self.width, self.win_length)
        b.pieces = np.copy(board)
        return np.asarray(b.get_valid_moves())

    def getGameEnded(self, board, player):
        b = Board(self.height, self.width, self.win_length)
        b.pieces = np.copy(board)
        is_ended, winner = b.get_win_state()
        if is_ended:
            if winner is None:
                # draw has very little value.
                return 1e-4
            elif winner == player:
                return +1
            elif winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', is_ended, winner)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        # Flip player from 1 to -1
        return board * player

    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi)]

    def stringRepresentation(self, board):
        return board.tostring()


def display(board):
    print(" -----------------------")
    print(' '.join(map(str, range(len(board[0])))))
    print(board)
    print(" -----------------------")

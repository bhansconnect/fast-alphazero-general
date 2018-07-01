import torch.multiprocessing as mp
import torch
import numpy as np
from queue import Full

from MCTS import MCTS


class SelfPlayAgent(mp.Process):

    def __init__(self, id, game, ready_queue, batch_ready, batch_tensor, policy_tensor, value_tensor, output_queue,
                 complete_count, games_played, args):
        super().__init__()
        self.id = id
        self.game = game
        self.ready_queue = ready_queue
        self.batch_ready = batch_ready
        self.batch_tensor = batch_tensor
        self.batch_size = self.batch_tensor.shape[0]
        self.policy_tensor = policy_tensor
        self.value_tensor = value_tensor
        self.output_queue = output_queue
        self.games = []
        self.canonical = []
        self.histories = []
        self.player = []
        self.turn = []
        self.mcts = []
        self.games_played = games_played
        self.complete_count = complete_count
        self.args = args
        self.valid = torch.zeros_like(self.policy_tensor)
        for _ in range(self.batch_size):
            self.games.append(self.game.getInitBoard())
            self.histories.append([])
            self.player.append(1)
            self.turn.append(1)
            self.mcts.append(MCTS(self.game, None, self.args))
            self.canonical.append(None)

    def run(self):
        while self.games_played.value < self.args.gamesPerIteration:
            self.generateCanonical()
            for i in range(self.args.numMCTSSims):
                self.generateBatch()
                self.processBatch()
            self.playMoves()
        with self.complete_count.get_lock():
            self.complete_count.value += 1
        self.output_queue.close()
        self.output_queue.join_thread()

    def generateBatch(self):
        for i in range(self.batch_size):
            board = self.mcts[i].findLeafToProcess(self.canonical[i])
            if board is not None:
                self.batch_tensor[i] = torch.from_numpy(board)
        self.ready_queue.put(self.id)

    def processBatch(self):
        self.batch_ready.wait()
        self.batch_ready.clear()
        for i in range(self.batch_size):
            self.mcts[i].processResults(self.policy_tensor[i].data.numpy(), self.value_tensor[i][0])

    def playMoves(self):
        for i in range(self.batch_size):
            temp = int(self.turn[i] < self.args.tempThreshold)
            policy = self.mcts[i].getExpertProb(self.canonical[i], temp)
            action = np.random.choice(len(policy), p=policy)
            self.histories[i].append((self.canonical[i], self.mcts[i].getExpertProb(self.canonical[i]), self.player[i]))
            self.games[i], self.player[i] = self.game.getNextState(self.games[i], self.player[i], action)
            self.turn[i] += 1
            winner = self.game.getGameEnded(self.games[i], 1)
            if winner != 0:
                lock = self.games_played.get_lock()
                lock.acquire()
                if self.games_played.value < self.args.gamesPerIteration:
                    self.games_played.value += 1
                    lock.release()
                    for hist in self.histories[i]:
                        if self.args.symmetricSamples:
                            sym = self.game.getSymmetries(hist[0], hist[1])
                            for b, p in sym:
                                self.output_queue.put((b, p, winner*hist[2]))
                        else:
                            self.output_queue.put((hist[0], hist[1], winner*hist[2]))
                    self.games[i] = self.game.getInitBoard()
                    self.histories[i] = []
                    self.player[i] = 1
                    self.turn[i] = 1
                    self.mcts[i] = MCTS(self.game, None, self.args)
                else:
                    lock.release()

    def generateCanonical(self):
        for i in range(self.batch_size):
            self.canonical[i] = self.game.getCanonicalForm(self.games[i], self.player[i])

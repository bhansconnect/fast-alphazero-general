import torch.multiprocessing as mp
import torch
import random
import numpy as np
from queue import Full


class ApprenticeAgent(mp.Process):

    def __init__(self, id, game, ready_queue, batch_ready, batch_tensor, policy_tensor, output_queue, complete_count, args):
        super().__init__()
        self.id = id
        self.game = game
        self.ready_queue = ready_queue
        self.batch_ready = batch_ready
        self.batch_tensor = batch_tensor
        self.batch_size = self.batch_tensor.shape[0]
        self.policy_tensor = policy_tensor
        self.output_queue = output_queue
        self.games = []
        self.histories = []
        self.player = []
        self.turn = []
        self.complete_count = complete_count
        self.args = args
        self.done = False
        self.valid = torch.zeros_like(self.policy_tensor)
        for _ in range(self.batch_size):
            self.games.append(self.game.getInitBoard())
            self.histories.append([])
            self.player.append(1)
            self.turn.append(1)

    def run(self):
        while not self.done:
            self.generateBatch()
            self.processBatch()
        with self.complete_count.get_lock():
            self.complete_count.value += 1

    def generateBatch(self):
        for i in range(self.batch_size):
            canonical_board = self.game.getCanonicalForm(self.games[i], self.player[i])
            self.histories[i].append((canonical_board, self.player[i]))
            self.batch_tensor[i] = torch.from_numpy(canonical_board)
        self.ready_queue.put(self.id)

    def processBatch(self):
        for i in range(self.batch_size):
            self.valid[i] = torch.from_numpy(self.game.getValidMoves(self.games[i], self.player[i]))
        self.batch_ready.wait()
        self.batch_ready.clear()
        policy = (self.policy_tensor * self.valid)
        for i in range(self.batch_size):
            policy_sum = torch.sum(policy[i])
            if policy_sum == 0:
                policy[i] = self.valid[i] / torch.sum(self.valid[i])
            else:
                policy[i] = policy[i] / policy_sum
            if self.turn[i] < self.args.tempThreshold:
                action = np.random.choice(policy.shape[1], p=policy[i].data.numpy())
            else:
                action = np.argmax(policy[i].data.numpy())
            self.games[i], self.player[i] = self.game.getNextState(self.games[i], self.player[i], action)
            self.turn[i] += 1
            over = self.game.getGameEnded(self.games[i], 1)
            if over != 0:
                r = random.randint(0, len(self.histories[i]) - 1)
                winner = self.game.getGameEnded(self.games[i], self.histories[i][r][1])
                try:
                    self.output_queue.put_nowait((self.histories[i][r][0], winner))
                except Full:
                    self.done = True
                    return
                self.games[i] = self.game.getInitBoard()
                self.histories[i] = []
                self.player[i] = 1
                self.turn[i] = 1

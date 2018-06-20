import torch.multiprocessing as mp
import torch
import random
import numpy as np
from queue import Full


class ApprenticeAgent(mp.Process):

    def __init__(self, id, game, ready_queue, batch_ready, batch_tensor, policy_tensor, output_queue,):
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
        self.turn = []
        self.valid = torch.zeros_like(self.policy_tensor)
        for _ in range(self.batch_size):
            self.games.append(self.game.getInitBoard())
            self.histories.append([])
            self.turn.append(1)

    def run(self):
        while not self.output_queue.full():
            self.generateBatch()
            self.processBatch()

    def generateBatch(self):
        for i in range(self.batch_size):
            self.histories[i].append(self.game.getCanonicalForm(self.games[i], self.turn[i]))
            self.batch_tensor[i] = torch.from_numpy(self.histories[i][-1])
        self.ready_queue.put(self.id)

    def processBatch(self):
        for i in range(self.batch_size):
            self.valid[i] = torch.from_numpy(self.game.getValidMoves(self.games[i], self.turn[i]))
        self.batch_ready.wait()
        self.batch_ready.clear()
        policy = (self.policy_tensor * self.valid)
        for i in range(self.batch_size):
            sum = torch.sum(policy[i])
            if sum == 0:
                policy[i] = self.valid[i] / torch.sum(self.valid[i])
            else:
                policy[i] = policy[i] / sum
            action = np.random.choice(policy.shape[1], p=policy[i].data.numpy())
            self.games[i], _ = self.game.getNextState(self.games[i], self.turn[i], action)
            self.turn[i] *= -1
            winner = self.game.getGameEnded(self.games[i], 1)
            if winner != 0:
                r = random.randint(0, len(self.histories[i]) - 1)
                try:
                    if r % 2 == 0:
                        self.output_queue.put_nowait((self.game.getCanonicalForm(self.histories[i][r], 1), winner))
                    else:
                        self.output_queue.put_nowait((self.game.getCanonicalForm(self.histories[i][r], -1), -1 * winner))
                except Full:
                    pass
                self.games[i] = self.game.getInitBoard()
                self.histories[i] = []
                self.turn[i] = 1

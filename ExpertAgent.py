from torch import multiprocessing as mp
import torch
from queue import Empty, Full
from MCTS import MCTS


class ExpertAgent(mp.Process):
    def __init__(self, id, game, ready_queue, batch_ready, batch_tensor, policy_tensor, value_tensor, input_queue,
                 output_queue, args):
        super().__init__()
        self.id = id
        self.game = game
        self.ready_queue = ready_queue
        self.batch_ready = batch_ready
        self.batch_tensor = batch_tensor
        self.batch_size = self.batch_tensor.shape[0]
        self.policy_tensor = policy_tensor
        self.value_tensor = value_tensor
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.args = args
        self.boards = [None] * self.batch_size
        self.values = [None] * self.batch_size
        self.mcts = [None] * self.batch_size
        self.samples = 0

    def run(self):
        while not self.output_queue.full():
            self.getSamples()
            if self.samples > 0:
                self.processSamples()
                self.outputSamples()

    def getSamples(self):
        self.samples = 0
        try:
            while self.samples < self.batch_size:
                self.boards[self.samples], self.values[self.samples] = self.input_queue.get(timeout=1)
                self.mcts[self.samples] = MCTS(self.game, None, self.args)
                self.samples += 1
        except Empty:
            pass

    def processSamples(self):
        for i in range(self.args.numMCTSSims):
            for j in range(self.samples):
                board = self.mcts[j].findLeafToProcess(self.boards[j])
                if board is not None:
                    self.batch_tensor[j] = torch.from_numpy(board)
            self.ready_queue.put(self.id)
            self.batch_ready.wait()
            self.batch_ready.clear()
            for j in range(self.samples):
                self.mcts[j].processResults(self.policy_tensor[j].data.numpy(), self.value_tensor[j].data.numpy())

    def outputSamples(self):
        try:
            for i in range(self.samples):
                policy = self.mcts[i].getExpertProb(self.boards[i])
                self.output_queue.put_nowait((self.boards[i], policy, self.values[i]))
        except Full:
            pass

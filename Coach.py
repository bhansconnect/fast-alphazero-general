from ApprenticeAgent import ApprenticeAgent
from ExpertAgent import ExpertAgent
import torch
from torch import multiprocessing as mp
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

from Arena import Arena
from MCTS import MCTS
from pytorch_classification.utils import Bar, AverageMeter
from queue import Empty
from time import time
import numpy as np


class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.args = args

        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f'iteration-best.pkl')

        self.apprentices = []
        self.experts = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.ready_queue = mp.Queue()
        self.sample_queue = mp.Queue(self.args.gamesPerIteration)
        self.file_queue = mp.Queue(self.args.gamesPerIteration)
        self.queue_full_event = mp.Event()
        self.apprenticesCompleted = mp.Value('i', 0)
        self.expertsCompleted = mp.Value('i', 0)
        boardx, boardy = self.game.getBoardSize()

        for i in range(self.args.workers):
            self.input_tensors.append(torch.zeros([self.args.batch_size, boardx, boardy]))
            self.input_tensors[i].pin_memory()
            self.input_tensors[i].share_memory_()

            self.policy_tensors.append(torch.zeros([self.args.batch_size, self.game.getActionSize()]))
            self.policy_tensors[i].pin_memory()
            self.policy_tensors[i].share_memory_()

            self.value_tensors.append(torch.zeros([self.args.batch_size, 1]))
            self.value_tensors[i].pin_memory()
            self.value_tensors[i].share_memory_()

            self.batch_ready.append(mp.Event())

    def learn(self):
        for i in range(self.args.numIters):
            print(f'------ITER {i+1}------')
            self.generateApprenticeAgents()
            self.processApprenticeBatches()
            self.generateExpertAgents()
            self.processExpertBatches()
            self.killApprenticeAgents()
            self.saveIterationSamples(i)
            self.killExpertAgents()
            self.train(i)
            self.compareToBest(i)
            print()

    def generateApprenticeAgents(self):
        self.queue_full_event = mp.Event()
        self.ready_queue = mp.Queue()
        for i in range(self.args.workers):
            self.apprentices.append(
                ApprenticeAgent(i, self.game, self.ready_queue, self.batch_ready[i],
                                self.input_tensors[i], self.policy_tensors[i], self.sample_queue, self.apprenticesCompleted))
            self.apprentices[i].start()

    def processApprenticeBatches(self):
        sample_time = AverageMeter()
        bar = Bar('Generating Apprentice Samples', max=self.args.gamesPerIteration)
        end = time()

        n = 0
        while self.apprenticesCompleted.value != self.args.workers:
            try:
                id = self.ready_queue.get(timeout=1)
                self.policy, _ = self.nnet.process(self.input_tensors[id])
                self.policy_tensors[id].copy_(self.policy, non_blocking=True)
                self.batch_ready[id].set()
            except Empty:
                pass
            size = self.sample_queue.qsize()
            if size > n:
                sample_time.update((time() - end) / (size - n), size - n)
                n = size
                end = time()
            bar.suffix = f'({size}/{self.args.gamesPerIteration}) Sample Time: {sample_time.avg:.3f}s | Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            bar.goto(size)
        bar.finish()
        print()
        for i in range(self.args.workers):
            self.batch_ready[i].set()

    def killApprenticeAgents(self):
        for i in range(self.args.workers):
            self.apprentices[i].terminate()
            self.batch_ready[i] = mp.Event()
        self.apprentices = []
        self.ready_queue = mp.Queue()
        self.apprenticesCompleted = mp.Value('i', 0)

    def generateExpertAgents(self):
        self.queue_full_event = mp.Event()
        self.ready_queue = mp.Queue()
        for i in range(self.args.workers):
            self.experts.append(
                ExpertAgent(i, self.game, self.ready_queue, self.batch_ready[i],
                            self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.sample_queue,
                            self.file_queue, self.expertsCompleted, self.args))
            self.experts[i].start()

    def processExpertBatches(self):
        sample_time = AverageMeter()
        bar = Bar('Generating Expert Samples', max=self.args.gamesPerIteration)
        end = time()

        n = 0
        while self.expertsCompleted.value != self.args.workers:
            try:
                id = self.ready_queue.get(timeout=1)
                self.policy, self.value = self.nnet.process(self.input_tensors[id])
                self.policy_tensors[id].copy_(self.policy, non_blocking=True)
                self.value_tensors[id].copy_(self.value, non_blocking=True)
                self.batch_ready[id].set()
            except Empty:
                pass
            size = self.file_queue.qsize()
            if size > n:
                sample_time.update((time() - end) / (size - n), size - n)
                n = size
                end = time()
            bar.suffix = f'({size}/{self.args.gamesPerIteration}) Sample Time: {sample_time.avg:.3f}s | Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            bar.goto(size)
        bar.finish()
        print()
        for i in range(self.args.workers):
            self.batch_ready[i].set()

    def killExpertAgents(self):
        for i in range(self.args.workers):
            self.experts[i].terminate()
            self.batch_ready[i] = mp.Event()
        self.experts = []
        self.ready_queue = mp.Queue()
        self.expertsCompleted = mp.Value('i', 0)

    def saveIterationSamples(self, iteration):
        boardx, boardy = self.game.getBoardSize()
        data_tensor = torch.zeros([self.args.gamesPerIteration, boardx, boardy])
        policy_tensor = torch.zeros([self.args.gamesPerIteration, self.game.getActionSize()])
        value_tensor = torch.zeros([self.args.gamesPerIteration, 1])
        try:
            for i in range(self.args.gamesPerIteration):
                data, policy, value = self.file_queue.get_nowait()
                data_tensor[i] = torch.from_numpy(data)
                policy_tensor[i] = torch.tensor(policy)
                value_tensor[i, 0] = value

            torch.save(data_tensor, f'data/iteration-{iteration:04d}-data.pkl')
            torch.save(policy_tensor, f'data/iteration-{iteration:04d}-policy.pkl')
            torch.save(value_tensor, f'data/iteration-{iteration:04d}-value.pkl')
        except Empty:
            print("Error, Missing data.")

    def train(self, iteration):
        datasets = []
        for i in range(max(0, iteration - self.args.numItersForTrainExamplesHistory), iteration + 1):
            data_tensor = torch.load(f'data/iteration-{i:04d}-data.pkl')
            policy_tensor = torch.load(f'data/iteration-{i:04d}-policy.pkl')
            value_tensor = torch.load(f'data/iteration-{i:04d}-value.pkl')
            datasets.append(TensorDataset(data_tensor, policy_tensor, value_tensor))

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers)

        self.nnet.train(dataloader)

        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f'iteration-{iteration:04d}.pkl')

        del dataloader
        del dataset
        del datasets

    def compareToBest(self, iteration):
        self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='iteration-best.pkl')
        pmcts = MCTS(self.game, self.pnet, self.args)
        nmcts = MCTS(self.game, self.nnet, self.args)
        print('PITTING AGAINST BEST VERSION')

        arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                      lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
        pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

        print('NEW/BEST WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
        if not (pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.args.updateThreshold):
            print('ACCEPTING NEW BEST MODEL')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='iteration-best.pkl')

from ApprenticeAgent import ApprenticeAgent
from ExpertAgent import ExpertAgent
import torch
from torch import multiprocessing as mp
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from queue import Empty
from time import time


class Coach:
    def __init__(self, game, nn, args):
        self.game = game
        self.nn = nn
        self.args = args

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
            print(f"Main: Generating {self.args.gamesPerIteration} samples")
            start = time()
            self.generateApprenticeAgents()
            self.processApprenticeBatches()
            print(f"Main: Generated samples in {time()-start} seconds")
            print(f"Main: Running Expert on samples")
            start = time()
            self.generateExpertAgents()
            self.processExpertBatches()
            self.killApprenticeAgents()
            print(f"Main: Ran Expert on samples in {time()-start} seconds")
            self.saveIterationSamples(i)
            self.killExpertAgents()
            print(f"Main: Training network on samples from last {min(self.args.numItersForTrainExamplesHistory, i+1)} iterations")
            start = time()
            self.train(i)
            print(f"Main: Trained network on samples in {time()-start} seconds")
            print(f"Main: Competing against self {self.args.arenaCompare} times")
            w, t, l = self.compareToLastIter()
            print(f"Main: Results -> wins: {w}, ties: {t}, losses: {l}\n")

    def generateApprenticeAgents(self):
        self.queue_full_event = mp.Event()
        self.ready_queue = mp.Queue()
        for i in range(self.args.workers):
            self.apprentices.append(
                ApprenticeAgent(i, self.game, self.ready_queue, self.batch_ready[i],
                                self.input_tensors[i], self.policy_tensors[i], self.sample_queue))
            self.apprentices[i].start()

    def processApprenticeBatches(self):
        while not self.sample_queue.full():
            try:
                id = self.ready_queue.get(timeout=1)
                self.policy, _ = self.nn.process(self.input_tensors[id])
                self.policy_tensors[id].copy_(self.policy, non_blocking=True)
                self.batch_ready[id].set()
            except Empty:
                pass

    def killApprenticeAgents(self):
        for i in range(self.args.workers):
            self.batch_ready[i].set()
            self.apprentices[i].terminate()
            self.batch_ready[i] = mp.Event()
        self.apprentices = []
        self.ready_queue = mp.Queue()

    def generateExpertAgents(self):
        self.queue_full_event = mp.Event()
        self.ready_queue = mp.Queue()
        for i in range(self.args.workers):
            self.experts.append(
                ExpertAgent(i, self.game, self.ready_queue, self.batch_ready[i],
                            self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.sample_queue,
                            self.file_queue, self.args))
            self.experts[i].start()

    def processExpertBatches(self):
        while not self.file_queue.full():
            try:
                id = self.ready_queue.get(timeout=1)
                self.policy, self.value = self.nn.process(self.input_tensors[id])
                self.policy_tensors[id].copy_(self.policy, non_blocking=True)
                self.value_tensors[id].copy_(self.value, non_blocking=True)
                self.batch_ready[id].set()
            except Empty:
                pass

    def killExpertAgents(self):
        for i in range(self.args.workers):
            self.batch_ready[i].set()
            self.experts[i].terminate()
            self.batch_ready[i] = mp.Event()
        self.experts = []

    def saveIterationSamples(self, iteration):
        boardx, boardy = self.game.getBoardSize()
        data_tensor = torch.zeros([self.args.gamesPerIteration, boardx, boardy])
        policy_tensor = torch.zeros([self.args.gamesPerIteration, self.game.getActionSize()])
        value_tensor = torch.zeros([self.args.gamesPerIteration, 1])
        for i in range(self.args.gamesPerIteration):
            data, policy, value = self.file_queue.get()
            data_tensor[i] = torch.from_numpy(data)
            policy_tensor[i] = torch.tensor(policy)
            value_tensor[i, 0] = value

        torch.save(data_tensor, f'data/iteration-{iteration:04d}-data.pkl')
        torch.save(policy_tensor, f'data/iteration-{iteration:04d}-policy.pkl')
        torch.save(value_tensor, f'data/iteration-{iteration:04d}-value.pkl')

    def train(self, iteration):
        datasets = []
        for i in range(max(0, iteration - self.args.numItersForTrainExamplesHistory), iteration+1):
            data_tensor = torch.load(f'data/iteration-{i:04d}-data.pkl')
            policy_tensor = torch.load(f'data/iteration-{i:04d}-policy.pkl')
            value_tensor = torch.load(f'data/iteration-{i:04d}-value.pkl')
            datasets.append(TensorDataset(data_tensor, policy_tensor, value_tensor))

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers)

        self.nn.train(dataloader)

        self.nn.save_checkpoint(folder=self.args.checkpoint, filename=f'iteration-{iteration:04d}.pkl')

        del dataloader
        del dataset
        del datasets

    def compareToLastIter(self):
        return 0, 0, 0

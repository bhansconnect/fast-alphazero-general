from MCTS import MCTS
from SelfPlayAgent import SelfPlayAgent
import torch
from torch import multiprocessing as mp
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from tensorboardX import SummaryWriter
from Arena import Arena
from GenericPlayers import RandomPlayer, NNPlayer
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

        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='iteration-0000.pkl')

        self.agents = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.ready_queue = mp.Queue()
        self.file_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)
        self.writer = SummaryWriter()
        boardx, boardy = self.game.getBoardSize()
        self.args.expertValueWeight.current = self.args.expertValueWeight.start

        for i in range(self.args.workers):
            self.input_tensors.append(torch.zeros([self.args.process_batch_size, boardx, boardy]))
            self.input_tensors[i].pin_memory()
            self.input_tensors[i].share_memory_()

            self.policy_tensors.append(torch.zeros([self.args.process_batch_size, self.game.getActionSize()]))
            self.policy_tensors[i].pin_memory()
            self.policy_tensors[i].share_memory_()

            self.value_tensors.append(torch.zeros([self.args.process_batch_size, 1]))
            self.value_tensors[i].pin_memory()
            self.value_tensors[i].share_memory_()

            self.batch_ready.append(mp.Event())

    def learn(self):
        print('Because of batching, it can take a long time before any games finish.')
        for i in range(1, self.args.numIters + 1):
            print(f'------ITER {i}------')
            self.generateSelfPlayAgents()
            self.processSelfPlayBatches()
            self.saveIterationSamples(i)
            self.killSelfPlayAgents()
            self.train(i)
            if i == 1:
                print('Note: Comparisons with Random do not use monte carlo tree search.')
            self.compareToRandom(i)
            self.compareToLast(i)
            z = self.args.expertValueWeight
            self.args.expertValueWeight.current = min(i, z.iterations)/z.iterations * (z.end - z.start) + z.start

    def generateSelfPlayAgents(self):
        self.ready_queue = mp.Queue()
        for i in range(self.args.workers):
            self.agents.append(
                SelfPlayAgent(i, self.game, self.ready_queue, self.batch_ready[i],
                              self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.file_queue,
                              self.completed, self.games_played, self.args))
            self.agents[i].start()

    def processSelfPlayBatches(self):
        sample_time = AverageMeter()
        bar = Bar('Generating Samples', max=self.args.gamesPerIteration)
        end = time()

        n = 0
        while self.completed.value != self.args.workers:
            try:
                id = self.ready_queue.get(timeout=1)
                self.policy, self.value = self.nnet.process(self.input_tensors[id])
                self.policy_tensors[id].copy_(self.policy, non_blocking=True)
                self.value_tensors[id].copy_(self.value, non_blocking=True)
                self.batch_ready[id].set()
            except Empty:
                pass
            size = self.games_played.value
            if size > n:
                sample_time.update((time() - end) / (size - n), size - n)
                n = size
                end = time()
            bar.suffix = f'({size}/{self.args.gamesPerIteration}) Sample Time: {sample_time.avg:.3f}s | Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            bar.goto(size)
        bar.finish()
        print()

    def killSelfPlayAgents(self):
        for i in range(self.args.workers):
            self.agents[i].join()
            self.batch_ready[i] = mp.Event()
        self.agents = []
        self.ready_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)

    def saveIterationSamples(self, iteration):
        num_samples = self.file_queue.qsize()
        print(f'Saving {num_samples} samples')
        boardx, boardy = self.game.getBoardSize()
        data_tensor = torch.zeros([num_samples, boardx, boardy])
        policy_tensor = torch.zeros([num_samples, self.game.getActionSize()])
        value_tensor = torch.zeros([num_samples, 1])
        for i in range(num_samples):
            data, policy, value = self.file_queue.get()
            data_tensor[i] = torch.from_numpy(data)
            policy_tensor[i] = torch.tensor(policy)
            value_tensor[i, 0] = value

        torch.save(data_tensor, f'{self.args.data}/iteration-{iteration:04d}-data.pkl')
        torch.save(policy_tensor, f'{self.args.data}/iteration-{iteration:04d}-policy.pkl')
        torch.save(value_tensor, f'{self.args.data}/iteration-{iteration:04d}-value.pkl')
        del data_tensor
        del policy_tensor
        del value_tensor

    def train(self, iteration):
        datasets = []
        for i in range(max(1, iteration - self.args.numItersForTrainExamplesHistory), iteration + 1):
            data_tensor = torch.load(f'{self.args.data}/iteration-{i:04d}-data.pkl')
            policy_tensor = torch.load(f'{self.args.data}/iteration-{i:04d}-policy.pkl')
            value_tensor = torch.load(f'{self.args.data}/iteration-{i:04d}-value.pkl')
            datasets.append(TensorDataset(data_tensor, policy_tensor, value_tensor))

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                num_workers=self.args.workers, pin_memory=True)

        l_pi, l_v = self.nnet.train(dataloader)
        self.writer.add_scalar('loss/policy', l_pi, iteration)
        self.writer.add_scalar('loss/value', l_v, iteration)
        self.writer.add_scalar('loss/total', l_pi + l_v, iteration)

        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=f'iteration-{iteration:04d}.pkl')

        del dataloader
        del dataset
        del datasets

    def compareToLast(self, iteration):
        self.pnet.load_checkpoint(folder=self.args.checkpoint, filename=f'iteration-{iteration-1:04d}.pkl')
        pplayer = MCTS(self.game, self.pnet, self.args)
        nplayer = MCTS(self.game, self.nnet, self.args)
        print(f'PITTING AGAINST LAST VERSION')

        arena = Arena(lambda x: np.argmax(pplayer.getActionProb(x, temp=self.args.arenaTemp)),
                      lambda x: np.argmax(nplayer.getActionProb(x, temp=self.args.arenaTemp)), self.game)
        pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

        print(f'NEW/LAST WINS : {nwins} / {pwins} ; DRAWS : {draws}')
        self.writer.add_scalar('win_rate/last', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)

    def compareToRandom(self, iteration):
        r = RandomPlayer(self.game)
        nnplayer = NNPlayer(self.game, self.nnet, self.args.arenaTemp)
        print('PITTING AGAINST RANDOM')

        arena = Arena(r.play, nnplayer.play, self.game)
        pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

        print(f'NEW/RANDOM WINS : {nwins} / {pwins} ; DRAWS : {draws}')
        self.writer.add_scalar('win_rate/random', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)
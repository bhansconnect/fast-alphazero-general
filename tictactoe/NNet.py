import os
import numpy as np
import sys

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet
from pytorch_classification.utils import Bar, AverageMeter
from time import time

import torch
import torch.optim as optim
from .TicTacToeNNet import TicTacToeNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.optimizer = optim.Adam(self.nnet.parameters())

        if args.cuda:
            self.nnet.cuda()

    def train(self, batches):
        self.nnet.train()
        overall_v_losses = AverageMeter()
        overall_pi_losses = AverageMeter()

        for epoch in range(args.epochs):
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time()

            bar = Bar(f'EPOCH {epoch+1:02d} ::: Training Net', max=len(batches))

            for batch_idx, batch in enumerate(batches):
                boards, target_pis, target_vs = batch

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # measure data loading time
                data_time.update(time() - end)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time() - end)
                end = time()

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                    batch=batch_idx + 1,
                    size=len(batches),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                bar.next()
            bar.finish()
            overall_v_losses.update(v_losses.avg)
            overall_pi_losses.update(pi_losses.avg)
            print()
        return overall_pi_losses.avg, overall_v_losses.avg

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        with torch.no_grad():
            board = board.view(1, self.board_x, self.board_y)

            self.nnet.eval()
            pi, v = self.nnet(board)

            # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
            return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def process(self, batch):
        if args.cuda:
            batch = batch.cuda(non_blocking=True)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(batch)
            return torch.exp(pi), v

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])

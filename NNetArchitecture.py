import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
sys.path.append('..')


# 1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)

# 3*3 convolution


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 1
        if downsample:
            stride = 2
            self.conv_ds = conv1x1(in_channels, out_channels, stride)
            self.bn_ds = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        residual = x
        out = x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual)
        out += residual
        return out


class NNetArchitecture(nn.Module):
    def __init__(self, game, args):
        super(NNetArchitecture, self).__init__()
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.conv1 = conv3x3(1, args.num_channels)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.res_layers = []
        for _ in range(args.depth):
            self.res_layers.append(ResidualBlock(
                args.num_channels, args.num_channels))
        self.resnet = nn.Sequential(*self.res_layers)

        self.v_conv = conv1x1(args.num_channels, 1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(self.board_x*self.board_y,
                               self.board_x*self.board_y//2)
        self.v_fc2 = nn.Linear(self.board_x*self.board_y//2, 1)

        self.pi_conv = conv1x1(args.num_channels, 2)
        self.pi_bn = nn.BatchNorm2d(2)
        self.pi_fc1 = nn.Linear(self.board_x*self.board_y*2, self.action_size)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        # batch_size x 1 x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        # batch_size x num_channels x board_x x board_y
        s = self.resnet(s)

        v = self.v_conv(s)
        v = self.v_bn(v)
        v = torch.flatten(v, 1)
        v = self.v_fc1(v)
        v = self.v_fc2(v)

        pi = self.pi_conv(s)
        pi = self.pi_bn(pi)
        pi = torch.flatten(pi, 1)
        pi = self.pi_fc1(pi)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

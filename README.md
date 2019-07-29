# Fast AlphaZero General

An implementation of [AlphaZero](https://deepmind.com/blog/alphago-zero-learning-scratch/) for any game, inspired by [alpha-zero-general](https://github.com/suragnair/alpha-zero-general/). It uses multiprocessing and cython to greatly increase the performance. Due to certain optimizations, the project requires a gpu. Also note, `.pyx` files are cython files that are very similar to python (identical in some cases).

This project only supports [Pytorch](https://pytorch.org/) models because of Pytorch's support for multiproccesing.

To use a game of your choice, subclass the classes in `Game.py` implement its functions. Example implementations for Othello can be found in `othello/OthelloGame.pyx`.

You may want to edit `NNetArchitecture.py` or `NNetWrapper.py` to implement different neural networks for various games.

`Coach.py` contains the core training loop and `MCTS.pyx` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in `main.py`. Additional neural network parameters are in `NNetWrapper.py` (learning rate, number of filters, depth of resnet).

### Installation

Install pytorch for gpu following [their guide](https://pytorch.org/get-started/locally/).
Then run:
```bash
pip install -r requirements.txt
```

### Execution

To start training a model for Connect4:
```bash
python main.py
```

Choose your game in `main.py`.

#### Performance

After training on Connect4 for 50 iteration on my desktop, the ai has definitely improved a ton:
```
1. iteration-0050 with 1.69 rating
2. iteration-0045 with 1.61 rating
3. iteration-0040 with 1.40 rating
4. iteration-0035 with 1.04 rating
5. iteration-0030 with 0.49 rating
6. iteration-0025 with 0.34 rating
7. iteration-0020 with -0.30 rating
8. iteration-0015 with -0.56 rating
9. iteration-0010 with -0.95 rating
10. iteration-0005 with -1.39 rating
11. iteration-0000 with -3.35 rating

(Rating Diff, Winrate) -> (0.5, 62%), (1, 73%), (2, 88%), (3, 95%), (5, 99%)
```
These ratings aren't exactly elo, but they give a good sense of how much the AI has improved. From basically being random with a monte carlo tree search in iteration 0 to winning about 99% of the time against random with a montecarlo tree search by the end. On top of that, when iteration 45 and 50 play each other, they tie over half of the games.

When I tell iteration 50 to play itself optimally, it ties every single game. This means that it does not yet play perfectly because in perfect play, the first play wins every game. Still, it is definitely very good.

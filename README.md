# Fast AlphaZero General

> There is a new version of this project that is significantly faster. It has large parts written in C++ and may be less beginner friendly, but it is much more practical to use. It lives at [bhansconnect/alphazero-pybind11](https://github.com/bhansconnect/alphazero-pybind11). 

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

After training on Connect4 for 200 iteration(~1 day) on my laptop(Intel i7-8750H, GTX 1060 6GB), the ai has definitely improved a ton:
```
1. iteration-0200 with 2.70 rating
2. iteration-0190 with 2.53 rating
3. iteration-0180 with 2.49 rating
4. iteration-0170 with 2.35 rating
5. iteration-0160 with 2.14 rating
6. iteration-0150 with 1.75 rating
7. iteration-0140 with 1.63 rating
8. iteration-0130 with 1.43 rating
9. iteration-0120 with 1.14 rating
10. iteration-0110 with 0.95 rating
11. iteration-0100 with 0.50 rating
12. iteration-0090 with 0.21 rating
13. iteration-0080 with -0.13 rating
14. iteration-0070 with -0.66 rating
15. iteration-0060 with -0.90 rating
16. iteration-0050 with -1.55 rating
17. iteration-0040 with -1.97 rating
18. iteration-0030 with -2.48 rating
19. iteration-0020 with -3.10 rating
20. iteration-0010 with -3.67 rating
21. iteration-0000 with -5.36 rating

(Rating Diff, Winrate) -> (0.5, 62%), (1, 73%), (2, 88%), (3, 95%), (5, 99%)
```
These ratings aren't exactly elo, but they give a good sense of how much the AI has improved. From basically being random with a monte carlo tree search in iteration 0 to winning about 100% of the time against random with a montecarlo tree search by the end. On top of that, when iteration 190 and 200 play each other, they tie over half of the games.

When I tell iteration 200 to play itself optimally, it wins every single game that it is the first play. This means that it plays perfectly because in perfect play, the first play wins every game.

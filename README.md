# Fast AlphaZero General

An implementation of [AlphaZero](https://deepmind.com/blog/alphago-zero-learning-scratch/) for any game, inspired by [alpha-zero-general](https://github.com/suragnair/alpha-zero-general/). It uses multiprocessing and cython to greatly increase the performance. Due to certain optimizations, the project requires a gpu. Also note, `.pyx` files are cython files that are very similar to python (identical in some cases).

This project only supports [Pytorch](https://pytorch.org/) models because of Pytorch's support for multiproccesing.

To use a game of your choice, subclass the classes in `Game.py` and `NeuralNet.py` and implement their functions. Example implementations for Othello can be found in `othello/OthelloGame.pyx` and `othello/NNet.py`.

`Coach.py` contains the core training loop and `MCTS.pyx` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in `main.py`. Additional neural network parameters are in `othello/NNet.py` (cuda flag, batch size, epochs, learning rate etc.).

To start training a model for Othello:

```bash
pip install -r requirements.txt
python main.py
```

Choose your game in `main.py`.

#### Performance

TODO: Needs updating with new changes

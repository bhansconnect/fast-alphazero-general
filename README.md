# expert-iteration-general
An implementation of [Expert Iterations](https://davidbarber.github.io/blog/2017/11/07/Learning-From-Scratch-by-Thinking-Fast-and-Slow-with-Deep-Learning-and-Tree-Search/) merge with [AlphaZero](https://deepmind.com/blog/alphago-zero-learning-scratch/) for any game, inspired by [alpha-zero-general](https://github.com/suragnair/alpha-zero-general/).

This project only supports [Pytorch](https://pytorch.org/) models because of Pytorch's support for multiproccesing.

By combining Expert Iterations' parallelism and AlphaZero's training strategies, this project is able to increase throughput on commodity hardware. This means better models for the average user in less time. On top of combining the two projects, the ability to slow change the value target from wins and losses to monte carlo value allows for the training to go more smoothly and reach more robust results in the same amount of time.

To use a game of your choice, subclass the classes in ```Game.py``` and ```NeuralNet.py``` and implement their functions. Example implementations for Othello can be found in ```othello/OthelloGame.py``` and ```othello/NNet.py```. 

```Coach.py``` contains the core training loop and ```MCTS.py``` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in ```main.py```. Additional neural network parameters are in ```othello/NNet.py``` (cuda flag, batch size, epochs, learning rate etc.). 

To start training a model for Othello:
```bash
python main.py
```
Choose your game in ```main.py```.

### Experiments
Soon I will directly compare this with alpha-zero-general given the same hardware and time limit for training.

# expert-iteration-general

An implementation of [Expert Iterations](https://davidbarber.github.io/blog/2017/11/07/Learning-From-Scratch-by-Thinking-Fast-and-Slow-with-Deep-Learning-and-Tree-Search/) merge with [AlphaZero](https://deepmind.com/blog/alphago-zero-learning-scratch/) for any game, inspired by [alpha-zero-general](https://github.com/suragnair/alpha-zero-general/).

This project only supports [Pytorch](https://pytorch.org/) models because of Pytorch's support for multiproccesing.

By combining Expert Iterations' parallelism and AlphaZero's training strategies, this project is able to increase throughput on commodity hardware. This means better models for the average user in less time. The main difference between Expert Iterations and alpha-zero-general is how they create training samples and how much the samples are trained on. For alpha-zero-general, the games are played using the MCTS. This leads to much stronger training games at the cost of much more processing power. Expert Iterations takes advantage of the parallelism of even a quadcore laptop for more efficiency. It is important to note that Expert Iterations leads to playing a broader space of games to avoid overfitting.

To use a game of your choice, subclass the classes in `Game.py` and `NeuralNet.py` and implement their functions. Example implementations for Othello can be found in `othello/OthelloGame.py` and `othello/NNet.py`.

`Coach.py` contains the core training loop and `MCTS.py` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in `main.py`. Additional neural network parameters are in `othello/NNet.py` (cuda flag, batch size, epochs, learning rate etc.).

To start training a model for Othello:

```bash
python main.py
```

Choose your game in `main.py`.

### Experiments

#### Expert Iterations Winrate vs 6x100x25_best.pth.tar

alpha-zero-general(pretrained model: 6x100x25_best.pth.tar) vs expert-iterations-general. The alpha-zero-general model took 3 days to train on a K80, playing about 8000 games over 80 iterations. On my laptop(960m), that would have taken approximately 1.25 days. I trained for Expert Iterations for 1.25 days. This lead to 184320 games played over 60 iterations.
![Win Rate vs 6x100x25_best.pth.tar](https://github.com/bhansconnect/expert-iteration-general/raw/master/winrate_vs_alpha_zero_general.png)

This graph shows that in the same amount of time that my model at a minimum trains as well as alpha-zero-general. What is imporatant to note is that Expert Iterations is better at playing in games that go outside of its comfort zone due to all the games it has trained on. alpha-zero-general is very good at a small subset of games, but Expert Iterations is better at a larger variation of games.

Also, it is interesting to note that iteration 0(which is equivalent to random with a MCTS) is actually better than a number of following iterations before the network manages to learn any valid strategy.

#### Roundrobin with 6x100x25_best.pth.tar

For more detail on how good the model became as it trained, I ran a roundrobin of every fifth model as well as 6x100x25_best.pth.tar Below is the estimated elo for each model with random anchored to 0.

```
Rankings:
1. iteration-0060 with 4.30 rating
2. iteration-0050 with 3.93 rating
3. iteration-0055 with 3.87 rating
4. iteration-0045 with 3.84 rating
5. 6x100x25_best.pth with 3.80 rating
6. iteration-0040 with 3.35 rating
7. iteration-0035 with 3.26 rating
8. iteration-0030 with 2.51 rating
9. iteration-0025 with 1.86 rating
10. iteration-0000 with 1.79 rating
11. iteration-0020 with 1.53 rating
12. iteration-0015 with 1.08 rating
13. iteration-0010 with 0.64 rating
14. iteration-0005 with 0.10 rating
15. random with 0.00 rating

(Rating Diff, Winrate) -> (0.5, 62%), (1, 73%), (2, 88%), (3, 95%), (5, 99%)
```

### Side note

I believe that alpha-zero-general has parameters that are selected in a way that leads to more overfitting. alpha-zero-general generates about 260 samples per game(32.4 moves per game and 8 symetries per move) and then trains on them for 10 epocks. Though these training samples are good, the network becomes extremely overfitted to them. Basically the network learns how to play a very small subset of games very well at the cost of poor generalization. In order to train better, each sample should be looked at a much smaller amount of times. Sadly, due to the high cost of generating more games, this will greatly slow down the learning pipeline and is why Expert Iterations has an advantage on commodity hardware.

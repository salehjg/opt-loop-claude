# Hyperparameters for the gradient-descent optimizer in train.py.
# THIS IS YOUR TARGET — edit these three values to make the optimizer reach the
# target loss in as FEW iterations as possible, without diverging.
#
# Problem: minimize f(w) = 0.5 * (w0^2 + 100 * w1^2), starting at w = [1, 1].
# It is ill-conditioned (condition number 100): the w1 direction is 100x
# steeper than w0. Too-large a learning rate makes w1 oscillate and blow up;
# too-small makes w0 crawl. Momentum lets you go faster without diverging.

LEARNING_RATE = 0.0331  # step size
MOMENTUM      = 0.669   # heavy-ball momentum in [0, 1)
NUM_ITERS     = 100     # max iterations to run

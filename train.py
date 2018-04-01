#!/usr/bin/python
# -*- coding: utf-8 -*-

from make_data import *
from lstm import *
import numpy as np
from chainer import optimizers, cuda
import time
import sys
import _pickle as cPickle

IN_UNITS = 5
HIDDEN_UNITS = 80
OUT_UNITS = 5
TRAINING_EPOCHS = 4000
DISPLAY_EPOCH = 10
MINI_BATCH_SIZE = 100
LENGTH_OF_SEQUENCE = 100
STEPS_PER_CYCLE = 50
NUMBER_OF_CYCLES = 100

xp = cuda.cupy

def compute_loss(model, sequences):
    loss = 0
    num, rows, cols = sequences.shape
    length_of_sequence = cols
    for i in range(cols - 1):
        x = chainer.Variable(
            xp.asarray(
                [[sequences[k, j, i + 0] for k in range(num)] for j in range(rows)], 
                dtype=np.float32
                )
        )
        t = chainer.Variable(
            xp.asarray(
                [[sequences[k, j, i + 1] for k in range(num)] for j in range(rows)], 
                dtype=np.float32
                )
        )
        loss += model(x, t)
    return loss 


if __name__ == "__main__":

    # make training data
    data_maker = DataMaker(steps_per_cycle=STEPS_PER_CYCLE, number_of_cycles=NUMBER_OF_CYCLES)
    train_data = data_maker.make()

    # setup model
    model = LSTM(IN_UNITS, HIDDEN_UNITS, OUT_UNITS)
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    model.to_gpu(0)

    # setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    start = time.time()
    cur_start = start
    for epoch in range(TRAINING_EPOCHS):
        sequences = data_maker.make_mini_batch(train_data, mini_batch_size=MINI_BATCH_SIZE, length_of_sequence=LENGTH_OF_SEQUENCE)
        model.reset_state()
        model.zerograds()
        loss = compute_loss(model, sequences)
        loss.backward()
        optimizer.update()

        if epoch != 0 and epoch % DISPLAY_EPOCH == 0:
            cur_end = time.time()
            # display loss
            print(
                "[{j}]training loss:\t{i}\t{k}[sec/epoch]".format(
                    j=epoch, 
                    i=loss.data/(sequences.shape[1] - 1), 
                    k=(cur_end - cur_start)/DISPLAY_EPOCH
                )
            )
            cur_start = time.time() 
            sys.stdout.flush()

    end = time.time()

    # save model
    cPickle.dump(model, open("./model.pkl", "wb"))

    print("{}[sec]".format(end - start))


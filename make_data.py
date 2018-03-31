#!/usr/bin/env python
# -*- coding:utf-8 -*-
 
import numpy as np
import math
import random
import pandas as pd 

ROW_SIZE = 3
TRAINING_START = 0
TRAINING_SIGNAL_LENGTH = 100000

random.seed(0)
   

class DataMaker(object):
        
    def __init__(self, steps_per_cycle, number_of_cycles):
        #self.steps_per_cycle = steps_per_cycle
        #self.number_of_cycles = number_of_cycles
        self.df = pd.read_csv("5sins.csv",encoding="shift_jis")     
             
    def make(self):
        return self.df 

    def make_mini_batch(self, data, mini_batch_size, length_of_sequence):  
        sequences = np.ndarray((ROW_SIZE, mini_batch_size, length_of_sequence), dtype=np.float32)
        for j in range(ROW_SIZE):
            data = self.df.ix[TRAINING_START:TRAINING_SIGNAL_LENGTH,j+1]
            for i in range(mini_batch_size):
                index = random.randint(0, len(data) - length_of_sequence)
                sequences[j][i] = data[index:index+length_of_sequence]
        return sequences


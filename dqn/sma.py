#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:09:38 2021

@author: monikkabusto
"""

import numpy as np
with open('b.txt') as g:
    data = [line for line in g.readlines()]
    
row_offset = 1
col_offset = 6
    
data = [line.strip().split(';') for line in data][row_offset:]
data_ = np.zeros((len(data), max([len(line) for line in data]) - col_offset))

for row in range(len(data)):
    print(row)
    if(col_offset < len(data[row])):
        data_available = np.array(data[row][col_offset:])
        
        data_[row,:len(data_available)] =  data_available


# Read the numeric values in the file. Specify a space delimiter, a row offset of 1, and a column offset of 0.



with open('b.txt') as g:
    data = [line for line in g.readlines()]

data = [line.strip().split(';') for line in data][row_offset:]
read_data = np.zeros((len(data), max([len(line) for line in data]) - col_offset))

for row in range(len(data)):
    if(col_offset < len(data[row])):
        data_available = np.array(data[row][col_offset:])
        read_data[row,:len(data_available)] =  data_available
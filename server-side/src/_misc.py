


import math
import os
import random
import re
import sys



#
# Complete the 'prison' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER m
#  3. INTEGER_ARRAY h
#  4. INTEGER_ARRAY v
#
n = 3
m = 2
# n += 2
# m += 2

h = [1, 2, 3]
v = [1, 2]

h_bars = list(range(n + 2))
v_bars = list(range(m + 2))

hh_bars = [b for b in h_bars if not b in h]
vv_bars = [b for b in v_bars if not b in v]



for i in range(len(hh_bars) - 1):
    for j in range(len(vv_bars) - 1):
        H = hh_bars[i+1] - hh_bars[i]
        V = vv_bars[i+1] - vv_bars[i]        
        print(H*V)




def prison(n, m, h, v):
    # Write your code here

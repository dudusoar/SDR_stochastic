import numpy as np
import matplotlib.pyplot as plt

def generate_time_intervals(time_range:int, time_step:int)-> tuple:
    '''
    产生时间区间
    '''
    intervals = tuple((i, min(i+time_step,time_range)) for i in range(0,time_range,time_step))
    return intervals





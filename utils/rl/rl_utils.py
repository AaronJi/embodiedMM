
import numpy as  np

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def get_str_from_1darray(v):
    round_digits = 4
    sv = [str(round(vv, round_digits)) for vv in v]
    return ','.join(sv)

def get_nparray_from_str(s):
    v = s.split(',')
    return np.array([float(vv) for vv in v])
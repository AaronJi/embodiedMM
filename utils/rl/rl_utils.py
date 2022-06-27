
import numpy as np

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

def reform_window_len(data, window_len, index_end = None, padding_num=None, batch_dim=1):
    if index_end is None:
        window_data = data[-window_len:]
    else:
        index_start = max(index_end - window_len + 1, 0)
        window_data = data[index_start:index_end+1]

    # padding
    if padding_num is not None:
        window_data = padding_to_window(window_data, window_len, padding_num=padding_num, batch_dim=batch_dim)
    return window_data

def padding_to_window(win_data, window_len, padding_num=None, batch_dim=1):

    if padding_num is None:
        padding_num = 0

    if batch_dim == 2:
        tlen = win_data.shape[1]
        dim = win_data.shape[2]
        dummy_array = np.ones((1, window_len - tlen, dim))
        cat_dim = 1
    elif batch_dim == 1:
        tlen = win_data.shape[0]
        dim = win_data.shape[1]
        dummy_array = np.ones((window_len - tlen, dim))
        cat_dim = 0
    else:
        tlen = win_data.shape[0]
        dummy_array = np.ones(window_len - tlen)
        cat_dim = 0


    win_data = np.concatenate([padding_num * dummy_array, win_data],  axis=cat_dim)
    return win_data
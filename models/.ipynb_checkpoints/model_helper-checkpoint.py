import torch.nn as nn


def activation_helper(activation, dim=None):
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU()
    elif activation is None:
        def act(x):
            return x
    else:
        raise ValueError('unsupported activation: %s' % activation)
    return act



'''
本文件只定义了一个函数，输入参数为一个字符，在函数内部通过判断activation来创建act对象，是一个方法对象，并返回，如果未传入字符，就手动创建一个act方法，返回其本身
'''

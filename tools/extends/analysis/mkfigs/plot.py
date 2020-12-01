import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use('fast')


def plot(X, fx):
    if callable(X):
        X = X()
    Y = [[f(x) for x in X] for f in fx]
    fig = plt.figure(figsize=(6.4, 4.2))
    for y in Y:
        plt.plot(X, y)
    # plt.subplots_adjust(left=0.05, right=0.97)
    ax = plt.gca()  # 获取当前坐标的位置
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    # 指定坐标的位置
    ax.xaxis.set_ticks_position('bottom')  # 设置bottom为x轴
    ax.yaxis.set_ticks_position('left')  # 设置left为x轴
    ax.spines['bottom'].set_position(('data', 0))  # 这个位置的括号要注意
    ax.spines['left'].set_position(('data', 0))
    plt.show()


if __name__ == "__main__":
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    tanh = lambda x: 2 * sigmoid(x) - 1
    relu = lambda x: x if x > 0 else 0
    leakyRelu = lambda x: x if x >= 0 else 0.1 * x
    plot(np.linspace(-10, 10, 1000), [sigmoid])
    plot(np.linspace(-10, 10, 1000), [tanh])
    plot(np.linspace(-10, 10, 1000), [relu])
    plot(np.linspace(-10, 10, 1000), [leakyRelu])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use('fast')


def plot(X, fx):
    if callable(X):
        X = X()
    Y = [[f(x) for x in X] for f in fx]
    fig = plt.figure(figsize=(6.4, 6.4))
    for y in Y:
        plt.plot(X, y)
    # plt.subplots_adjust(left=0.05, right=0.97)
    plt.show()


def line_plot(ax, x, ys, labels=None, styles=None, param_dict=None):
    """
    A helper function to make a line graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    x : array
       The x data

    ys : array
       The y data

    labels : list
       The ys labels

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    for i, y in enumerate(ys):
        if labels is not None:
            if styles is not None:
                ax.plot(x, y, label=labels[i], linestyle=styles[i])
            else:
                ax.plot(x, y, label=labels[i])
        else:
            ax.plot(x, y)
    if param_dict is not None:
        ax.set_xlabel(param_dict['xlabel'])  # Add an x-label to the axes.
        ax.set_ylabel(param_dict['ylabel'])  # Add a y-label to the axes.
        ax.set_title(param_dict['title'])  # Add a title to the axes.
    ax.legend()  # Add a legend
    return ax


def make_figure3():
    # make Figure 3
    fig = plt.figure(figsize=(6.4 * 3, 6))
    axes = [fig.add_subplot(1, 3, i) for i in range(1, 4)]
    a = np.linspace(0, 1, 101)
    bs = [0.75, 0.50, 0.25]
    titles = ['0.5<β<=1', 'β=0.5', '0<=β<0.5']
    for ax, b, title in zip(axes, bs, titles):
        t = 1 - (1 - b) * a
        e = (1 - b) * a
        y_eq_b = np.linspace(b, b, len(a))
        y_eq_1_minus_b = np.linspace(1 - b, 1 - b, len(a))
        intersect = (1 / (2 * (1 - b)), 1 / 2)
        if b == 0.5:
            x, ys = a, [t, e, y_eq_b]
            labels, ax_text = ['t', 'e', 'β=0.5'], {'xlabel': 'α', 'ylabel': 'y', 'title': title}
            styles = [None, None, '-.']
        else:
            x, ys = a, [t, e, y_eq_b, y_eq_1_minus_b]
            labels, ax_text = ['t', 'e', 'y=β', 'y=1-β'], {'xlabel': 'α', 'ylabel': 'y', 'title': title}
            styles = [None, None, '-.', '-.']
        line_plot(ax, x, ys, labels, styles, ax_text)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(linestyle='--')
    plt.subplots_adjust(left=0.05, right=0.97)
    # save_plt('./figures/Figure_3.Detection_efficiency_on_a.jpg')
    plt.show()


if __name__ == "__main__":
    make_figure3()
    relu = lambda x: x if x > 0 else 0
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    plot(np.linspace(-10, 10, 1000), [relu])
    plot(np.linspace(-10, 10, 1000), [sigmoid])

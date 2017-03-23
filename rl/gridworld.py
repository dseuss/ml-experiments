# coding: utf-8


import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Rectangle
from cached_property import cached_property
from itertools import product


GOAL_STYLE = {'facecolor': 'green', 'alpha': .5}
AGENT_STYLE = {'facecolor': 'blue', 'alpha': 1}


def _render_grid(gridsize, ax=None):
    ax = pl.gca() if ax is None else ax
    for axis, size in zip((ax.xaxis, ax.yaxis), gridsize):
        axis.limit_range_for_scale(0, size)
        axis.set_ticks(range(size + 1))
        axis.set_ticklabels([])

    ax.set_aspect('equal')
    return ax


def _render_boxes(boxes, ax=None):
    ax = pl.gca() if ax is None else ax
    for pos, style in boxes:
        ax.add_patch(Rectangle(pos, 1, 1, **style))

    return ax


class Gridworld(object):
    def __init__(self, gridsize, goal=np.array([0, 0])):
        self._gridsize = np.array(gridsize)
        self._goal = goal

        self.reset()

    def reset(self, position=None):
        position = position if position is not None \
            else np.array(self._gridsize) // 2
        self._agent_pos = np.array(position)

    def render(self, ax=None):
        ax = pl.gca() if ax is None else ax
        _render_grid(self._gridsize, ax=ax)
        boxes = [(self._goal, GOAL_STYLE), (self._agent_pos, AGENT_STYLE)]
        _render_boxes(boxes, ax=ax)

    @cached_property
    def stp_matrix(self):
        stp_left = np.zeros(tuple(self._gridsize) * 2)
        for x, y in product(range(1, self._gridsize[0]), range(self._gridsize[1])):
            stp_left[x - 1, y, x, y] = 1
        stp_left[0, :, 0, :] = np.eye(self._gridsize[1])

        stp_right = np.zeros(tuple(self._gridsize) * 2)
        for x, y in product(range(self._gridsize[0] - 1), range(self._gridsize[1])):
            stp_right[x + 1, y, x, y] = 1
        stp_right[-1, :, -1, :] = np.eye(self._gridsize[1])

        stp_up = np.zeros(tuple(self._gridsize) * 2)
        for x, y in product(range(self._gridsize[0]), range(self._gridsize[1] - 1)):
            stp_up[x, y + 1, x, y] = 1
        stp_up[:, -1, :, -1] = np.eye(self._gridsize[0])

        stp_down = np.zeros(tuple(self._gridsize) * 2)
        for x, y in product(range(self._gridsize[0]), range(1, self._gridsize[1])):
            stp_down[x, y - 1, x, y] = 1
        stp_down[:, 0, :, 0] = np.eye(self._gridsize[0])

        return {'up': stp_up, 'down': stp_down,
                'left': stp_left, 'right': stp_right}

    @property
    def state(self):
        return self._agent_pos[0] * self._gridsize[1] + self._agent_pos[1]

    @property
    def nr_states(self):
        return np.prod(self._gridsize)

    def step(self, action):
        """@todo: Docstring for .

        :param action: @todo
        :returns: @todo

        """
        stp = self.stp_matrix[action].reshape((self.nr_states,) * 2)
        new_state = np.random.choice(self.nr_states, p=stp[:, self.state])
        self._agent_pos = (new_state // self._gridsize[1],
                           new_state % self._gridsize[1])


if __name__ == '__main__':
    from sys import stdout

    world = Gridworld((10, 5))

    def keypress_event(event):
        if event.key in world.stp_matrix.keys():
            result = world.step(event.key)
            print(result)
            stdout.flush()
            ax.clear()
            world.render(ax)
            pl.draw()

    pl.style.use('ggplot')
    fig, ax = pl.subplots()
    fig.canvas.mpl_connect('key_press_event', keypress_event)
    world.render()
    pl.show()

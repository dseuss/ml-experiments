# coding: utf-8


from itertools import product
from sys import stdout

import matplotlib.pyplot as pl
import numpy as np
from cached_property import cached_property
from matplotlib.patches import Rectangle

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

    def reset(self, agent_pos=None):
        agent_pos = agent_pos if agent_pos is not None \
            else np.array(self._gridsize) // 2
        self._state = self._pos_to_state(agent_pos)

    def render(self, ax=None):
        ax = pl.gca() if ax is None else ax
        _render_grid(self._gridsize, ax=ax)
        agent_position = self._state_to_pos(self._state)
        boxes = [(self._goal, GOAL_STYLE), (agent_position, AGENT_STYLE)]
        _render_boxes(boxes, ax=ax)

    @cached_property
    def stprobs(self):
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

        stps = {'up': stp_up, 'down': stp_down,
                'left': stp_left, 'right': stp_right}

        # encoding the final state
        goal_x, goal_y = self._goal
        for stp in stps.values():
            stp[:, :, goal_x, goal_y] = 0.
            stp[goal_x, goal_y, goal_x, goal_y] = 1.0
        return {action: stp.reshape((self.nr_states,) * 2)
                for action, stp in stps.items()}

    @cached_property
    def rewards(self):
        reward = -np.ones(self.nr_states)
        target_state = self._pos_to_state(self._goal)
        reward[target_state] = 0
        return reward

    @property
    def state(self):
        return self._state

    @property
    def nr_states(self):
        return np.prod(self._gridsize)

    def _state_to_pos(self, state):
        """@todo: Docstring for _state_to_pos.

        :param state: @todo
        :returns: @todo

        """
        assert 0 <= state < self.nr_states
        return state // self._gridsize[1], state % self._gridsize[1]

    def _pos_to_state(self, position):
        """@todo: Docstring for _pos_to_state.

        :param position: @todo
        :returns: @todo

        """
        assert 0 <= position[0] < self._gridsize[0]
        assert 0 <= position[1] < self._gridsize[1]

        return position[0] * self._gridsize[1] + position[1]

    def step(self, action):
        """@todo: Docstring for .

        :param action: @todo
        :returns: @todo

        """
        stp = self.stprobs[action]
        new_state = np.random.choice(self.nr_states, p=stp[:, self.state])
        self._state = new_state
        return self.rewards[new_state]


def launch_interactive(world):
    """@todo: Docstring for launch_interactive.

    :param world: @todo
    :returns: @todo

    """
    def keypress_event(event):
        if event.key in world.stprobs.keys():
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
if __name__ == '__main__':
    world = Gridworld((10, 5))
    launch_interactive(world)

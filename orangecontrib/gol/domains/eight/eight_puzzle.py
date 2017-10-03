""" State representation for eight puzzle in goal-oriented programming. """
import pickle
import os
import random
import numpy as np
from Orange.data import DiscreteVariable


class EightPuzzleDomain:
    """ Contains a dictionary of states as keys and dtg as values.
    """
    def __init__(self):
        # check if db3x3.pickle available
        pickle_path = os.path.join(os.path.dirname(__file__), "db3x3.pickle")
        if not os.path.isfile(pickle_path):
            dtg = {}
            txt_path = os.path.join(os.path.dirname(__file__), "db3x3.txt")
            for l in open(txt_path):
                desc, nmoves = l.strip().split(",")
                desc = bytes(desc, encoding="utf8")
                dtg[desc] = float(nmoves)
                if dtg[desc] < 0:
                    dtg[desc] = -dtg[desc]
            pickle.dump(dtg, open(pickle_path, "wb"))
        self.dtg = pickle.load(open(pickle_path, "rb"))

        self.attributes = [DiscreteVariable.make("{}_{}".format(i, v),
                                                 values=["no", "yes"])
                           for i in range(9) for v in range(9)]

    def get_examples_from_traces(self, n, seed=0):
        """Function returns a pair (states, traces), where each state
        is then used as a single learning example. The function first selects
        N random legitimate problems and then finds  optimal solutions
        for each problem. The solution becomes the trace and all states
        on this trace become learning examples.  """

        gen = random.Random(seed)

        # select N random problems
        problems = set()
        keys = list(self.dtg.keys())
        while len(problems) < n:
            new_problem = gen.choice(keys)
            if self.dtg[new_problem] < 50:  # over 50 are unsolvable problems
                problems.add(new_problem)
        states, traces = [], []
        trace_id = 1
        for start in problems:
            state = EightPuzzleState(start, self)
            trace = [state]
            dtg = self.dtg[state.get_id()]
            while True:
                if dtg == 0:
                    break
                # find next states
                next_states = state.get_next_states()
                next_states = list(filter(lambda x: self.dtg[x.get_id()] < dtg,
                                          next_states))
                state = gen.choice(next_states)
                trace.append(state)
                dtg = self.dtg[state.get_id()]
            states += trace
            traces += [trace_id] * len(trace)
            trace_id += 1
        return np.array(states, dtype=object), np.array(traces, dtype=np.int16)

    def get_attributes(self):
        return self.attributes

    def get_attribute(self, at, state):
        if at.name == "solved":
            return state == b"123456780"
        tile, pos = at.name.split('_')
        if state[int(pos)] == ord(tile):
            return True
        return False


neighbours = {
    0: (1, 3),
    1: (0, 2, 4),
    2: (1, 5),
    3: (0, 4, 6),
    4: (1, 3, 5, 7),
    5: (2, 4, 8),
    6: (3, 7),
    7: (4, 6, 8),
    8: (5, 7)}


class EightPuzzleState:
    def __init__(self, state, domain, ind0=None):
        """
        :param state: an bytes representation of state
        :param domain: an EightPuzzleDomain instance
        """
        self.state = state
        if ind0 is None:
            self.ind0 = self.state.index(b'0')
        else:
            self.ind0 = ind0
        self.domain = domain

    def get_id(self):
        return self.state

    def get_next_states(self):
        """ Method that generates possible moves.
        Basically it is moving the empty square.
        """
        states = []
        for indx in neighbours[self.ind0]:
            new_state = bytearray(self.state)
            new_state[self.ind0] = self.state[indx]
            new_state[indx] = self.state[self.ind0]
            states.append(EightPuzzleState(bytes(new_state),
                                           self.domain, ind0=indx))
        return states

    def get_attribute(self, at):
        return self.domain.get_attribute(at, self.state)

    def achieved(self, goal):
        return self.state == b"123456780"

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

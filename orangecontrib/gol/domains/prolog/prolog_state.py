""" State representation for logic programing in
goal - oriented learning. 
"""
import os
import pickle
import numpy as np
from Orange.data import DiscreteVariable

class PrologData:
    def __init__(self, problem_id):
        # load data
        filename = os.path.join(os.path.dirname(__file__), "data/problem-{}.pickle".format(problem_id))
        self.graph, self.lex2id, self.atts, self.examples = pickle.load(open(filename, "rb"))
        # mapping from attribute name to attribute id
        self.atts_dict = {at:i for i, at in enumerate(self.atts)}
        # create sets of attributes for each state
        self.state_atts = {}
        for this_id, trace_id, str_sol, atts, lex in self.examples:
            if this_id not in self.state_atts:
                self.state_atts[this_id] = set(self.atts_dict[a] for a in atts)
        self.state_atts[0] = set()
        # create attributes
        self.attributes = [DiscreteVariable.make("a{}".format(i), values=["no", "yes"]) for i, at in enumerate(self.atts)]
        self.attributes.append(DiscreteVariable.make("solved", values=["no", "yes"]))

    def get_example_states(self):
        """Function returns a list of pairs (state, trace), 
        each pair is then used as a single learning example. """
        states = [PrologState(this_id, self) for this_id, trace_id, str_sol, atts, lex in self.examples]
        traces = np.array([trace_id for this_id, trace_id, str_sol, atts, lex in self.examples], dtype=np.int16)
        return states, traces

    def get_attributes(self):
        return self.attributes

class PrologState:
    def __init__(self, state, domain):
        """ The first argument of constructor is always the state,
        the second always domain. """
        self.state = state
        self.domain = domain

    def get_id(self):
        return self.state

    def get_moves(self):
        return [PrologState(s, self.domain) for s in self.domain.graph[self.state]]

    def get_attribute(self, at):
        if at.name == "solved":
            return self.solved()
        ex_atts = self.domain.state_atts[self.state]
        if int(at.name[1:]) in ex_atts:
            return "yes"
        return "no"

    def solved(self):
        return self.state == 0

    def __hash__(self):
        return self.state
      
    def __eq__(self, other):
        return self.state == other.state



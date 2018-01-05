import sympy
import numpy as np
from Orange.data import DiscreteVariable
import orangecontrib.gol.examples as ex

def minus(st):
    """ if any(a, b, c, or d) are not zero, then you can substract
    those values. """
    new_states = []
    if st[0] != 0: # and st[0] != 1:
        new_states.append((0, st[1], sympy.simplify(st[2]-st[0]), st[3]))
    if st[1] != 0:
        new_states.append((st[0], 0, st[2], sympy.simplify(st[3]-st[1])))
    if st[2] != 0: # and st[2] != 1:
        new_states.append((sympy.simplify(st[0]-st[2]), st[1], 0, st[3]))
    if st[3] != 0:
        new_states.append((st[0], sympy.simplify(st[1]-st[3]), st[2], 0))
    return new_states

def divide(st):
    """ Whether a or c are not zero or 1, you can divide those values. """
    new_states = []
    if st[0] != 0 and st[0] != 1:
        new_states.append((1, sympy.simplify(st[1]/st[0]), sympy.simplify(st[2]/st[0]), sympy.simplify(st[3]/st[0])))
    if st[2] != 0 and st[2] != 1:
        new_states.append((sympy.simplify(st[0]/st[2]), sympy.simplify(st[1]/st[2]), 1, sympy.simplify(st[3]/st[2])))
    return new_states        
        

a = sympy.Symbol("a")
b = sympy.Symbol("b")
c = sympy.Symbol("c")
d = sympy.Symbol("d")

start = (a,b,c,d)
states = {}

toexplore = [start]
for st in toexplore:
    if st in states:
        continue
    states[st] = []
    mn = minus(st)
    dv = divide(st)
    states[st].extend(mn)
    states[st].extend(dv)
    toexplore.extend(states[st])
    
strstates = {}
for s in states:
    strstates[str(s)] = s

class Eq1Domain:
    """ Prepares domain (attributes) used in GOL learning. """
    def __init__(self):
        self.attributes = []
        for var in ["a", "b", "c", "d"]:
            self.attributes.append(DiscreteVariable.make(var, values=["0","1","2"]))
        self.attributes.append(DiscreteVariable.make("solved", values=["no", "yes"]))

    def create_learning_examples(self):
        eq_states = []
        for s in states:
            state = Eq1State(s, self)
            eq_states.append(state)
        traces = [1] * len(eq_states)
        return (ex.create_data_from_states(eq_states, traces), 
                np.array(eq_states, dtype=object), 
                np.array(traces, dtype=int))

    def get_attributes(self):
        return self.attributes

    def get_attribute(self, at, state):
        if at.name == "solved":
            val = state.state
            if val[0] == 1 and val[1] == 0 and val[2] == 0:
                return "yes"
            return "no"
        return eval("self.{}(state)".format(at.name))

    def a(self, state):
        return self.val_at(0, state)

    def b(self, state):
        return self.val_at(1, state)

    def c(self, state):
        return self.val_at(2, state)

    def d(self, state):
        return self.val_at(3, state)

    def val_at(self, i, state):
        val = state.state
        if val[i] == 0:
            return "0"
        elif val[i] == 1:
            return "1"
        else:
            return "2"


class Eq1State:
    """ State-space for solving equations with one unknown. """
    def __init__(self, state, domain):
        self.state = state
        self.domain = domain

    def get_id(self):
        return self.state

    def get_next_states(self):
        """ Method that generates possible moves. """
        new_states = []
        neighbors = states[self.state]
        for ns in neighbors:
            new_states.append(Eq1State(ns, self.domain))
        return new_states

    def get_attribute(self, at):
        return self.domain.get_attribute(at, self)

    def __hash__(self):
        return hash(str(self.state))

    def __eq__(self, other):
        return self.state == other.state

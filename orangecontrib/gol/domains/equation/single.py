import sympy
from Orange.data import DiscreteVariable

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
    
def create_domain():
    """ Prepares domain (attributes) used in GOL learning.
    """
    features = []
    for var in ["a", "b", "c", "d"]:
        features.append(DiscreteVariable.make(var, values=["0","1","2"]))
    features.append(DiscreteVariable.make("solved", values=["no", "yes"]))
    return features

class State:
    features = create_domain()

    def __init__(self, state=None):
        if not state:
            self.state = start
        else:
            self.state = state

    @staticmethod
    def get_all_states():
        all_states = []
        for k in strstates:
            state = State()
            state.state = strstates[k]
            if state.solved() == "no":
                all_states.append(state)
        return all_states

    def get_id(self):
        return self.state

    def get_moves(self):
        """ Method that generates possible moves. """
        moves = []
        new_states = states[self.state]
        for ns in new_states:
            moves.append(State(ns))
        return moves

    def get_feature(self, at):
        return eval("self.{}()".format(at.name))

    def a(self):
        return self.val_at(0)

    def b(self):
        return self.val_at(1)

    def c(self):
        return self.val_at(2)

    def d(self):
        return self.val_at(3)

    def val_at(self, i):
        val = self.state
        if val[i] == 0:
            return "0"
        elif val[i] == 1:
            return "1"
        else:
            return "2"

    def solved(self):
        val = self.state
        if val[0] == 1 and val[1] == 0 and val[2] == 0:
            return "yes"
        return "no"    

    def __hash__(self):
        return hash(str(self.state))

    def __eq__(self, other):
        return self.state == other.state 

import sympy
import Orange
#import orangol

def minus(st):
    """ if any(a,b,c, or d) are not zero, then you can substract
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
    """ WHether a or c are not zero or 1, you can divide those values. """
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

print(len(states))
print(states)
strstates = {}
for s in states:
    strstates[str(s)] = s
    
def a(self):
    return val_at(self, 0)

def b(self):
    return val_at(self, 1)

def c(self):
    return val_at(self, 2)

def d(self):
    return val_at(self, 3)

def val_at(self, i):
    val = strstates[self.state]
    if val[i] == 0:
        return "0"
    elif val[i] == 1:
        return "1"
    else:
        return "2"

def finished(self):
    val = strstates[self.state]
    if val[0] == 1 and val[1] == 0 and val[2] == 0:
        return "yes"
    return "no"    


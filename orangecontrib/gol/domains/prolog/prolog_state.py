""" State representation for logic programing in
goal - oriented learning. 
""" 

# TODO: reimmplement PrologState

import pickle
import collections
import Orange

def finished(self):
    return "yes" if self.examples[self.state][1] else "no"

class HasValue:
    def __init__(self, value, examples):
        self.value = value
        self.examples = examples
        
    def __call__(self, obj):
        return "yes" if self.value in obj.examples[obj.state][0] else "no"

class Prolog(Orange.core.GOL_State):
    def __init__(self, domain_name="del", skip_domain=False):
        """ Currently are included only del and member pickled files. """
        if not skip_domain:
            self.create_domain(domain_name)
        self.state = ""

    def create_domain(self, domain_name):
        """ Prepares domain (attributes) used in GOL learning.
        """
        print "creating domain ...", "member"
        self.graph, self.example_list = pickle.load(open("D:\\work\\domains\\chess\\kbnk\\goalol\\prolog\\%s.pickle" % domain_name, "rb"))

        # parsing previously pickled files
        atts = set()
        self.examples = {}
        self.str_sols = collections.defaultdict(list)
        for s, ti, vs, solv, str_sol in self.example_list:
            self.str_sols[str(s)].append(str_sol)
            if str(s) in self.examples:
                self.examples[str(s)].append((ti, vs, solv))
            else:
                self.examples[str(s)] = [(ti, vs, solv)]
            atts |= vs
        traces = set()
        for k in self.examples:
            vals = self.examples[k]
            if any(s for ti,v,s in vals):
                solved = True
            else:
                solved = False
            self.examples[k] = (vals[0][1], solved, vals[0][0])
            traces.add(vals[0][0])
        
        orng_atts = [Orange.feature.Discrete(str(sa), values = ["yes", "no"]) 
                     for sa in atts]
        fin = Orange.feature.Discrete("Finished", values=["yes","no"])

        self.attributes = orng_atts + [fin]
        self.funcs = {fin.name : finished}
        for sa in orng_atts:
            self.funcs[sa.name] = HasValue(sa.name, self.examples)
            
        self.att_ids = {a.name:i for i,a in enumerate(self.attributes)}
        self.domain = Orange.data.Domain(self.attributes, fin)
        id = Orange.feature.String("id")
        mid = Orange.feature.Descriptor.new_meta_id()
        self.domain.add_meta(mid, id)

        trace = Orange.feature.Continuous("trace")
        mid = Orange.feature.Descriptor.new_meta_id()
        self.domain.add_meta(mid, trace)

        eval = Orange.feature.Continuous("eval")    
        mid = Orange.feature.Descriptor.new_meta_id()
        self.domain.add_meta(mid, eval)    

        achieved = Orange.feature.String("achieved")    
        mid = Orange.feature.Descriptor.new_meta_id()
        self.domain.add_meta(mid, achieved)    

    def create_example(self):
        example = Orange.data.Instance(self.domain)
        example["id"] = str(self.state)
        example["eval"] = 1
        example["trace"] = self.examples[str(self.state)][2]
        for a in self.attributes:
            example[a] = self.funcs[a.name](self)
        return example

    def get_all_states(self):
        # first compute counts of values
        all_states = []
        for k in self.examples:
            self.state = k
            if finished(self) == "no":
                all_states.append(k)
        """for k in self.graph:
            for val in self.graph[k]:
                self.state = val
                if finished(self) == "no":
                    all_states.append(k)"""
        return all_states

    def id(self):
        return self.state

    def set_state(self, id):
        self.state = id

    def evaluate(self):
        return 1.0

    def get_moves(self):
        """ Method that generates possible moves."""
        moves = Orange.core.GOL_MoveList()
        if self.state not in self.graph:
            return moves
        
        new_states = self.graph[self.state]
        for ns in new_states:
            moves.append(ProMove(self.state, str(ns)))
        return moves

    def or_node(self):
        return True

    def do_move(self, move):
        self.state = move.new_state
            
    def undo_move(self, move):
        self.state = move.old_state

    def prune_prob(self, state, goal_list, max_depth, depth):
        return 0.

    def get_attribute_value(self, at):
        """ Returns value of at in this current state.
            It has to be of Orange.data.Value type. """
        return Orange.data.Value(self.attributes[at], 
                                 self.funcs[self.attributes[at].name](self))


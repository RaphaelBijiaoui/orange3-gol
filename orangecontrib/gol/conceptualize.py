import numpy as np
from Orange.classification.rules import Selector, Rule
from Orange.data import DiscreteVariable, Table
import orangecontrib.evcrules.rules as rules
import orangecontrib.gol.examples as ex

class BasicConceptualizer:
    """ Basic conceptualizer implements basic goal-oriented learning: 
    no active learning, no continuous goals (increase/decrease),
    no and-or problems. """
    def __init__(self, rule_learner=None, select_goal=None):
        if not rule_learner:
            self.rule_learner = RuleLearner()
        else:
            self.rule_learner = rule_learner
        if not select_goal:
            self.select_goal = BreadthFirstSelector()
        else:
            self.select_goal = select_goal

    def __call__(self, learn_examples, learn_states, final_goal, max_depth):
        # preprocessing: create a set of all reachable states
        # and corresponding examples to evaluate goals faster
        # all_states ... a dictionary of all states, value is index in examples
        # all_examples ... all_examples in Orange format
        # ach_states ... numpy arrays of achievable states from each learning example
        all_states, all_examples, ach_states = self.preprocess(learn_states, max_depth)
        all_examples.save("all.tab")

        # create initial goal tree
        gtree = GoalNode(final_goal, None, 1.0, np.zeros(len(learn_examples), dtype=bool))
        while True:
            selected = self.select_goal(gtree)
            if not selected:
                break
            selected.expanded = True
            if np.all(selected.covered):
                continue
            # find positive and negative examples
            Y = np.zeros(learn_examples.X.shape[0], dtype=int)
            indices = np.where(selected.covered == False)[0]
            for ix in indices:
                # check whether selected.goal is solvable
                ach = ach_states[ix]
                achX = all_examples.X[ach]
                achieved = selected.goal(learn_examples.X[ix], achX)
                Y[ix] = np.any(achieved)
            # create a subset of learn_examples
            examples = Table.from_table_rows(learn_examples, indices)
            examples.Y = Y[indices]
            # learn rules
            rules = self.rule_learner(examples)
            # list of covered positive examples
            cov = np.zeros(len(examples), dtype=bool)
            for r in rules:
                cov |= r.covered_examples
            Y[indices] = Y[indices] & cov # Y = covered positive examples
            for r in rules:
                new_goal = Goal(r)
                success = r.quality
                new_covered = np.array(selected.covered) | Y
                new_covered[indices] |= r.covered_examples
                selected.children.append(GoalNode(new_goal, selected, success, new_covered))
        return gtree

    def preprocess(self, learn_states, max_depth):
        all_states = {}
        ach_sets = [set() for s in learn_states]
        for si, s in enumerate(learn_states):
            if s not in all_states:
                all_states[s] = len(all_states)
            # find states that are achievable from s in max_depth
            ach = achievable(s, max_depth)
            for a in ach:
                if a not in all_states:
                    all_states[a] = len(all_states)
                ach_sets[si].add(all_states[a])
        state_list = sorted(all_states.keys(), key = lambda s: all_states[s])
        all_examples = ex.create_data(state_list)
        ach_states = []
        for s in ach_sets:
            ach = np.zeros(len(all_examples), dtype=bool)
            ach[list(s)] = True
            ach_states.append(ach)
        return all_states, all_examples, ach_states


def achievable(state, depth):
    ach = [state]
    if depth == 0:
        return ach
    for ns in state.get_moves():
        ach += achievable(ns, depth-1)
    return ach

class Goal:
    INCREASE = 1
    DECREASE = 2
    CHANGED = 3

    def __init__(self, static, dynamic=None):
        """ Static goals are those that can be implemented with
        Orange's implementation of a rule, that is, a list of selectors. 
        Dynamic goals evaluate
        whether some change between starting and finishing example
        was achieved (e.g. value changed, or increased, or decreased 
        for continuous attributes). """
        self.static=static
        self.dynamic=dynamic

    def __call__(self, start, ends):
        """ 
        start: starting example (numpy array), 
        ends: a list of ending examples (2d numpy arrays). """
        return self.static.evaluate_data(ends)

    @staticmethod
    def create_initial_goal(domain, conditions):
        """ Initial goal has only static conditions. """
        selectors = []
        for c in conditions:
            column = domain.index(c[0])
            feature = domain[column]
            if isinstance(feature, DiscreteVariable):
                value = feature.values.index(c[2])
            else:
                value = c[2]
            selectors.append(Selector(column, c[1], value))
        rule = Rule(selectors=selectors, domain=domain)
        rule.prediction = 0
        return Goal(rule)

    def __str__(self):
        return "Static: {}, dynamic: {}".format(str(self.static), str(self.dynamic))


class GoalNode:
    def __init__(self, goal, parent, success, covered):
        """
        A goal-node contains the actual goal, a link to its parent GoalNode, 
        the probability (success) to achieve the parent goal, and instances 
        (an array of ones and zeros) covered by this particular strategy. 
        """
        self.goal = goal
        self.children = []
        self.parent = parent
        self.success = success
        self.covered = covered
        self.expanded = False

    def __str__(self):
        return self.str_rec(0)

    def str_rec(self, indent):
        gstr = ' ' * indent
        gstr += 'g: {}, succ: {:4.4f}, cov: {:4.4f}\n'.format(str(self.goal), self.success, 
                                                              float(sum(self.covered))/self.covered.shape[0])
        for c in self.children:
            gstr += c.str_rec(indent+2)
        return gstr



class BreadthFirstSelector:
    """ 
    Class returns the unexpanded goal that is the closest to the root. 
    """
    def __call__(self, goal_tree):
        goal, depth = self.find_closest_unexpanded(goal_tree)
        return goal

    def find_closest_unexpanded(self, goal_tree):
        if not goal_tree.expanded:
            return goal_tree, 0

        closest, cl_depth = None, -1
        for c in goal_tree.children:
            goal, depth = self.find_closest_unexpanded(c)
            if goal and (cl_depth == -1 or cl_depth > depth):
                closest, cl_depth = goal, depth
        return closest, cl_depth

class QualityValidator:
    def __init__(self, validator, threshold):
        self.validator = validator
        self.threshold = threshold

    def validate_rule(self, rule):
        if rule.quality < self.threshold:
            return False
        return self.validator.validate_rule(rule)


class RuleLearner:
    def __init__(self, nrules=5, min_cov=1, m=2, min_acc=0.5):
        self.learner = rules.RulesStar(evc=False, width=nrules*2, m=m)
        self.learner.rule_validator = QualityValidator(self.learner.rule_validator, min_acc)
        self.nrules = nrules
        self.min_cov = min_cov

    def __call__(self, examples):
        # learn rules for goal=yes
        self.learner.target_class = "yes"
        rules = self.learner(examples).rule_list

        # return only self.nrules rules (each rule must cover at least 
        # self.min_cov examples)
        sel_rules = []
        all_covered = np.zeros(len(examples), dtype=bool)
        for r in rules:
            if len(sel_rules) >= self.nrules:
                break
            new_covered = r.covered_examples & ~all_covered
            if np.count_nonzero(new_covered) >= self.min_cov:
                sel_rules.append(r)
        return sel_rules



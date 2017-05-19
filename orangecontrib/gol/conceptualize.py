import numpy as np
from Orange.classification.rules import Selector, Rule
from Orange.data import DiscreteVariable, Table
import orangecontrib.evcrules.rules as rules
import orangecontrib.gol.examples as ex

class BasicConceptualizer:
    """ Basic conceptualizer implements basic goal-oriented learning: 
    no continuous goals (increase/decrease),
    no and-or problems or for two-player games. """
    def __init__(self, rule_learner=None, select_goal=None):
        if not rule_learner:
            self.rule_learner = RuleLearner()
        else:
            self.rule_learner = rule_learner
        if not select_goal:
            self.select_goal = BreadthFirstSelector()
        else:
            self.select_goal = select_goal

    def __call__(self, learn_examples, learn_states, learn_traces, final_goal, max_depth):
        goal_validator = GoalValidator(learn_states, max_depth)

        # create initial goal tree
        # comment: usually we do not have any relevant positives at the beginning,
        # so set them all to be relavant
        relevant = np.ones(len(learn_examples), dtype=bool)
        covered = np.zeros(len(learn_examples), dtype=bool)
        gtree = GoalNode(final_goal, None, 1.0, 1.0, covered, relevant, set(learn_traces))
        while True:
            selected = self.select_goal(gtree)
            if not selected:
                break
            selected.expanded = True
            if np.all(selected.covered):
                continue
            # find positive and negative examples
            Y = np.zeros(learn_examples.X.shape[0], dtype=int)
            Yrel = np.zeros(learn_examples.X.shape[0], dtype=int)
            indices = np.where(selected.covered == False)[0]
            for ix in indices:
                # check whether selected.goal is solvable
                # check whether this example is relevant
                # (is close to a relevant example in the above goal)
                Y[ix], Yrel[ix] = goal_validator(learn_examples[ix], ix, selected.goal, selected.relevant)
            # create a subset of learn_examples
            examples = Table.from_table_rows(learn_examples, indices)
            examples.Y = Y[indices]
            # create a subset of traces
            traces = learn_traces[indices]
            # learn rules
            rules = self.rule_learner(examples, traces, selected.traces, Yrel)
            for r in rules:
                new_goal = Goal(r)
                success = r.quality
                default = sum(examples.Y) / examples.Y.shape[0]
                new_covered = np.array(selected.covered) | Y
                new_covered[indices] |= r.covered_examples
                # only positive examples should be added as covered
                # (other are not solved yet)

                new_traces = set(traces[r.covered_examples]) & selected.traces
                selected.children.append(GoalNode(new_goal, selected, success, default, new_covered, new_traces))
            print(gtree)
        return gtree

class GoalValidator:
    """
    Class used to validate whether a goal is achievable in a certain state
    """
    def __init__(self, learn_states, max_depth):
        # all_states ... a dictionary of all states, value is index in examples
        # all_examples ... all_examples in Orange format
        # ach_states ... numpy arrays of achievable states from each learning example
        self.all_states, self.all_examples, self.ach_states = self.preprocess(learn_states, max_depth)

    def achievable(self, state, depth):
        ach = [state]
        if depth == 0:
            return ach
        for ns in state.get_moves():
            ach += self.achievable(ns, depth-1)
        return ach

    def preprocess(self, learn_states, max_depth):
        all_states = {}
        ach_sets = [set() for s in learn_states]
        for si, s in enumerate(learn_states):
            if s not in all_states:
                all_states[s] = len(all_states)
            # find states that are achievable from s in max_depth
            ach = self.achievable(s, max_depth)
            for a in ach:
                if a not in all_states:
                    all_states[a] = len(all_states)
                ach_sets[si].add(all_states[a])
        state_list = sorted(all_states.keys(), key = lambda s: all_states[s])
        all_examples = ex.create_data_from_states(state_list, np.zeros(len(state_list)))
        ach_states = []
        for s in ach_sets:
            ach = np.zeros(len(all_examples), dtype=bool)
            ach[list(s)] = True
            ach_states.append(ach)
        return all_states, all_examples, ach_states

    def __call__(self, example, ix, goal):
        ach = self.ach_states[ix]
        achX = self.all_examples.X[ach]
        achieved = goal(example, achX)
        return np.any(achieved)


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
    def __init__(self, goal, parent, success, default, covered, traces):
        """
        A goal-node contains the actual goal, a link to its parent GoalNode, 
        the probability (success) to achieve the parent goal, and instances 
        (an array of ones and zeros) covered by this particular strategy. Traces
         is a set of traces covered by the strategy.
         default: it is the default probability of achieving the parent goal (used
         it to compare to parent)
        """
        self.goal = goal
        self.children = []
        self.parent = parent
        self.success = success
        self.default = default
        self.covered = covered
        self.expanded = False
        self.traces = traces

    def __str__(self):
        return self.str_rec(0)

    def str_rec(self, indent):
        gstr = ' ' * indent
        gstr += 'g: {}, succ: {:4.4f}, default: {:4.4f}, cov: {:4.4f}, ntraces: {}\n'.format(
            str(self.goal), self.success, self.default,
            float(sum(self.covered))/self.covered.shape[0],
            len(self.traces))
        for c in self.children:
            gstr += c.str_rec(indent+2)
        return gstr

    def goal2str(self):
        return 'g: {}, succ: {:4.4f}, default: {:4.4f}, cov: {:4.4f}, ntraces: {}'.format(
            str(self.goal), self.success, self.default,
            float(sum(self.covered))/self.covered.shape[0],
            len(self.traces))



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
        return closest, cl_depth + 1

class QualityValidator:
    def __init__(self, validator, threshold):
        self.validator = validator
        self.threshold = threshold

    def validate_rule(self, rule):
        if rule.quality < self.threshold:
            return False
        return self.validator.validate_rule(rule)

class TracesValidator:
    def __init__(self, general_validator, min_traces):
        self.general_validator = general_validator
        self.max_rule_length = self.general_validator.max_rule_length
        self.min_covered_examples = self.general_validator.min_covered_examples
        self.min_traces = min_traces # each rule has to cover at least this number of traces
        self.ex_traces = None # a numpy array of traces for each learning example
        self.cover_traces = None # a set of traces that should be covered
        self.positive = None

    def validate_rule(self, rule):
        if self.ex_traces is not None:
            # select covered positive examples and check their traces
            cov_traces = set(self.ex_traces[rule.covered_examples & self.positive]) & self.cover_traces
            if len(cov_traces) < self.min_traces:
                return False
        return self.general_validator.validate_rule(rule)


class RuleLearner:
    def __init__(self, nrules=5, min_cov=1, m=2, min_acc=0.0, implicit=False,
                 active=False, min_traces=0, default_alpha=1.0, parent_alpha=1.0,
                 max_rule_length=5):
        """

        :param nrules: maximal number of returned rules
        :param min_cov: minimal uniquely covered examples that a rule should have
        :param m:
        :param min_acc:
        :param implicit:
        :param active:
        :param min_traces: each rule should cover at least min_traces unique traces
        :return:
        """
        self.learner = rules.RulesStar(evc=False, width=100, m=m,
                default_alpha=default_alpha, parent_alpha=parent_alpha,
                max_rule_length=max_rule_length)
        self.learner.rule_validator = QualityValidator(self.learner.rule_validator, min_acc)
        self.learner.rule_finder.general_validator = TracesValidator(self.learner.rule_finder.general_validator, min_traces)
        self.nrules = nrules
        self.min_cov = min_cov
        self.implicit = implicit
        self.active = active
        self.min_traces = min_traces

    def __call__(self, examples, example_traces, cover_traces):
        """
        Learns a set of rules.

        :param examples: learning examples described with attributes and class describing whether
         goal is achievable or not.
        :param states: states corresponding to learning examples
        :param example_traces: traces of examples
        :param cover_traces: a set of traces that have to be covered by rules
        :return:
        """
        assert len(examples) == len(example_traces)

        # learn rules for goal=yes
        self.learner.target_class = "yes"
        # set traces to general validator
        self.learner.rule_finder.general_validator.ex_traces = example_traces
        self.learner.rule_finder.general_validator.cover_traces = cover_traces
        self.learner.rule_finder.general_validator.positive = examples.Y == 1

        # learn rules
        rules = self.learner(examples).rule_list
        if self.implicit:
            # add implicit conditions
            rules = self.add_implicit_conditions(rules, examples)

        # return only self.nrules rules (each rule must cover at least 
        # self.min_cov positive examples)
        # self,min_traces: each rule should have self.min_trace unique traces
        sel_rules = []
        all_covered = np.zeros(len(examples), dtype=bool)
        cov_traces = set()
        positive = examples.Y == 1
        for r in rules:
            if self.nrules >= 0 and len(sel_rules) >= self.nrules:
                break
            new_covered = r.covered_examples & ~all_covered
            new_covered &= positive
            new_traces = set(example_traces[new_covered])
            if len(new_traces - cov_traces) < self.min_traces:
                continue
            if np.count_nonzero(new_covered) >= self.min_cov:
                setattr(r, "new_covered_examples", examples.X[new_covered])
                sel_rules.append(r)
                all_covered |= new_covered
                cov_traces |= new_traces

        return sel_rules

    def add_implicit_conditions(self, rules, examples):
        """ This method adds implicit conditions to rules. """
        X, Y, W = examples.X, examples.Y, examples.W if examples.W else None
        Y = Y.astype(dtype=int)
        refiner = self.learner.rule_finder.search_strategy.refine_rule
        positive = Y == 1
        new_rules = []
        for rule in rules:
            print("rule", rule, rule.curr_class_dist)
            pos_covered = rule.covered_examples & positive
            # keep refining rule until id does not lose positive examples
            refined = True
            while refined:
                refined = False
                refined_rules = refiner(X, Y, W, rule)
                for ref_rule in refined_rules:
                    ref_rule.create_model()
                    print("ref", ref_rule, ref_rule.curr_class_dist)
                    print("val", self.learner.rule_finder.general_validator.validate_rule(ref_rule))
                    print("traces", self.learner.rule_finder.general_validator.general_validator.min_traces)
                    if np.array_equal(ref_rule.covered_examples & positive, pos_covered):
                        # set new rule, break
                        ref_rule.quality = rule.quality
                        rule = ref_rule
                        refined = True
                        break
            rule.create_model()
            new_rules.append(rule)
        return new_rules

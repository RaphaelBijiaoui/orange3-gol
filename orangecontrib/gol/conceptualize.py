import numpy as np
import collections
from Orange.classification.rules import Selector, Rule
from Orange.data import DiscreteVariable, Table
import orangecontrib.evcrules.rules as rules
import orangecontrib.gol.examples as ex

class BasicConceptualizer:
    """ Basic conceptualizer implements basic goal-oriented learning: 
    no continuous goals (increase/decrease),
    no and-or problems or for two-player games. """
    def __init__(self, rule_learner=None, select_goal=None, tree_depth=-1, search_depth=3):
        if not rule_learner:
            self.rule_learner = RuleLearner()
        else:
            self.rule_learner = rule_learner
        if not select_goal:
            self.select_goal = BreadthFirstSelector()
        else:
            self.select_goal = select_goal
        self.tree_depth = tree_depth
        self.search_depth = search_depth

    def __call__(self, learn_examples, learn_states, learn_traces, final_goal, target_trace=None):
        goal_validator = GoalValidator(learn_states, self.search_depth)

        # create initial goal tree
        covered = np.zeros(len(learn_examples), dtype=bool)
        if target_trace and target_trace > -1:
            traces = set([target_trace])
        else:
            traces = set(learn_traces)
        gtree = GoalNode(final_goal, None, 1.0, 1.0, covered, traces)
        while True:
            selected = self.select_goal(gtree, self.tree_depth)
            if not selected:
                break
            selected.expanded = True
            if np.all(selected.covered):
                continue

            # find solved, positive and negative examples
            indices = np.where(selected.covered == False)[0]
            if not np.any(selected.covered):
                prev_exemplar = np.ones(len(learn_examples), dtype=bool)
            else:
                prev_exemplar = selected.covered
            solved, pos_unsolved, exemplars = goal_validator(learn_examples, indices, prev_exemplar, selected.goal)
            # if there are any exemplars within solved, add exemplars
            # to the previous node and select node as unexpanded
            solved_exemplars = solved & exemplars
            if np.any(solved_exemplars):
                selected.covered[indices] |= solved_exemplars
                selected.expanded = False
                continue

            # create learning examples (remove solved examples)
            indices = indices[~solved]
            exemplars = exemplars[~solved]
            examples = Table.from_table_rows(learn_examples, indices)
            traces = learn_traces[indices]
            examples.Y = pos_unsolved[~solved]

            # learn rules
            rules = self.rule_learner(examples, exemplars, traces, selected.traces, selected.goal)
            for r in rules:
                new_goal = Goal(r)
                #success = r.quality
                success = r.curr_class_dist[r.target_class] / r.curr_class_dist.sum()
                default = sum(examples.Y) / examples.Y.shape[0]
                # only exemplars can be added as covered
                new_covered = np.array(selected.covered)
                new_covered[indices] = selected.covered[indices] | exemplars
                new_traces = set(traces[exemplars & r.covered_examples]) & selected.traces
                selected.children.append(GoalNode(new_goal, selected, success, 
                                         default, new_covered, new_traces))
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
        # ach_learn ... which learning examples are achievable 
        self.all_states, self.all_examples, self.ach_states, self.ach_learn = self.preprocess(learn_states, max_depth)

    def achievable(self, state, depth):
        ach = set()
        ach.add(state)
        horizon = ach
        for d in range(depth):
            new_ach = set()
            for s in horizon:
                new_ach |= set(s.get_moves())
            horizon = new_ach - ach
            ach |= horizon
        return ach

    def preprocess(self, learn_states, max_depth):
        # add learn states to all states
        set_learn = set(learn_states)
        all_states = {} # contains all states / first learn states are added / then all other
        index = 0
        for st in set_learn:
            all_states[st] = index
            index += 1

        state_ind = collections.defaultdict(list)
        for li, l in enumerate(learn_states):
            state_ind[l].append(li)

        ach_sets = [None] * len(learn_states)
        ach_learn = [None] * len(learn_states)
        for s in set_learn: 
            # find states that are achievable from s in max_depth
            ach = self.achievable(s, max_depth)
            # create indices of states
            ach_set = set()
            for a in ach:
                if a not in all_states:
                    all_states[a] = len(all_states)
                ach_set.add(all_states[a])
            for si in state_ind[s]:
                ach_sets[si] = ach_set

            # create a numpy array of achievable learning examples
            alearn = np.zeros(len(learn_states), dtype=bool)
            learn_ach = ach & set_learn 
            for a in learn_ach:
                alearn[state_ind[a]] = 1
            for si in state_ind[s]:
                ach_learn[si] = alearn

        state_list = sorted(all_states.keys(), key = lambda s: all_states[s])
        all_examples = ex.create_data_from_states(state_list, np.zeros(len(state_list)))
        ach_states = []
        for s in ach_sets:
            ach = np.zeros(len(all_examples), dtype=bool)
            ach[list(s)] = True
            ach_states.append(ach)
        return all_states, all_examples, ach_states, ach_learn

    def __call__(self, learn_examples, indices, prev_covered, goal):
        n = len(indices)
        solved = np.zeros(n, dtype=bool)
        positive = np.zeros(n, dtype=bool)
        exemplars = np.zeros(n, dtype=bool)
        for i, ix in enumerate(indices):
            example = learn_examples[ix]
            ach = self.ach_states[ix]
            solved[i] = goal.achieved(example)
            if not solved[i]:
                achX = self.all_examples.X[ach]
                achieved = goal(example, achX)
                positive[i] = np.any(achieved)
            if solved[i] or positive[i]:
                exemplars[i] = np.any(self.ach_learn[ix] & prev_covered)
        return solved, positive, exemplars


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

    def achieved(self, state_instance):
        """ Is goal achieved in state instance? """
        return self.static.evaluate_instance(state_instance)

    def selectors(self):
        return set(self.static.selectors)

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
        gstr += self.goal2str() + '\n'
        for c in self.children:
            gstr += c.str_rec(indent+2)
        return gstr

    def goal2str(self):
        return 'g: {}, succ: {:4.4f}, default: {:4.4f}, cov: {:4.4f}, ntraces: {}, traces: {}'.format(
            str(self.goal), self.success, self.default,
            float(sum(self.covered))/self.covered.shape[0],
            len(self.traces), str(self.traces))



class BreadthFirstSelector:
    """ 
    Class returns the unexpanded goal that is the closest to the root. 
    """
    def __call__(self, goal_tree, tree_depth):
        goal, depth = self.find_closest_unexpanded(goal_tree)
        if tree_depth>-1 and depth >= tree_depth:
            return None
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
        self.additional_validator = None
        self.min_traces = min_traces # each rule has to cover at least this number of traces
        self.ex_traces = None # a numpy array of traces for each learning example
        self.cover_traces = None # a set of traces that should be covered
        self.pos = None
        self.exemplars = None

    def validate_rule(self, rule):
        # each rule should cover at least min_traces traces
        # and at least one exemplar

        # exemplar
        exemp = rule.covered_examples & self.exemplars
        if not np.any(exemp):
            return False

        # traces should continue with at least one exemplar
        cov_traces = set(self.ex_traces[exemp]) & self.cover_traces
        if len(cov_traces) < 1:
            return False

        # pattern should be found in different traces
        pos_cov = rule.covered_examples & self.pos
        cov_traces = set(self.ex_traces[pos_cov])
        if len(cov_traces) < self.min_traces:
            return False

        if self.additional_validator and not self.additional_validator.validate_rule(rule):
            return False
        
        if self.general_validator:
            return self.general_validator.validate_rule(rule)

        return True

class MEstimateEvaluator(rules.Evaluator):
    def __init__(self, m=2):
        self.m = m
        self.selectors = set()

    def evaluate_rule(self, rule):
        tc = rule.target_class
        dist = rule.curr_class_dist
        target = dist[tc]
        p_dist = rule.initial_class_dist
        pa = p_dist[tc] / p_dist.sum()
        dsum = dist.sum()
        rf = target / dsum
        if rf < pa:
            return rf
        eps = 0
        for s in rule.selectors:
            if s in self.selectors:
                eps += 0.01
        return (target + self.m * pa) / (dist.sum() + self.m) + eps

class RuleLearner:
    def __init__(self, nrules=5, m=2, min_acc=0.0, implicit=False,
                 active=False, min_covered_examples=1, min_traces=1, unique_cov=1, unique_traces=0, 
                 default_alpha=1.0, parent_alpha=1.0, max_rule_length=5):
        """

        :param nrules: maximal number of returned rules
        :param min_cov: minimal uniquely covered examples that a rule should have
        :param m:
        :param min_acc:
        :param implicit:
        :param active:
        :param min_traces: each rule should cover at least min_traces traces
        :param unique_cov: 
        :param unique_traces: each rule should cover at least this many unique traces
        :return:
        """
        self.learner = rules.RulesStar(evc=False, width=100, m=m,
                default_alpha=default_alpha, parent_alpha=parent_alpha,
                max_rule_length=max_rule_length, min_covered_examples=min_covered_examples)
        self.learner.rule_validator = QualityValidator(self.learner.rule_validator, min_acc)
        self.learner.rule_finder.general_validator = TracesValidator(self.learner.rule_finder.general_validator, min_traces)
        self.learner.evaluator = MEstimateEvaluator()
        self.nrules = nrules
        self.implicit = implicit
        self.active = active
        self.min_traces = min_traces

    def __call__(self, examples, exemplars, example_traces, cover_traces, parent_goal):
        """
        Learns a set of rules.

        :param examples: learning examples described with attributes and class describing whether
         goal is achievable or not.
        :param pos_nonsolved: positive examples that are not solved with the current goal
        :param example_traces: traces of examples
        :param cover_traces: a set of traces that were covered by previous goal
        :param exemplars: examples that can be used as typical instances of each learned rule
        :return:
        """
        assert len(examples) == len(example_traces) == len(exemplars)

        # learn rules for goal=yes
        self.learner.target_class = "yes"
        # set traces to general validator
        self.learner.rule_finder.general_validator.ex_traces = example_traces
        self.learner.rule_finder.general_validator.cover_traces = cover_traces
        self.learner.rule_finder.general_validator.exemplars = exemplars
        self.learner.rule_finder.general_validator.pos = examples.Y == 1
        self.learner.evaluator.selectors = parent_goal.selectors()

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
        #teprint("len:", len(rules), len(examples), examples.Y.sum(), exemplars.sum())
        for r in rules:
            #if self.nrules >= 0 and len(sel_rules) >= self.nrules:
            #    break
            new_covered = r.covered_examples & ~all_covered
            new_covered &= exemplars
            new_traces = set(example_traces[new_covered]) & cover_traces
            if len(new_traces - cov_traces) >= 1: #self.min_traces:
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
            #rule.create_model()
            #print("rule", rule, rule.curr_class_dist)
            # a trick to allow arbitrary long rules
            gv = rule.general_validator.general_validator
            rule.general_validator.general_validator = None

            pos_covered = rule.covered_examples & positive
            # keep refining rule until id does not lose positive examples
            refined = True
            while refined:
                refined = False
                refined_rules = refiner(X, Y, W, rule)
                for ref_rule in refined_rules:
                    #ref_rule.create_model()
                    #print("ref", ref_rule, ref_rule.curr_class_dist)
                    #print("val", self.learner.rule_finder.general_validator.validate_rule(ref_rule))
                    #print("traces", self.learner.rule_finder.general_validator.general_validator.min_traces)
                    if np.array_equal(ref_rule.covered_examples & positive, pos_covered):
                        # set new rule, break
                        ref_rule.quality = rule.quality
                        rule = ref_rule
                        refined = True
                        break
            rule.general_validator.general_validator = gv
            new_rules.append(rule)
        return new_rules

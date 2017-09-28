import numpy as np
from Orange.data import DiscreteVariable
from Orange.classification.rules import Selector
from orangecontrib.gol.rule import RRule


class Goal:
    """ Class represents goals. At the moment only static goals can be 
    represented: goals that can be represented with the Orange's implementation
    of a rule, that is, a list of selectors. Dynamic goals (change in terms
    of increase / decrease) is not supported yet. """

    def __init__(self, rule):
        self.rule = rule
        self.str_rule = "Goal: {}".format(str(self.rule))
        self.hash_rule = hash(self.str_rule)

    def __call__(self, instances):
        """ 
        instances: an Orange data table describing states. 
        """
        return self.rule.evaluate_data(instances.X)

    def selectors(self):
        return set(self.rule.selectors)

    def __str__(self):
        return self.str_rule

    def __eq__(self, other):
        return self.str_rule == other.str_rule

    def __hash__(self):
        return self.hash_rule

    @staticmethod
    def from_conditions_factory(domain, conditions):
        selectors = []
        for c in conditions:
            column = domain.index(c[0])
            feature = domain[column]
            if isinstance(feature, DiscreteVariable):
                value = feature.values.index(c[2])
            else:
                value = c[2]
            selectors.append(Selector(column, c[1], value))
        rule = RRule(0, selectors=selectors, domain=domain)
        return Goal(rule)


class GoalValidatorDepth:
    """
    Validate whether a goal is achievable in a certain state.
    """
    def __init__(self, local_search_depth = 4):
        self.goal_ach = {}
        self.lsd = local_search_depth

    def initialize(self, goal, examples, states):
        # find examples, where goal is already achieved, then incrementally
        # set the minimum distance to goal for the following examples in trace.
        ach = {}
        solved = goal(examples)
        solved = np.where(solved)[0]

        # set of all states for which we know distance to goal
        solved_states = set(states[s] for s in solved)         
        for ss in solved_states:
            ach[str(ss.get_id())] = 0
        # current sets from where we continue searching
        current_states = set(solved_states)
        unsolved_states = set(states) - solved_states
        distance = 0
        while unsolved_states:
            distance += 1
            # generate new states from current_states
            new_solved_states = set()
            for ss in current_states:
                new_solved_states |= set(ss.get_next_states())

            # update new solved (only those that are not solved yet)
            new_solved_states -= solved_states
            for nss in new_solved_states:
                ach[str(nss.get_id())] = distance

            # update track of states
            solved_states |= new_solved_states
            current_states = new_solved_states & unsolved_states
            unsolved_states -= new_solved_states
        return ach


    def bfs(self, example):
        """ Returns a dictionary of all examples close to example
        and their distances to example. """
        return None

    def __call__(self, goal, learn_examples, learn_states, covered):
        if goal not in self.goal_ach:
            self.goal_ach[goal] = self.initialize(goal, learn_examples, learn_states)
        complexities = np.zeros(len(learn_examples), dtype=np.float32)
        for ei in np.where(~covered)[0]:
            ex = learn_examples[ei]
            # do we already know the distance from this example to goal?
            eid = str(ex['id'])
            if eid in self.goal_ach[goal]:
                dist = self.goal_ach[goal][eid]
                complexities[ei] = 0 if dist <= 3 else 2**dist # self.goal_ach[goal][eid]**2
            else: 
                # need to determine examples distance to goal
                # TODO!!!
                close_examples = self.bfs(ex)
                for ce in close_examples:
                    pass
        return complexities


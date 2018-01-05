"""
Classes related to goal representation and goal manipulation in
goal-oriented learning.
"""

import numpy as np
from Orange.data import DiscreteVariable
from Orange.classification.rules import Selector
from orangecontrib.gol.rule import RRule
from orangecontrib.gol.examples import create_data_from_states


class Goal:
    """ Class represents goals. At the moment only static goals can be
    represented: goals that can be represented with the Orange's implementation
    of a rule, that is, a list of selectors. Dynamic goals (change in terms
    of increase / decrease) are not supported yet. """

    def __init__(self, rule):
        self.rule = rule
        self.selectors = tuple(sorted(set(rule.selectors)))

    def __call__(self, instances):
        """
        instances: an Orange data table describing states.
        """
        return self.rule.evaluate_data(instances.X)

    def selectors(self):
        """ Returns selectors of goal rule. """
        return self.selectors

    def __str__(self):
        return "Goal: {}".format(str(self.rule))

    def __eq__(self, other):
        return self.selectors == other.selectors

    def __hash__(self):
        return hash(self.selectors)

    @staticmethod
    def from_conditions_factory(domain, conditions):
        """ Creates a goal from a list of conditions. """
        selectors = []
        for cond in conditions:
            column = domain.index(cond[0])
            feature = domain[column]
            if isinstance(feature, DiscreteVariable):
                value = feature.values.index(cond[2])
            else:
                value = cond[2]
            selectors.append(Selector(column, cond[1], value))
        rule = RRule(None, None, selectors=selectors, domain=domain)
        return Goal(rule)


class GoalValidatorExponentialDepth:
    """
    1. First computes search depth required in instances to achieve a certain goal.
    2. Complexity of the instances corresponds to exponential value of depth.
    """
    def __init__(self, base=2, zero_depth=10, search_depth=5, inf_depth=100):
        """
        complexity = base ** (depth - zero_depth)

        Keyword Arguments:
        base -- base in exponent
        zero_depth -- if depth to reach goal is less or equal to
            this value, complexity of instance equals 0.
        search_depth -- minimal search depth used in goal validation
        """
        self.goal_ach = {}
        self.search_depth = search_depth
        self.base = base
        self.zero_depth = zero_depth
        self.inf_depth=inf_depth

    def initialize(self, goal, examples, states):
        """
        Find examples, where goal is already achieved, then incrementally
        set the minimum distance to goal for the following examples in trace.
        """
        ach = {}
        solved = goal(examples)
        solved = np.where(solved)[0]

        # set of all states for which we know distance to goal
        solved_states = set(states[s] for s in solved)
        for ss in solved_states:
            ach[ss] = 0
        # current sets from where we continue searching
        unsolved_states = set(states) - solved_states
        distance = 0
        while unsolved_states:
            distance += 1
            # from each unsolved state generate new states and check whether 
            # they are in solved
            new_solved = set()
            for ss in unsolved_states:
                next_states = set(ss.get_next_states())
                if solved_states & next_states:
                    new_solved.add(ss)
                    ach[ss] = distance
            # update solved and unsolved
            solved_states |= new_solved
            unsolved_states -= new_solved
        return ach


    def bfs(self, state, goal):
        """ Runs bfs till self.search_depth and find best depth for state. """
        queue = [state]
        visited = {state:0}
        depth = 0
        while queue and depth <= 5:
            depth += 1
            curr, queue = queue[0], queue[1:]
            for new_state in curr.get_next_states():
                if new_state not in visited:
                    queue.append(new_state)
                    visited[new_state] = depth
        # create instances
        states = list(visited.keys())
        traces = np.full(len(states), -1)
        data = create_data_from_states(states, traces)
        solved = goal(data)
        # compute best depth for this state
        best = None
        for si, st in enumerate(states):
            if solved[si]:
                if best is None or visited[st] < best:
                    best = visited[st]
            elif st in self.goal_ach[goal]:
                if best is None or visited[st] + self.goal_ach[goal] < best:
                    best = visited[st] + self.goal_ach[goal]
        return best

    def __call__(self, goal, learn_examples, learn_states, covered):
        """ Determines complexity of examples. """
        if goal not in self.goal_ach:
            self.goal_ach[goal] = self.initialize(goal, learn_examples, learn_states)
        complexities = np.zeros(len(learn_examples), dtype=np.float32)
        for ei in np.where(~covered)[0]:
            state = learn_states[ei]
            # do we already know the distance from this example to goal?
            if state in self.goal_ach[goal]:
                dist = self.goal_ach[goal][state]
            else:
                # Need to determine depth for this example. This is usually
                # needed for examples, which were added later (active learning).
                dist = self.bfs(state, goal)
                if dist is None:
                    dist = self.inf_depth + 1
                self.goal_ach[goal][state] = dist
            # determine complexity
            if dist > self.zero_depth:
                complexities[ei] = self.base ** (dist - self.zero_depth)
        return complexities



class GoalSelector:
    """ A class that selects relevant parent goals for a rule.  """
    def __init__(self, all_goals, all_complexities, min_examples):
        """
        all_goals -- all possible goals
        all_complexities -- complexities for each goal
        min_examples -- minimal number of examples for each goal
        """
        self.all_goals = all_goals
        self.all_complexities = all_complexities
        self.min_examples = min_examples

    def __call__(self, goals, covered):
        """
        goals -- id of goals
        covered -- indexes of covered examples
        """
        new_goals = goals[:]
        if len(new_goals) == 0:
            return [], None
        if len(new_goals) == 1:
            return new_goals, self.all_complexities[new_goals[0]]
        all_goals_good = False
        while not all_goals_good and len(new_goals) > 1:
            all_goals_good = True
            min_complexities = self.get_min_values(new_goals)
            to_remove = None
            for ni in new_goals:
                gcompx = self.all_complexities[ni][covered]
                if (gcompx <= min_complexities[covered]).sum() < self.min_examples:
                    to_remove = ni
                    break
            if to_remove is not None:
                new_goals.remove(to_remove)
                all_goals_good = False
        return new_goals, min_complexities

    def get_min_values(self, goals):
        """ Return minimal complexities given a list of goals. """
        min_complexities = self.all_complexities[goals[0]]
        for ni in goals[1:]:
            min_complexities = np.minimum(min_complexities, self.all_complexities[ni])
        return min_complexities

""" Implementation of goal oriented learning. """

import numpy as np
from Orange.data import Table
from orangecontrib.gol.goal import GoalValidatorExponentialDepth, Goal, GoalSelector
from orangecontrib.gol.rule_learner import RegressiveRuleLearner

EPS = 1e-3
class GraphConceptualizer:
    """ Main class that implements the main algorithm of goal oriented learning. """
    def __init__(self, rule_learner=None, goal_validator=None):
        self.rule_learner = rule_learner if rule_learner is not None else \
                            RegressiveRuleLearner()
        self.goal_validator = goal_validator if goal_validator is not None else \
                              GoalValidatorExponentialDepth()

    def __call__(self, learn_examples, learn_states, learn_traces, final_goal):

        # determine which examples can be used as learning examples
        # (those that have a final state)
        covered = final_goal(learn_examples)
        traces = set(learn_traces[covered])
        final_examples = np.in1d(learn_traces, list(traces))
        learn_examples = Table.from_table_rows(learn_examples, final_examples)
        learn_traces = learn_traces[final_examples]
        learn_states = learn_states[final_examples]
        # create initial goal graph
        complexities = self.goal_validator(final_goal, learn_examples,
                                           learn_states, covered)
        print("complexities")
        # const containing maximal possible complexity of learning examples
        HIGH_COMP_CONST = np.max(complexities) + EPS
        ggraph = GoalNode(final_goal, 0, [], complexities, traces)
        while True:
            print("covered", covered.sum())
            if np.all(covered):
                break

            indices = ~covered
            
            # create current learning examples
            curr_examples = Table.from_table_rows(learn_examples, indices)
            curr_traces = learn_traces[indices]
            
            # learn a rule
            all_complexities = [n.complexities[indices] for n in ggraph.get_nodes()]
            rule = self.rule_learner.fit_storage(curr_examples, all_complexities)
            
            #nodes = list(ggraph.get_nodes())
            #nodes.sort(key=lambda n: np.mean(n.complexities[indices]), reverse=True)
            #print(nodes)
            
            
            # select relevant goals
            #gsel = GoalSelector(nodes, [n.complexities for n in nodes],
            #                    self.min_transition_examples)
            node_indexes, _ = gsel(list(range(len(nodes))), indices)
            # create a list of all goal complexities in nodes
            nodes = [nodes[ni] for ni in node_indexes]
            all_complexities = [n.complexities[indices] for n in nodes]

            # create current learning examples
            curr_examples = Table.from_table_rows(learn_examples, indices)
            curr_traces = learn_traces[indices]

            # learn a rule
            goal_selector = GoalSelector(nodes, all_complexities, self.min_transition_examples)
            rule = self.rule_learner(curr_examples, goal_selector)
            print("Learned rule: ", rule)

            mean_complexity = rule.quality
            parents = [nodes[ni] for ni in rule.predicted_goals]
            pruned_complexities = nodes[rule.predicted_goals[0]].complexities
            for ni in rule.predicted_goals[1:]:
                pruned_complexities = np.minimum(pruned_complexities, nodes[ni].complexities)

            # create new goal
            goal = Goal(rule)
            # compute complexities for new goal
            goal_complexities = self.goal_validator(goal, learn_examples,
                                                    learn_states, covered)
            # examples from other (not covered) traces should have goal_complexity
            # set to HIGH_COMP_CONST
            covered_traces = set(curr_traces[rule.covered_examples])
            not_same_trace = ~np.in1d(learn_traces, list(covered_traces))
            goal_complexities[not_same_trace] = HIGH_COMP_CONST
            # create new goal node
            new_node = GoalNode(goal, mean_complexity, parents,
                                mean_complexity + goal_complexities, covered_traces)
            # update children
            for p in parents:
                p.children.append(new_node)

            # calculate new covered
            covered[indices] |= rule.covered_examples
            # add to covered also those examples that are closer to new goal
            # than the distance between goal and previous goals
            close_goal = goal_complexities <= rule.mean
            close_before = pruned_complexities <= rule.mean
            covered |= close_goal & close_before
            print("Current graph: ")
            print(ggraph)
        return ggraph


class GoalNode:
    """ Class represents a single node in goal graph. """
    def __init__(self, goal, mean_complexity, parents, complexities, traces):
        """
        A goal-node contains the actual goal, links to its parent nodes,
        instances (an array of ones and zeros) covered by this particular strategy.
        Traces is a set of traces covered by the strategy.
        Covered are indexes of covered examples by this strategy.
        """
        self.goal = goal
        self.children = []
        self.parents = parents
        self.traces = traces
        self.complexities = complexities
        self.mean_complexity = mean_complexity

    def get_nodes(self):
        """ Returns a set of all nodes in this graph. """
        nodes = [self]
        for c in self.children:
            nodes.extend(c.get_nodes())
        return nodes

    def __str__(self):
        """ String representation of graph. """
        nodes = self.get_nodes()
        nodes.sort(key=lambda gn: gn.goal.rule.mean)
        gstr = ''
        for node in nodes:
            gstr += '{}, complexity={}, parents:\n'.format(node.goal2str(), node.mean_complexity)
            for p in node.parents:
                gstr += '  ' + p.goal2str() + '\n'
            gstr += 'end parents\n'
        return gstr

    def __eq__(self, other):
        return self.goal == other.goal

    def __hash__(self):
        return self.goal.__hash__()

    def str_rec(self):
        """ Return string representation of this goal. """
        nodes = list(self.get_nodes())
        nodes.sort(key=lambda gn: gn.goal.rule.mean)
        gstr = ''
        for node in nodes:
            gstr += '{}, complexity={}, parents:\n'.format(node.goal2str(), node.mean_complexity)
            for p in node.parents:
                gstr += '  ' + p.goal2str() + '\n'
            gstr += 'end parents\n'
        return gstr

    def goal2str(self):
        """  Return string representation of this goal. """
        return 'g: {}'.format(
            str(self.goal))

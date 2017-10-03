"""
A simple example of learning a strategy for 3x3 puzzle.
"""

import os
import pickle
from Orange.data import Table
from orangecontrib.gol.goal_graph import GraphConceptualizer
from orangecontrib.gol.goal import Goal, GoalValidatorExponentialDepth
from orangecontrib.gol.rule_learner import RegressiveRuleLearner

import orangecontrib.gol.examples as ex
import orangecontrib.gol.domains.eight.eight_puzzle as eight_puzzle

data = eight_puzzle.EightPuzzleDomain()
if not os.path.isfile("eight.tab"):
    example_states, example_traces = data.get_examples_from_traces(1000)
    pickle.dump((example_states, example_traces), open("eight.pickle", "wb"))
    examples = ex.create_data_from_states(example_states, example_traces)
    examples.save("eight.tab")

example_states, example_traces = pickle.load(open("eight.pickle", "rb"))
examples = Table("eight.tab")

# create rule learner:
# * each goal should cover at least 200 examples,
# * each transition between goals should represent at least 100 examples
# * the significance of each condition of rule should be at least 0.05 (t-test)
rule_learner = RegressiveRuleLearner(min_covered_examples=500,
                                     min_transition_examples=100,
                                     cond_alpha=0.05, k=100)

# create goal validator (a class for estimating goal complexity)
goal_validator = GoalValidatorExponentialDepth(base=2.6, inf_complexity_depth=12,
                                               search_depth=5)

# create conceptualizer
conceptualizer = GraphConceptualizer(rule_learner=rule_learner,
                                     goal_validator=goal_validator)
# create final goal
conditions = [
    ("1_0", "==", "yes"),
    ("2_1", "==", "yes"),
    ("3_2", "==", "yes"),
    ("4_3", "==", "yes"),
    ("5_4", "==", "yes"),
    ("6_5", "==", "yes"),
    ("7_6", "==", "yes"),
    ("8_7", "==", "yes")]
final_goal = Goal.from_conditions_factory(examples.domain, conditions)
# learn goal graph
strategy = conceptualizer(examples, example_states, example_traces, final_goal)
print(strategy)

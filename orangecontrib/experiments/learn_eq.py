""" Learning a strategy for solving equations with one unknown. """

import orangecontrib.gol.domains.equation.single as eq
from orangecontrib.gol.rule_learner import RegressiveRuleLearner
from orangecontrib.gol.goal import Goal, GoalValidatorExponentialDepth
from orangecontrib.gol.goal_graph import GraphConceptualizer

domain = eq.Eq1Domain()
examples, example_states, example_traces = domain.create_learning_examples()

rule_learner = RegressiveRuleLearner(min_covered_examples=5, parent_alpha=1.0,
                                     C=1, max_complexity=0)
goal_validator = GoalValidatorExponentialDepth(zero_depth=1, search_depth=5)

conceptualizer = GraphConceptualizer(rule_learner=rule_learner,
                                     goal_validator=goal_validator)

final_goal = Goal.from_conditions_factory(examples.domain, [("solved", "==", "yes")])

strategy = conceptualizer(examples, example_states, example_traces, final_goal)
print(strategy)

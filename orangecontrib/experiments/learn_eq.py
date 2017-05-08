import Orange

import orangecontrib.gol.conceptualize as concept
import orangecontrib.gol.examples as ex
import orangecontrib.gol.domains.equation.single as eq

MAX_DEPTH = 3
NRULES = 3

sg = concept.BreadthFirstSelector()
rl = concept.RuleLearner(nrules=NRULES, m=0, min_acc=0.99)
conceptualizer = concept.BasicConceptualizer(rule_learner=rl,
                                              select_goal=sg)

all_states = eq.State.get_all_states()
examples = ex.create_data(all_states)
final_goal = concept.Goal.create_initial_goal(examples.domain, [("solved", "==", "yes")])

strategy = conceptualizer(examples, all_states, final_goal, MAX_DEPTH)
print(strategy)



import collections

import orangecontrib.gol.conceptualize as concept
import orangecontrib.gol.examples as ex
import orangecontrib.gol.domains.prolog.prolog_state_trace as prolog

PROBLEM_ID = 2
MAX_DEPTH = 2
MIN_TRACES = 5
NRULES = -1

class PositiveConditions:
    def __init__(self, general_validator):
        self.general_validator = general_validator
        self.max_rule_length = general_validator.max_rule_length
        self.min_covered_examples = general_validator.min_covered_examples

    @property
    def ex_traces(self):
        return self.general_validator.ex_traces

    @ex_traces.setter
    def ex_traces(self, traces):
        self.general_validator.ex_traces = traces

    @property
    def positive(self):
        return self.general_validator.positive

    @positive.setter
    def positive(self, traces):
        self.general_validator.positive = traces

    @property
    def cover_traces(self):
        return self.general_validator.cover_traces

    @cover_traces.setter
    def cover_traces(self, traces):
        self.general_validator.cover_traces = traces

    def validate_rule(self, rule):
        #  rules should have positive conditions (val = 1.0 equals value "yes")
        for att, op, val in rule.selectors:
            if (op == "!=" and val > 0.5 or
                op == "==" and val < 0.5):
                return False
        return self.general_validator.validate_rule(rule)

sg = concept.BreadthFirstSelector()
rl = concept.RuleLearner(nrules=NRULES, m=2, min_traces=MIN_TRACES,
                         min_cov = MIN_TRACES, max_rule_length=10, 
                         parent_alpha=0.1, default_alpha=0.001, implicit=True)#, parent_alpha=0.1, default_alpha=0.01)
rl.learner.rule_finder.general_validator = PositiveConditions(rl.learner.rule_finder.general_validator)

conceptualizer = concept.BasicConceptualizer(rule_learner=rl,
                                              select_goal=sg)

# problem ids: member(1), conc(2), del(3)
data = prolog.PrologData(PROBLEM_ID)
examples, example_states, example_traces = ex.create_data(data)
final_goal = concept.Goal.create_initial_goal(examples.domain, [("solved", "==", "yes")])

strategy = conceptualizer(examples, example_states, example_traces, final_goal, MAX_DEPTH)
print(strategy)

attrs = open("../gol/domains/prolog/data/attributes_{}.tab".format(PROBLEM_ID), "rt")
attr_names = {}
for l in attrs:
    k, val = l.strip().split('\t')
    attr_names[k] = val

# for each trace, find the most interesting strategy (always select the first
# strategy covering this trace
traces = set(example_traces)
trcnt = collections.Counter()
for t in traces:
    node, nodes = strategy, []
    while node:
        nodes.append(node)
        children = node.children
        node = None
        for c in children:
            if t in c.traces:
                node = c
                break
    # print strategy
    str_rep = "\n".join(n.goal2str() for n in nodes)

    if not nodes[-1].children and trcnt[str_rep] == 0: # we have a complete strategy
        print("trace: ", t)
        print(str_rep)
        print()
        for n in nodes:
            # print goal
            print(n.goal2str())
            # print attributes
            for att, op, val in n.goal.static.selectors:
                if "a{}".format(att) in attr_names:
                    print("a{} = {}".format(att, attr_names["a{}".format(att)]))
            # print solution
            rule = n.goal.static
            for e in examples:
                if int(e["trace"]) == t and rule.evaluate_instance(e):
                    id_inst = e["id"]
                    print(id_inst, e["trace"])
                    print(data.id2sol[int(str(id_inst))])
                    break


    trcnt[str_rep] += 1

"""for k, v in trcnt.most_common():
    #if v >= MIN_TRACES:
    print(k)
    print()"""


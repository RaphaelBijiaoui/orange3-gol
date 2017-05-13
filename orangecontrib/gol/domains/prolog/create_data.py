import pickle
import collections
import argparse

from prolog.util import parse
from patterns import get_patterns

# load pickled program traces
traces = pickle.load(open('data/export.pickle', 'rb'))
parser = argparse.ArgumentParser()
parser.add_argument("pid", type=int, help="Problem ID")
args = parser.parse_args()
SELECTED = args.pid

# gather all attributes first
# over all student traces
patterns = collections.Counter()
for key in traces:
    # select only SELECTED problem
    if key[0] != SELECTED:
        continue

    tr = traces[key]
    for str_sol, lex, solved in tr:
    # extract attributes from training data
        for pat, nodes in get_patterns(str_sol, ["all"]):
            patterns[pat] += 1

# save attributes to file
attrs = []
with open('data/attributes_{}.tab'.format(SELECTED), 'w') as pattern_file:
    for i, (pat, count) in enumerate(patterns.most_common()):
        if count < 10:
            break
        attrs.append(pat)
        print('a{}\t{}'.format(i, pat), file=pattern_file)

# again, lets go over traces, create examples and create graph
id_counter = 0 # example counter
trace_id = 0
state_graph = collections.defaultdict(set)
lex_to_id = {}
examples = []
for key in traces:
    # select only SELECTED problem
    if key[0] != SELECTED:
        continue
    trace_id += 1

    prev_id = None
    tr = traces[key]
    for str_sol, lex, solved in tr:
        # if program is not parseable, skip it
        if not parse(str_sol) and str_sol.strip():
            continue

        # store lex (if new)
        lex = tuple(lex)
        if lex not in lex_to_id:
            id_counter += 1
            lex_to_id[lex] = id_counter
        # get id of current example
        this_id = lex_to_id[lex]
        
        # if applicable add a connection in state graph between previous id and
        # this id
        if prev_id:
            state_graph[prev_id].add(this_id)
        # now store prev_id, so that next time we can add successor
        prev_id = this_id

        # create attributes for this example
        expatts = set()
        for pat, nodes in get_patterns(str_sol, ["all"]):
            if pat in patterns:
                expatts.add(pat)

        # each learning example has only one trace_id
        # if the same state was visited several times, then have several 
        # examples in the data
        example = (this_id, trace_id, str_sol, expatts, lex)
        examples.append(example)

        # if program is a solution, then should add a connection to id=0 (main goal)
        if solved:
            state_graph[this_id].add(0)
            # after a solution is found, break loop, following programs are not relevant
            break

    trace_id += 1

print("Number of states: {}".format(len(state_graph)))
print("Number of learning examples: {}".format(len(examples)))
pickle.dump((state_graph, lex_to_id, patterns, examples), open("data/problem-{}.pickle".format(SELECTED), "wb"))


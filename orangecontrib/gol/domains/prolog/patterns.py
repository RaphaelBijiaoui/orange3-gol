# CodeQ: an online programming tutor.
# Copyright (C) 2016,2017 UL FRI
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import collections
from itertools import chain, combinations, product
import pickle
import random
import sys
import argparse
from os import makedirs

from nltk import ParentedTree, Tree

from prolog.util import parse as prolog_parse, Token

# construct pattern to match the structure of nodes given by [include],
# supports variables and literals
def pattern(node, include):
    if isinstance(node, Token):
        if any(n is node for n in include):
            return '"{}"'.format(node.val)
        return None
    
    if not isinstance(node, Tree):
        return None

    label = node.label()
    if any(n is node for n in include):
        if label == 'literal':
            return '"{}"'.format(node[0].val)
        if label == 'variable':
            return '{}'.format(label)
        return None
    if label == 'functor':
        # get arity
        arity = 0
        tmp = node._parent
        if tmp.label() == 'compound':
            while isinstance(tmp, Tree) and len(tmp) == 2:
                arity += 1
                tmp = tmp[1]
        return '({} "{}/{}")'.format(label, node[0].val, arity)

    subpats = [pattern(child, include) for child in node]
    pat = None
    if any(subpats):
        if label == 'and':
            if subpats[1]:
                pat = subpats[1]
            if subpats[0]:
                if pat:
                    pat = subpats[0] + ' ' + pat
                else:
                    pat = subpats[0]
        elif label == 'or': # should have only one path, otherwise None
            if subpats[0] and subpats[1]:
                return None
        elif label == 'args':
            pat = label
            for i, subpat in enumerate(subpats):
                if subpat:
                    pat += ' {}'.format(subpat)
            pat = '(' + pat + ')'
        elif label == 'unop':
            pat = '(' + label + ' ' + node[0].val + ' ' + subpats[1] + ')'
        elif label == 'binop':
            pat = label
            pat += ' "{}"'.format(node[1].val)
            if subpats[0]:
                pat += ' {}'.format(subpats[0])
            #pat += ' "{}"'.format(node[1].val)
            if subpats[2]:
                pat += ' {}'.format(subpats[2])
            pat = '(' + pat + ')'
        elif label == 'clause':
            pat = label
            for i, subpat in enumerate(subpats):
                if subpat:
                    pat += ' {}'.format(subpats[i])
            return '(' + pat + ')'
        elif label == 'compound':
            if len(subpats) > 1 and subpats[1]:
                pat = label
                for i, subpat in enumerate(subpats):
                    pat += ' {}'.format(subpat)
                pat = '(' + pat + ')'
            else:
                return None
        elif label == 'head':
            pat = label
            pat += ' {}'.format(subpats[0])
            pat = '(' + pat + ')'
        elif label == 'list':
            pat = 'list '
            if subpats[0]:
                pat += '(h {})'.format(subpats[0])
            if subpats[0] and subpats[1]:
                pat += ' '
            if subpats[1]:
                pat += '(t {})'.format(subpats[1])
            pat = '(' + pat + ')'
        if not pat:
            for s in subpats:
                if s:
                    pat = s
                    break
    return pat

def get_vars_between(node, start, end):
    nodes = [node]
    bet = False if start else True
    variables = set()
    while nodes:
        cnode = nodes.pop(0)
        if not isinstance(cnode, ParentedTree):
            continue
        if cnode == start:
            bet = True
        elif cnode == end:
            return variables
        else:
            if bet and cnode.label() == 'variable':
                variables.add(cnode[0].val)
            nodes = [child for child in cnode] + nodes
    return variables # when n2 is not found

def get_binop(node, variables):
    """ Find a binop node with one of the 
    given variables.  """
    if not isinstance(node, ParentedTree):
        return None, None
    label = node.label()
    if label == 'or' or label == 'unop' or \
            label == 'compound' and node[0].label() == 'functor' and \
            node[0][0].val == 'not':
        return None, None
    if label == 'binop' and node[1].val == '=' and node.parent().label() in ['and', 'clause']:
        if isinstance(node[0], ParentedTree) and \
                node[0].label() == 'variable' and \
                node[0][0].val in variables:
            return node[0], node[2]
        if isinstance(node[2], ParentedTree) and \
                node[2].label() == 'variable' and \
                node[2][0].val in variables:
            return node[2], node[0]
    for child in node:
        a, b = get_binop(child, variables)
        if a:
            return a, b
    return None, None

def find_var(node, variable):
    if not isinstance(node, Tree):
        return None
    if node == variable:
        return node
    for child in node:
        v = find_var(child, variable)
        if v:
            return v
    return None

def replace(node, variable, value):
    """ Replace variable node with another value node. """
    if not isinstance(node, Tree):
        return
    for i, child in enumerate(node):
        if child == variable:
            if isinstance(value, ParentedTree):
                value._parent = None
                node[i] = value.copy(deep=True)
            else:
                node[i] = value.clone() 

        replace(child, variable, value)

def norm_negation(node):
    if not isinstance(node, Tree):
        return
    for i, ni in enumerate(node):
        # is it a negation functor?
        if isinstance(ni, ParentedTree) and ni.label() == 'compound' and \
                ni[0].label() == 'functor' and ni[0][0].val in ['\\+','not']: 
                # take first argument
                first = ni[1][0]
                if isinstance(first, ParentedTree):
                    first._parent = None
                # create a new tree
                ni = node[i] = ParentedTree('unop', 
                        [Token('NOT', '\\+', ni[0][0].pos), first])
        norm_negation(ni)      

dict_funct = {
    'member': 'memb',
    'append': 'conc',
    'reverse': 'rev',
    'length': 'len'}
def translate(node):
    if not isinstance(node, Tree):
        return
    if node.label() == 'functor':
        if node[0].val in dict_funct:
            node[0] = node[0].clone(val=dict_funct[node[0].val])
    for child in node:
        translate(child)

def normalize(clause, full=False):
    """ Normalize a clause.
    It performs the following edits:
    1. Removes simple binops, such as A = val, where A occurs only once 
    in the remaining of the code, and replaces that occurence with val.
    New: it also normalizes variables that occur more often.  
    2. Translates functor names specified in dict_funct.
    """
    normalized = clause.copy(deep=True)

    var_counts = collections.defaultdict(int)
    for node in clause.subtrees():
        if isinstance(node, Tree) and node.label() == 'variable':
            name = node[0].val
            var_counts[name] += 1

    # select which vars to seek binop for
    cand_vars = set([k for k, c in var_counts.items()]) # if c == 2])

    while True:
        var, val = get_binop(normalized, cand_vars)
        if not var:
            break
        cand_vars.remove(var[0].val)
        node = var.parent() # binop node
        parent = node.parent() # parent of binop
        # check if var can be normalized
        repl = True
        # get all variables in val
        val_vars = get_vars_between(val, None, None)
        if var[0].val in val_vars: # if var in val, replace not possible
            repl = False
        elif full or not isinstance(val, ParentedTree) or val.label() == 'variable':
            repl = True
        elif len(normalized) == 2: # clauses with head only are irrelevant
            # find first occurence 
            if find_var(normalized[0], var): # if var is in head
                bet = get_vars_between(normalized[1], None, var)
            else:
                # find first occurence of var
                var_pos = find_var(normalized[1], var)
                if var_pos != var:
                    bet = get_vars_between(normalized[1], var_pos, var)
                else:
                    bet = set()
            # if intersection not empty, cannot replace
            if val_vars & bet:
                repl = False
        # if var is to be replaced, remove binop and replace variable
        if repl:
            if parent.label() == 'and':
                # have exactly two children, one to delete, one to keep
                keep = parent[0] if parent[1] == node else parent[1]
                # replace 'and' node with the one to keep
                gparent = parent.parent()
                for i, child in enumerate(gparent):
                    if child == parent:
                        and_index = i
                if isinstance(keep, ParentedTree):
                    keep._parent = None
                gparent[and_index] = keep
            else:
                parent.remove(node)
            # replace variable
            replace(normalized, var, val)

    # translate functor names
    translate(normalized)

    # normalize not, \+, etc.
    norm_negation(normalized)

    return normalized


def split_or(node):
    if not isinstance(node, ParentedTree):
        return []

    label = node.label()
    if label == 'or':
        return [node[0], node[1]]

    for i, child in enumerate(node):
        cs = split_or(child)
        # if child is split into two elements --> replace that child
        # create two nodes, one contains first element, the other contains 
        # the second element
        if len(cs) == 2:
            nc = node.copy(deep=True)
            if isinstance(cs[0], ParentedTree):
                cs[0]._parent = None
            if isinstance(cs[1], ParentedTree):
                cs[1]._parent = None
            node[i] = cs[0]
            nc[i] = cs[1]
            return [node, nc]
    return []

def has_cut(clause):
    return "!" in [l.val for l in clause.leaves() if l.type == 'NAME']

def get_patterns(tree, types):
    """ Types: a set of patterns types to be included in the program. """
    if isinstance(tree, str):
        tree = prolog_parse(tree)
        if tree is None:
            return
    tree = ParentedTree.convert(tree)

    # split clauses with "or" into several clauses 
    # (select only clauses with proper name)
    clauses = [clause for clause in tree]

    while True:
        for ci, c in enumerate(clauses):
            c1 = split_or(c)
            if len(c1) == 2:
                clauses[ci:ci+1] = c1
                break
        else:
            break

    # check whether there is a cut in clause
    cuts, cut = [], False
    for clause in clauses:
        cuts.append(cut)
        if has_cut(clause):
            cut = True

    # duplicate clauses: add original and normalized clause
    #clauses = [(clause, "", cuts[i]) for i, clause in enumerate(clauses)] + \
    #          [(normalize(clause), "norm ", cuts[i]) for i, clause in enumerate(clauses)]
    #clauses = [(normalize(clause, full=False), "", cuts[i]) for i, clause in enumerate(clauses)] + \
    #          [(normalize(clause, full=True), "norm ", cuts[i]) for i, clause in enumerate(clauses)]
    #clauses = [(normalize(clause), "", cuts[i]) for i, clause in enumerate(clauses)]
    
    #clauses = [(clause, "", False) for clause in clauses] 
    clauses = [(normalize(clause, full=False), "", cuts[i]) for i, clause in enumerate(clauses)]

    # get patterns separately for each clause
    for clause, prefix, cut in clauses:
        # collect variable nodes in this clause
        variables = collections.defaultdict(list)
        for node in clause.subtrees():
            if isinstance(node, Tree) and node.label() == 'variable':
                name = node[0].val
                variables[name].append(node)

        if "all" in types or "singleton" in types:
            # yield patterns for singleton variables
            for var, nodes in variables.items():
                if len(nodes) == 1:
                    #yield 'has_singleton', nodes
                    pat = pattern(clause, nodes)
                    if pat:
                        yield prefix+pat, nodes
                        if cut:
                            yield "cut "+prefix+pat, nodes

        if "all" in types or "var_pairs" in types:
            # yield patterns for variable-variable pairs (within a clause)
            for var, nodes in variables.items():
                for selected in combinations(nodes, 2):
                    pat = pattern(clause, selected)
                    if pat:
                        yield prefix + pat, selected
                        if cut:
                            yield "cut " + prefix + pat, selected

        """if "all" in types or "alt_vars" in types:
            # yield patterns for variable-variable + variable-variable
            # pairs/pairs (within a clause)
            combs = []
            for var, nodes in variables.items():
                combs.extend(combinations(nodes, 2))

            for selected in combinations(combs, 2):
                if selected[0][0] == selected[1][0]:
                    continue
                selected = selected[0] + selected[1]
                pat = pattern(clause, selected)
                if pat:
                    yield prefix + pat, selected
                    if cut:
                        yield "cut " + prefix + pat, selected"""
        
        # yield patterns for variable-literal / literal-literal pairs
        # yield patterns for singleton literals
        # (only within a topmost compound / binop / unop)
        def patterns_with_literals(node):
            if not isinstance(node, Tree):
                return
            if node.label() in {'compound', 'binop', 'unop'}:
                vars = [n for n in node.subtrees() if n.label() == 'variable']
                lits = [n for n in node.subtrees() if n.label() == 'literal']
                names = [n for n in node.leaves() if isinstance(n, Token) and n.type == 'NAME' and n.val == 'nil']
                lits = lits + names
                for selected in chain(combinations(lits, 1), 
                                      combinations(lits, 2), 
                                      product(lits, vars)):
                    pat = pattern(clause, selected)
                    if pat:
                        yield prefix+pat, selected
            else:
                for child in node:
                    yield from patterns_with_literals(child)
        if "all" in types or "literal_pairs" in types:
            yield from patterns_with_literals(clause)

        """if "all" in types or "names" in types:
            if len(clause) > 1 and isinstance(clause[1], Tree):
                name_leaves = [l for l in clause[1].leaves() if l.type == 'NAME']
                for comb in combinations(name_leaves, 2):
                    pat = pattern(clause, comb)
                    yield '{} {}'.format(comb[0].val, comb[1].val), comb"""

# Extract edits and other data from existing traces for each problem.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", type=int, help="Problem ID")
    parser.add_argument("name", type=str, help="Problem Name")
    parser.add_argument("data", type=str, help="Data directory")
    parser.add_argument("min", type=int, help="Minimal number of positive/negative testing instances")
    parser.add_argument("types", type=str, nargs="*", help="Types (all, var_pairs)")

    args = parser.parse_args()
    pid, name, data_folder, min_test, types = args.pid, args.name, \
            args.data, args.min, set(args.types)
    submissions = pickle.load(open('pickle/programs-{}.pickle'.format(pid), 'rb'))

    # find test/train users
    users = sorted({user for code, info in submissions.items() for user in info['users']})
    random.Random(0).shuffle(users)
    split = int(len(users)*0.7)
    learn_users = set(users[:split])
    test_users = set(users[split:])

    # find test/train programs
    data = {
        'train': [],
        'test': []
    }
    for code, info in submissions.items():
        if len(code) > 1000 or prolog_parse(code) is None:
            continue
        if name not in code:
            continue
        data['train'] += [(code, info['n_tests'] == info['n_passed'])] * len(info['users'] & learn_users)
        data['test'] += [(code, info['n_tests'] == info['n_passed'])] * len(info['users'] & test_users)

    data['train'].sort()
    data['test'].sort()

    # count passed & not passed in test
    passed = sum(correct for code, correct in data['test'])
    notpassed = sum(not correct for code, correct in data['test'])
    if passed < min_test or notpassed < min_test:
        print("Len of train data: {}, therefore skipping".format(len(data['train'])))
        sys.exit()

    # save test users to file
    makedirs('{}/{}'.format(data_folder, name), exist_ok=True)
    with open('{}/{}/users-test.txt'.format(data_folder, name), 'wt') as f:
        for user in test_users:
            print(user, file=f)

    # print info about test users and test/train programs
    print('Test users:')
    print(test_users)
    print('Count of users used in learning: ', len(learn_users))
    print('Count of users used in testing: ', len(test_users))
    print()
    for which in ['train', 'test']:
        print('Programs ({}):'.format(which))
        print('correct: {} ({} unique)'.format(
            len([code for code, correct in data[which] if correct]),
            len({code for code, correct in data[which] if correct})))
        print('incorrect: {} ({} unique)'.format(
            len([code for code, correct in data[which] if not correct]),
            len({code for code, correct in data[which] if not correct})))
        print()

    debug = False # debug a specific program
    if debug:
        code = 'memberBT(nil, nil). memberBT(nil, A). memberBT(A, b(_, A, _)).'
        print(code)
        for pat, nodes in get_patterns(code, types, name):
            print(pat)
        import sys
        sys.exit(1)

    
    # extract attributes from training data
    patterns = collections.Counter()
    for code, correct in data['train']:
        for pat, nodes in get_patterns(code, types, name):
            patterns[pat] += 1
    attrs = []

    with open('{}/{}/attributes.tab'.format(data_folder, name), 'w') as pattern_file:
        for i, (pat, count) in enumerate(patterns.most_common()):
            if count < 5:
                break
            attrs.append(pat)
            print('a{}\t{}'.format(i, pat), file=pattern_file)

    # check and write attributes for training/test data
    for t in ['train', 'test']:
        with open('{}/{}/programs-{}.tab'.format(data_folder, name, t), 'w') as f:
            # print header
            print('\t'.join(['code', 'correct'] + ['a'+str(i) for i in range(len(attrs))]+['Arguments']), file=f)
            print('\t'.join(['d'] * (len(attrs)+2) + ['string']), file=f)
            print('meta\tclass'+'\t'*len(attrs)+'\tmeta', file=f)

            # print rows (program, correct, attr1, attr2, …)
            for code, correct in data[t]:
                record = '{}\t{}'.format(repr(code), 'T' if correct else 'F')
                code_pats = [pat for pat, nodes in get_patterns(code, types, name)]
                for pat in attrs:
                    record += '\t{}'.format('T' if pat in code_pats else 'F')
                print(record, file=f)

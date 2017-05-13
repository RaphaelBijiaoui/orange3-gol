# CodeQ: an online programming tutor.
# Copyright (C) 2015 UL FRI
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

from collections import namedtuple
from collections.abc import Iterable
import string

from nltk import Tree

# Stores a token's type and value, and optionally the position of the first
# character in the lexed stream.
class Token(namedtuple('Token', ['type', 'val', 'pos'])):
    __slots__ = ()

    def __repr__(self):
        return str(self)

    # Custom constructor to support default parameters.
    def __new__(cls, type, val='', pos=None):
        return super(Token, cls).__new__(cls, type, val, pos)

    def __str__(self):
        return str(self.val)

    # Only consider type and value when comparing tokens. There is probably a
    # cleaner way of doing this.
    __eq__ = lambda x, y: x[0] == y[0] and x[1] == y[1]
    __ne__ = lambda x, y: x[0] != y[0] or x[1] != y[1]
    __lt__ = lambda x, y: tuple.__lt__(x[0:2], y[0:2])
    __le__ = lambda x, y: tuple.__le__(x[0:2], y[0:2])
    __ge__ = lambda x, y: tuple.__ge__(x[0:2], y[0:2])
    __gt__ = lambda x, y: tuple.__gt__(x[0:2], y[0:2])

    # Only hash token's value (we don't care about position, and types are
    # determined by values).
    def __hash__(self):
        return hash(self[1])

    # Return a copy of this token, possibly modifying some fields.
    def clone(self, type=None, val=None, pos=None):
        return Token(self.type if type is None else type,
                     self.val if val is None else val,
                     self.pos if pos is None else pos)

from .lexer import lexer, operators
from .parser import parser

def parse(code):
    try:
        return parser.parse(code)
    except SyntaxError:
        return None

# Return a list of tokens in [text].
def tokenize(text):
    lexer.input(text)
    return [Token(t.type, t.value, t.lexpos) for t in lexer]

# Return a one-line string representation of [obj] which may be a Tree or a
# list of tokens.
def stringify(obj):
    if isinstance(obj, Token):
        if obj.type in ('PERIOD', 'COMMA'):
            return str(obj) + ' '
        if obj.type in operators.values():
            return ' ' + str(obj) + ' '
        return str(obj)
    if isinstance(obj, Iterable):
        if isinstance(obj, Tree) and obj.label() == 'clause':
            return ''.join([stringify(child) for child in obj]) + '\n'
        return ''.join([stringify(child) for child in obj])

# Return a canonical name for the [n]th variable in scope.
def canonical_varname(n):
    names = string.ascii_uppercase
    if n < len(names):
        return names[n]
    return 'X{}'.format(n)

# Rename variables in [tokens] to A0, A1, A2,… in order of appearance.
def rename_vars_list(tokens, names=None):
    if names is None:
        names = {}
    next_id = len(names)

    # Return a new list.
    tokens = list(tokens)
    for i, t in enumerate(tokens):
        if t.type == 'VARIABLE' and t.val != '_':
            cur_name = t.val
            if cur_name not in names:
                names[cur_name] = canonical_varname(next_id)
                next_id += 1
            tokens[i] = t.clone(val=names[cur_name])
    return tokens

# Rename variables in AST rooted at [root] to A0, A1, A2,… in order of
# appearance.
def rename_vars_ast(root, fixed_names=None):
    if fixed_names is None:
        fixed_names = {}
    names = {}
    next_id = len(fixed_names) + len(names)

    def rename_aux(node):
        nonlocal fixed_names, names, next_id
        if isinstance(node, Tree):
            if node.label() == 'clause':
                names = {}
                next_id = len(fixed_names) + len(names)
            new_children = [rename_aux(child) for child in node]
            new_node = Tree(node.label(), new_children)
        elif isinstance(node, Token):
            if node.type == 'VARIABLE':
                token = node
                if token.val.startswith('_'):
                    new_node = token.clone(val=canonical_varname(next_id))
                    next_id += 1
                else:
                    cur_name = token.val
                    if cur_name in fixed_names:
                        new_name = fixed_names[cur_name]
                    else:
                        if cur_name not in names:
                            names[cur_name] = canonical_varname(next_id)
                            next_id += 1
                        new_name = names[cur_name]
                    new_node = token.clone(val=new_name)
            else:
                new_node = node
        return new_node
    return rename_aux(root)

# Yield "interesting" parts of a Prolog AST as lists of tokens.
def interesting_ranges(ast, path=()):
    if ast.label() in {'clause', 'head', 'or', 'if', 'and'}:
        if ast.label() == 'clause':
            # Special case for clause with one goal.
            if len(ast) == 4 and ast[2].label() == 'term':
                terminals = ast[2].leaves()
                yield terminals, path + (ast.label(), 'and')

        if ast.label() == 'and':
            for i in range(0, len(ast), 2):
                for j in range(i, len(ast), 2):
                    subs = ast[i:j+1]
                    terminals = []
                    for s in subs:
                        terminals.extend([s] if isinstance(s, Token) else s.leaves())
                    # We want at least some context.
                    if len(terminals) > 1:
                        yield terminals, path + (ast.label(),)
        else:
            terminals = ast.leaves()
            # We want at least some context.
            if len(terminals) > 1:
                yield terminals, path + (ast.label(),)

    for subtree in ast:
        if isinstance(subtree, Tree):
            yield from interesting_ranges(subtree, path + (ast.label(),))

# Map "formal" variable names in the edit a→b to actual names in code [tokens].
# The set [variables] contains all variable names in the current scope. These
# are used in cases such as [A]→[A,B], where the edit introduces new variables.
# Return a new version of b with actual variable names.
def map_vars(a, b, tokens, variables):
    mapping = {}
    new_index = 0
    for i in range(len(a)):
        if tokens[i].type == 'VARIABLE':
            formal_name = a[i].val
            if tokens[i].val != '_':
                actual_name = tokens[i].val
            else:
                actual_name = 'New'+str(new_index)
                new_index += 1
            mapping[formal_name] = actual_name

    remaining_formal = [t.val for t in b if t.type == 'VARIABLE' and t.val not in mapping.keys()]
    remaining_actual = [var for var in variables if var not in mapping.values()]

    while len(remaining_actual) < len(remaining_formal):
        remaining_actual.append('New'+str(new_index))
        new_index += 1

    for i, formal_name in enumerate(remaining_formal):
        mapping[formal_name] = remaining_actual[i]

    return [t if t.type != 'VARIABLE' else t.clone(val=mapping[t.val]) for t in b]

# Return a set of predicate names (e.g. conc/3) used in [program].
def used_predicates(program):
    predicates = set()
    def walk(tree, dcg=False):
        if isinstance(tree, Tree):
            # DCG predicates can be called without parameters
            if tree.label() == 'clause' and len(tree) == 4 and \
                    tree[1].type == 'FROMDCG':
                dcg = True
            if tree.label() == 'term' and len(tree) >= 3 and \
                    isinstance(tree[0], Tree) and tree[0].label() == 'functor':
                if len(tree) == 3:
                    predicates.add('{}/0'.format(tree[0][0]))
                else:
                    predicates.add('{}/{}'.format(tree[0][0], (len(tree[2])+1)//2))
            for subtree in tree:
                walk(subtree, dcg)
        elif isinstance(tree, Token):
            if dcg and tree.type == 'NAME':
                predicates.add('{}/{}'.format(tree.val, 2))
                predicates.add('{}/{}'.format(tree.val, 3))
                predicates.add('{}/{}'.format(tree.val, 4))
    tree = parse(program)
    if tree is not None:
        walk(tree)
    return predicates

# Basic sanity check.
if __name__ == '__main__':
    var_names = {}
    before = rename_vars(tokenize("dup([A0|A1], [A2|A3])"), var_names)
    after = rename_vars(tokenize("dup([A0|A1], [A5, A4|A3])"), var_names)

    line = lines[0]
    variables = [t.val for t in tokenize(code) if t.type == 'VARIABLE']
    mapped = map_vars(before, after, line, variables)
    print(mapped)

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

from nltk import Tree
import ply.yacc as yacc
from .lexer import tokens
from .util import Token

# PARSER
precedence = (
    ('nonassoc', 'FROM', 'FROMDCG'),
    ('right', 'PIPE'),
    ('right', 'IMPLIES'),
    ('right', 'NOT'),
    ('nonassoc', 'EQU', 'NEQU', 'EQ', 'NEQ', 'UNIV', 'IS', 'EQA', 'NEQA', 'LT', 'LE', 'GT', 'GE', 'LTL', 'LEL', 'GTL', 'GEL', 'IN', 'INS', 'THROUGH', 'EQFD', 'NEQFD', 'LTFD', 'LEFD', 'GTFD', 'GEFD'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'STAR', 'DIV', 'IDIV', 'MOD'),
    ('nonassoc', 'POW'),
    ('right', 'UMINUS', 'UPLUS'),
    ('nonassoc', 'UINTEGER', 'UREAL'),
    ('nonassoc', 'NAME', 'VARIABLE', 'STRING'),
    ('nonassoc', 'PERIOD'),
    ('nonassoc', 'LBRACKET', 'RBRACKET', 'LPAREN', 'RPAREN', 'COMMA', 'SEMI', 'LBRACE', 'RBRACE')
)

def make_token(p, n):
    lextoken = p.slice[n]
    return Token(lextoken.type, lextoken.value, lextoken.lexpos)

def p_text_empty(p):
    'text : '
    p[0] = Tree('text', [])
def p_text_clause(p):
    'text : text clause'
    p[0] = p[1]
    p[0].append(p[2])

def p_clause_head(p):
    'clause : head PERIOD'
    p[0] = Tree('clause', [p[1]])
def p_clause_rule(p):
    '''clause : head FROM or PERIOD
              | head FROMDCG or PERIOD'''
    p[0] = Tree('clause', [p[1], p[3]])

def p_head(p):
    'head : term'
    p[0] = Tree('head', [p[1]])

def p_or_single(p):
    'or : if'
    p[0] = p[1]
def p_or_if(p):
    'or : or SEMI if'
    p[0] = Tree('or', [p[1], p[3]])

def p_if_single(p):
    'if : and'
    p[0] = p[1]
def p_if_and(p):
    'if : and IMPLIES if'
    p[0] = Tree('if', [p[1], p[3]])

def p_and_single(p):
    'and : term'
    p[0] = p[1]
def p_and_term(p):
    'and : term COMMA and'
    p[0] = Tree('and', [p[1], p[3]])

# Special case for zero-arity predicates supported by SWI-Prolog.
def p_term_functor_zero(p):
    'term : functor LPAREN RPAREN'
    # No whitespace allowed between functor and LPAREN.
    t2 = make_token(p, 2)
    if p[1][0].pos + len(p[1][0].val) < t2.pos:
        raise SyntaxError('whitespace before ' + str(t2))
    p[0] = Tree('compound', [p[1]])
def p_term_functor(p):
    'term : functor LPAREN args RPAREN'
    # No whitespace allowed between functor and LPAREN.
    t2 = make_token(p, 2)
    if p[1][0].pos + len(p[1][0].val) < t2.pos:
        raise SyntaxError('whitespace before ' + str(t2))
    p[0] = Tree('compound', [p[1], p[3]])

def p_term_or(p):
    'term : LPAREN or RPAREN'
    p[0] = p[2]
def p_term_binary(p):
    '''term : term PLUS term
            | term MINUS term
            | term STAR term
            | term POW term
            | term DIV term
            | term IDIV term
            | term MOD term

            | term EQU term
            | term NEQU term
            | term EQ term
            | term NEQ term
            | term UNIV term
            | term IS term

            | term EQA term
            | term NEQA term
            | term LT term
            | term LE term
            | term GT term
            | term GE term

            | term LTL term
            | term LEL term
            | term GTL term
            | term GEL term

            | term PIPE term
            | term THROUGH term
            | term IN term
            | term INS term
            | term EQFD term
            | term NEQFD term
            | term LTFD term
            | term LEFD term
            | term GTFD term
            | term GEFD term'''
    p[0] = Tree('binop', [p[1], make_token(p, 2), p[3]])
def p_term_unary(p):
    '''term : NOT term
            | MINUS term %prec UMINUS
            | PLUS term %prec UPLUS'''
    p[0] = Tree('unop', [make_token(p, 1), p[2]])
def p_term_list(p):
    'term : list'
    p[0] = p[1]

def p_term_variable(p):
    'term : VARIABLE'
    p[0] = Tree('variable', [make_token(p, 1)])
def p_term_simple(p):
    '''term : STRING
            | UINTEGER
            | UREAL'''
    p[0] = Tree('literal', [make_token(p, 1)])
def p_term_name(p):
    'term : NAME'
    p[0] = make_token(p, 1)

def p_term_clpr(p):
    'term : LBRACE clpr RBRACE'
    p[0] = Tree('term', [make_token(p, 1), p[2], make_token(p, 3)])

# compound term arguments
def p_args_single(p):
    'args : term'
    p[0] = Tree('args', [p[1]])
def p_args_multiple(p):
    'args : term COMMA args'
    p[0] = Tree('args', [p[1], p[3]])

# list elements
def p_elems_single(p):
    'elems : term'
    if isinstance(p[1], Tree) and p[1].label() == 'binop' and p[1][1].type == 'PIPE':
        p[0] = Tree('list', [p[1][0]])
        #if p[1][2] != Tree('literal', [Token(type='NIL', val='[]')]):
        p[0].append(p[1][2])
    else:
        p[0] = Tree('list', [p[1], Tree('literal', [Token(type='NIL', val='[]')])])
def p_elems_multiple(p):
    'elems : term COMMA elems'
    p[0] = Tree('list', [p[1], p[3]])

def p_list_empty(p):
    'list : LBRACKET RBRACKET'
    p[0] = Tree('literal', [Token(type='NIL', val='[]')])
def p_list(p):
    'list : LBRACKET elems RBRACKET'
    p[0] = p[2]

def p_functor(p):
    'functor : NAME'
    p[0] = Tree('functor', [make_token(p, 1)])

# CLP(R) syntax
def p_clpr_single(p):
    'clpr : clpr_constr'
    p[0] = Tree('clpr', [p[1]])
def p_clpr_more(p):
    '''clpr : clpr_constr COMMA clpr
            | clpr_constr SEMI clpr'''
    p[0] = Tree('clpr', [p[1], make_token(p, 2), p[3]])
# XXX temporary until the new parser is in place, this also covers { } notation for DCGs
def p_clpr_constr(p):
    'clpr_constr : term'
    p[0] = p[1]

def p_error(t):
    #if t is None:
    #    raise SyntaxError('unexpected end of file')
    #print(t.type)
    if t is None:
        return
    #e = yacc.YaccSymbol()
    #e.type = 'error'
    #if t is not None:
    #    e.value = t.value
    #else:
    #    e.value = None
    #return e
    #while True:
    #    tok = parser.token()
    #    break
        #if not tok:
        #    break
        #print(tok)
    #parser.errok()
    #return parser.token()
    #else:
    #    raise SyntaxError('{}: unexpected {}'.format(t.lexpos, t.value))

parser = yacc.yacc(debug=False)

if __name__ == '__main__':
    from .util import stringify
    while True:
        try:
            s = input('> ')
        except EOFError:
            break
        if not s:
            continue
        ast = parser.parse(s)
        def pp(node):
            if isinstance(node, Tree):
                return '(' + node.label() + ' ' + ' '.join([pp(child) for child in node]) + ')'
            return '"' + str(node) + '"'
        print(pp(ast))
        #ast.pretty_print()
        #print(stringify(ast))

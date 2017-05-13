#!/usr/bin/python3

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

import ply.lex as lex

# LEXER

#states = (
#    ('comment', 'exclusive'),
#)

# tokens; treat operators as names if followed by (
operators = {
    r':-':  'FROM',
    r'-->':  'FROMDCG',
    r'->':  'IMPLIES',
    r'\+':  'NOT',
    r'=':   'EQU',
    r'\=':  'NEQU',
    r'==':  'EQ',
    r'\==': 'NEQ',
    r'=..': 'UNIV',
    r'is':  'IS',
    r'=:=': 'EQA',
    r'=\=': 'NEQA',
    r'<':   'LT',
    r'=<':  'LE',
    r'>':   'GT',
    r'>=':  'GE',
    r'@<':  'LTL',
    r'@=<': 'LEL',
    r'@>':  'GTL',
    r'@>=': 'GEL',
    r'#=':  'EQFD',
    r'#\=': 'NEQFD',
    r'#<':  'LTFD',
    r'#=<': 'LEFD',
    r'#>':  'GTFD',
    r'#>=': 'GEFD',
    r'in':  'IN',
    r'ins': 'INS',
    r'..':  'THROUGH',
    r'+':   'PLUS',
    r'-':   'MINUS',
    r'*':   'STAR',
    r'/':   'DIV',
    r'//':  'IDIV',
    r'mod': 'MOD',
    r'**':  'POW',
    r'^':   'POW',
    r'.':   'PERIOD',
    r',':   'COMMA',
    r';':   'SEMI'
}
tokens = sorted(list(operators.values())) + [
    'UINTEGER', 'UREAL',
    'NAME', 'VARIABLE', 'STRING',
    'LBRACKET', 'RBRACKET', 'LPAREN', 'RPAREN', 'PIPE', 'LBRACE', 'RBRACE',
    'INVALID'
]

# punctuation
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_PIPE = r'\|'
t_LBRACE = r'{'
t_RBRACE = r'}'

t_UINTEGER = r'[0-9]+'
t_UREAL    = r'[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?|inf|nan'
t_VARIABLE = r'(_|[A-Z])[a-zA-Z0-9_]*'
t_STRING   = r'"(""|\\.|[^\"])*"'

# no support for nested comments yet
def t_comment(t):
    r'(/\*(.|\n)*?\*/)|(%.*)'
    pass

def t_NAME(t):
    r"'(''|\\.|[^\\'])*'|[a-z][a-zA-Z0-9_]*|[-+*/\\^<>=~:.?@#$&]+|!|;|,"
    if t.value == ',' or \
       t.lexer.lexpos >= len(t.lexer.lexdata) or t.lexer.lexdata[t.lexer.lexpos] != '(':
        t.type = operators.get(t.value, 'NAME')
    return t

t_ignore  = ' \t'

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    # TODO send this to stderr
    #print("Illegal character '" + t.value[0] + "'")
    t.type = 'INVALID'
    t.value = t.value[0]
    t.lexer.skip(1)
    return t

lexer = lex.lex(errorlog=lex.NullLogger())

if __name__ == '__main__':
    while True:
        try:
            s = input('> ')
        except EOFError:
            break
        if not s:
            continue

        lexer.input(s)
        tokens = list(lexer)
        print(tokens)

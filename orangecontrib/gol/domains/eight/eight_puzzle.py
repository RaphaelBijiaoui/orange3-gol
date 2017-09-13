""" State representation for eight puzzle in goal-oriented programming. """
from Orange.data import DiscreteVariable

class EightPuzzleDomain:
    def __init__(self):
        self.attributes = [DiscreteVariable.make("{}_{}".format(i, v),
                                                 values=["no", "yes"])
                           for i in range(9) for v in range(9)]

    def get_attributes(self):
        return self.attributes

    def get_attribute(self, at, state):
        if at.name == "solved":
            return state == ['1', '2', '3', '4', '5', '6', '7', '8', '0']
        tile, pos = at.name.split('_')
        if state[int(pos)] == tile:
            return True
        return False

neighbours = {
    '0': ['1', '3'],
    '1': ['0', '2', '4'],
    '2': ['1', '5'],
    '3': ['0', '4', '6'],
    '4': ['1', '3', '5', '7'],
    '5': ['2', '4', '8'],
    '6': ['3', '7'],
    '7': ['4', '6', '8'],
    '8': ['5', '7']}

class EightPuzzleState:
    def __init__(self, state, domain, ind0=None):
        """
        :param state: state is a list of tiles at specific positions ...
         ['1', '3', '5', ...] means that '1' is at top left, '3' is top, '5' is
         top right, etc.
        :param domain: an EightPuzzleDomain instance
        """
        self.state = state
        if ind0 is None:
            self.ind0 = self.state.index('0')
        else:
            self.ind0 = ind0
        self.domain = domain

    def get_id(self):
        return "".join(self.state)

    def get_moves(self):
        """ Method that generates possible moves.
        Basically it is moving the empty square.
        """
        moves = []
        for indx in neighbours[self.ind0]:
            new_state = self.state[:]
            new_state[self.ind0] = self.state[indx]
            new_state[indx] = self.state[self.ind0]
            moves.append(EightPuzzleState(new_state, self.domain, ind0=indx))
        return moves

    def get_attribute(self, at):
        return self.domain.get_attribute(at, self.state)

    def solved(self):
        return self.state == "123456780"

    def __hash__(self):
        return self.state

    def __eq__(self, other):
        return self.state == other.state

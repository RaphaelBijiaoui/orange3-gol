""" Classes related to rule representation used in goal-oriented learning. """

import numpy as np
from scipy.stats import norm
from Orange.classification.rules import Rule, Validator, Evaluator

class RRule(Rule):
    """ Basic class describing a single rule. """
    def __init__(self, predicted_goals, goal_selector, **kwargs):
        Rule.__init__(self, **kwargs)
        self.predicted_goals = predicted_goals
        self.goal_selector = goal_selector
        self.ncovered = 0
        self.mean, self.sum, self.std, self.max = 0, 0, 0, 0
        self.min_values = None

    def filter_and_store(self, X, Y, W, target_class, predef_covered=None):
        """ Apply data and target class to a rule. """
        self.target_class = target_class
        if predef_covered is not None:
            self.covered_examples = predef_covered
        else:
            self.covered_examples = np.ones(X.shape[0], dtype=bool)
        for selector in self.selectors:
            self.covered_examples &= selector.filter_data(X)
        self.ncovered = self.covered_examples.sum()
        if self.ncovered:
            # select predicted goals
            if not self.selectors:
                self.min_values = self.goal_selector.get_min_values(self.predicted_goals)
            else:
                self.predicted_goals, self.min_values = \
                    self.goal_selector(self.predicted_goals, self.covered_examples)
            self.sum = self.min_values[self.covered_examples].sum()
            self.mean = np.mean(self.min_values[self.covered_examples])
            self.std = np.std(self.min_values[self.covered_examples])
            self.max = np.max(self.min_values[self.covered_examples])

    def __str__(self):
        attributes = self.domain.attributes
        if self.selectors:
            cond = " AND ".join([attributes[s.column].name + s.op +
                                 (str(attributes[s.column].values[int(s.value)])
                                  if attributes[s.column].is_discrete
                                  else str(s.value)) for s in self.selectors])
        else:
            cond = "TRUE"
        return cond

    def copy(self):
        return RRule(self.predicted_goals[:], self.goal_selector,
                     selectors=self.selectors[:], domain=self.domain,
                     significance_validator=self.significance_validator,
                     quality_evaluator=self.quality_evaluator,
                     complexity_evaluator=self.complexity_evaluator,
                     general_validator=self.general_validator)

    def __hash__(self):
        return hash(self.covered_examples.tostring())

    def __eq__(self, other):
        return (self.covered_examples == other.covered_examples).all()



class MeanEvaluator(Evaluator):
    """ Returns average complexity of covered examples. It adds k * max_value
    to guide towards rules that cover larger number of examples. """
    def __init__(self, k):
        self.k = k

    def evaluate_rule(self, rule):
        quality = rule.sum + self.k * rule.max
        quality /= rule.ncovered
        return quality

class GuardianValidator(Validator):
    """
    Discard rules that
    - cover less than the minimum required number of examples,
    - are too complex.
    """
    def __init__(self, max_rule_length=5, min_covered_examples=1):
        self.max_rule_length = max_rule_length
        self.min_covered_examples = min_covered_examples

    def validate_rule(self, rule):
        return (rule.ncovered >= self.min_covered_examples and
                rule.length <= self.max_rule_length)

class TTestValidator(Validator):
    """
    Discard rules that
    - cover less than the minimum required number of examples,
    - are too complex.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.t_threshold = norm.ppf(1-self.alpha)

    def validate_rule(self, rule):
        if self.alpha >= 1.0:
            return True
        if rule.parent_rule is None:
            return True
        t = rule.parent_rule.mean - rule.mean
        s = np.sqrt(rule.std**2 / rule.ncovered +
                    rule.parent_rule.std**2 / rule.parent_rule.ncovered)
        t /= s
        return t >= self.t_threshold

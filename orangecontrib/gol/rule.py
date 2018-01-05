""" Classes related to rule representation used in goal-oriented learning. """

import numpy as np
from scipy.stats import norm
from Orange.classification.rules import Rule, Validator, Evaluator

class RRule(Rule):
    """ Basic class describing a single rule. """
    def __init__(self, predicted_goals, complexities, **kwargs):
        Rule.__init__(self, **kwargs)
        self.predicted_goals = predicted_goals
        self.complexities = complexities
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
        self.general_avg = np.mean(self.complexities[0])
        if self.ncovered:
            self.min_values = self.complexities[self.predicted_goals[0]] 
            for ni in self.predicted_goals[1:]:
                self.min_values = np.minimum(self.min_values, self.complexities[ni])
            self.min_covered = self.min_values[self.covered_examples]
            self.goal_indices = []
            for ni in self.predicted_goals:
                self.goal_indices.append(self.complexities[ni][self.covered_examples] == self.min_covered)
            self.sum = self.min_covered.sum()
            self.mean = np.mean(self.min_covered)
            self.std = np.std(self.min_covered)
            self.max = np.max(self.min_covered)

    def __str__(self):
        attributes = self.domain.attributes
        if self.selectors:
            cond = " AND ".join([attributes[s.column].name + s.op +
                                 (str(attributes[s.column].values[int(s.value)])
                                  if attributes[s.column].is_discrete
                                  else str(s.value)) for s in self.selectors])
        else:
            cond = "TRUE"
        cond = "IF " + cond + " THEN " + str(self.predicted_goals)
        return cond

    def copy(self):
        return RRule(self.predicted_goals[:], self.complexities,
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
    """ Returns weighted average over all predicted classes. 
    It adds k * average (over all examples and all classes)
    to guide towards rules that cover larger number of examples. """
    def __init__(self, k):
        self.k = k

    def evaluate_rule(self, rule):
        quality = 0
        for ni in rule.predicted_goals:
            nbest = rule.min_covered[rule.goal_indices[ni]]
            nn = rule.goal_indices[ni].sum()
            nquality = nbest.sum() + self.k * rule.general_avg
            nquality /= nn + self.k
            quality += nn/rule.ncovered*nquality
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
    A rule validator that:
    a) tests whether rule covers enough examples
    b) tests whether a rule has too many conditions
    c) tests for maximal complexity value
    d) tests full rule and parent rule significances
    """
    def __init__(self, alpha=1.0, parent_alpha=1.0,
                 max_rule_length=5, min_covered_examples=1,
                 max_complexity = -1):
        self.alpha = alpha
        self.parent_alpha = parent_alpha
        self.parent_threshold = norm.ppf(1-self.parent_alpha)
        self.threshold = norm.ppf(1-self.alpha)

        self.max_rule_length = max_rule_length
        self.min_covered_examples = min_covered_examples
        self.max_complexity = max_complexity

    def validate_rule(self, rule):
        if rule.ncovered < self.min_covered_examples:
            return False
        if rule.length > self.max_rule_length:
            return False
        if rule.max > self.max_complexity > -1:
            return False
        # check overall rule significance
        if self.alpha < 1.0:
            t = rule.mean
            t /= (rule.std/np.sqrt(rule.ncovered))
            if t < self.threshold:
                return False
        if self.parent_alpha < 1.0:
            t = rule.parent_rule.mean - rule.mean
            s = np.sqrt(rule.std**2 / rule.ncovered +
                        rule.parent_rule.std**2 / rule.parent_rule.ncovered)
            t /= s
            if t < self.parent_threshold:
                return False
        return True

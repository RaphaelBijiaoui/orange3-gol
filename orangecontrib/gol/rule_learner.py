""" A module for learning a single rule in domains with continuous class. """
from copy import copy
import numpy as np
from Orange.classification.rules import Selector, LengthEvaluator
from orangecontrib.gol.rule import RRule, MeanEvaluator, GuardianValidator, TTestValidator
EPS = 1e-3

class RegressiveRuleLearner:
    """ A rule learner used in goal-oriented learning. It learns a single
     regression rule with a modified beam search strategy. """
    def __init__(self, min_covered_examples=1, C=10, alpha=0.05,
                 parent_alpha=1.0, width=5, max_rule_length=5,
                 max_complexity=0):
        self.width = width

        # memoization attributes (defined later, here used only for reference)
        self.visited = None
        self.storage = None

        self.significance_validator = TTestValidator(alpha=alpha,
                                                     max_complexity=max_complexity)
        self.condition_significance_validator = TTestValidator(
            parent_alpha=parent_alpha,
            max_rule_length=max_rule_length,
            min_covered_examples=min_covered_examples,

        )
        self.quality_evaluator = MeanEvaluator(C)
        self.complexity_evaluator = LengthEvaluator()
        #self.general_validator = GuardianValidator(max_rule_length=max_rule_length,
        #                                           min_covered_examples=min_covered_examples)

    def fit_storage(self, data, complexities):
        """
        :param data: learing instances (Orange format)
        :param complexities: for each possible goal, complexities for each example
        :return: a set of "best" rules
        """
        X, Y, W = data.X, data.Y, data.W if data.W else None
        # initialize empty rules (one rule for each goal)
        star = []
        for ci, c in enumerate(complexities):
            initial_rule = RRule([ci], complexities,
                                 selectors=[], domain=data.domain,
                                 significance_validator=self.significance_validator,
                                 quality_evaluator=self.quality_evaluator,
                                 complexity_evaluator=self.complexity_evaluator,
                                 general_validator=self.condition_significance_validator)
            initial_rule.filter_and_store(X, Y, W, None)
            initial_rule.do_evaluate()
            star.append(initial_rule)
        best_rule = max(star, key=lambda r: r.quality)
        print(best_rule, best_rule.quality, best_rule.ncovered, best_rule.general_avg)

        # use visited to prevent learning the same rule all over again
        self.visited = set(r for r in star)
        self.storage = {} # a dictionary for memoizing stuff

        # iterate until star contains candidate rules
        while star:
            # specialize each rule in star
            new_star = []
            for r in star:
                rules = self.refine_rule(X, Y, W, r)
                # work refined rules
                for nr in rules:
                    if (nr not in self.visited and nr.quality < nr.parent_rule.quality and
                            nr.is_valid()):
                        if nr.is_significant() and nr.quality < best_rule.quality:
                            best_rule = nr
                        new_star.append(nr)
                    self.visited.add(nr)
            new_star.sort(key=lambda r: r.quality)
            star = new_star[:self.width]
        return best_rule


    def refine_rule(self, X, Y, W, candidate_rule):
        """ Refines rule with new conditions. Returns several refinements. """
        (target_class, candidate_rule_covered_examples,
         candidate_rule_selectors, domain, initial_class_dist,
         prior_class_dist, quality_evaluator, complexity_evaluator,
         significance_validator, general_validator) = candidate_rule.seed()

        # optimisation: to develop further rules is futile
        if candidate_rule.length == general_validator.max_rule_length:
            return []

        possible_selectors = self.find_new_selectors(
            X[candidate_rule_covered_examples],
            Y[candidate_rule_covered_examples],
            W[candidate_rule_covered_examples]
            if W is not None else None,
            domain, candidate_rule_selectors)

        new_rules = []
        for curr_selector in possible_selectors:
            copied_selectors = copy(candidate_rule_selectors)
            copied_selectors.append(curr_selector)
            new_rule = RRule(candidate_rule.predicted_goals[:],
                             candidate_rule.goal_selector,
                             selectors=copied_selectors,
                             parent_rule=candidate_rule,
                             domain=domain,
                             quality_evaluator=quality_evaluator,
                             complexity_evaluator=complexity_evaluator,
                             significance_validator=significance_validator,
                             general_validator=general_validator)

            if curr_selector not in self.storage:
                self.storage[curr_selector] = curr_selector.filter_data(X)
            # optimisation: faster calc. of covered examples
            pdc = candidate_rule_covered_examples & self.storage[curr_selector]
            # to ensure that the covered_examples matrices are of
            # the same size throughout the rule_finder iteration
            new_rule.filter_and_store(X, Y, W, target_class, predef_covered=pdc)
            if new_rule.is_valid():
                new_rule.do_evaluate()
                new_rules.append(new_rule)
        return new_rules

    @staticmethod
    def find_new_selectors(X, Y, W, domain, existing_selectors):
        """ Returns potential selectors for a rule. """
        existing_selectors = (existing_selectors if existing_selectors is not
                              None else [])

        possible_selectors = []
        # examine covered examples, for each variable
        for i, attribute in enumerate(domain.attributes):
            # if discrete variable
            if attribute.is_discrete:
                # for each unique value, generate all possible selectors
                for val in np.unique(X[:, i]):
                    s1 = Selector(column=i, op="==", value=val)
                    s2 = Selector(column=i, op="!=", value=val)
                    possible_selectors.extend([s1, s2])
            # if continuous variable
            elif attribute.is_continuous:
                # choose best thresholds if constrain_continuous is True
                xvals = X[:, i]
                min_ = np.min(xvals)
                max_ = np.max(xvals)
                if np.isnan(min_) or min_ == max_:
                    continue
                values = set([0.0, np.mean(xvals), np.median(xvals)])
                step = (max_ - min_) / 5
                lg10 = np.log10(step)
                dec = 0 if lg10 > 0 else int(np.ceil(-lg10))
                rounded = set([np.round(min_ + i * step, dec) for i in range(1, 6)])
                values |= rounded
                # for each unique value, generate all possible selectors
                for val in values:
                    if min_ <= val <= max_:
                        s1 = Selector(column=i, op="<=", value=val)
                        s2 = Selector(column=i, op=">=", value=val)
                        possible_selectors.extend([s1, s2])

        # remove redundant selectors
        possible_selectors = [smh for smh in possible_selectors if
                              smh not in existing_selectors]

        return possible_selectors

""" A module for learning relationships (rules), where class
 is continuous and attributes can be of arbitrary types.
"""
from copy import copy
from collections import Counter
import numpy as np
from Orange.classification.rules import Selector, LengthEvaluator, Evaluator, Validator
from orangecontrib.gol.rule import RRule
EPS = 1e-3

class RegressiveRuleLearner:
    def __init__(self, min_cover=10, K=10):
        self.learner = BasicLearner(width=10, min_cover=min_cover, K=K)

    def __call__(self, data):
        X, Y, W = data.X, data.Y, data.W if data.W else None
        rules = self.learner.fit_storage(data)
        return rules[0]
        
        selected_rules = []
        covered_relevant = np.zeros(X.shape[0], dtype=bool)
        for r in rules:
            new_covered = ~covered_relevant & r.covered_examples & relevant
            if new_covered.any():
                selected_rules.append(r)
                covered_relevant |= new_covered

        return selected_rules

class BasicLearner:
    """
    A learner for regression rules that employes the beam search strategy.
    Rules are learned from a set of learning instances and an initial rule.
    """
    def __init__(self, width=30, min_cover=30, K=10):
        """
        :param width: beam width
        :param min_cover: minimal coverage of relevant examples
        :return:
        """
        self.width = width
        self.min_cover = min_cover
        self.K = K

        # memoization attributes (defined later, here used only for reference)
        self.visited = None
        self.storage = None
        self.bestr = None
        self.bestq = None

    def fit_storage(self, data):
        """
        :param data: learing instances (Orange format)
        :param relevant: a numpy array specifying which examples have to be covered by rules
        :return: a set of "best" rules
        """
        X, Y, W = data.X, data.Y, data.W if data.W else None
        # initialize empty rule
        initial_rule = RRule(np.mean(Y), selectors=[], domain=data.domain,
                significance_validator = None,
                quality_evaluator = MeanEvaluator(self.K),
                complexity_evaluator = LengthEvaluator(),
                general_validator = GuardianValidator())
        initial_rule.filter_and_store(X, Y, W, None)
        initial_rule.do_evaluate()
        initial_rule.is_valid()

        # initialize star
        star = [initial_rule]
        self.storage = {} # a dictionary for memoizing stuff

        # use visited to prevent learning the same rule all over again
        self.visited = set(r.covered_examples.tostring() for r in star)

        # update best rule
        self.bestr = np.empty(X.shape[0], dtype=object)
        self.bestq = np.full(X.shape[0], np.max(Y), dtype=float)
        for r in star:
            self.update_best(r)

        # iterate until star contains candidate rules
        while star:
            # specialize each rule in star
            new_star = []
            for r in star:
                rules = self.refine_rule(X, Y, W, r)
                # work refined rules
                for nr in rules:
                    rkey = nr.covered_examples.tostring()
                    if rkey not in self.visited and nr.quality < nr.parent_rule.quality:
                        # rule is consistent with basic conditions
                        # can it be new best?
                        if nr.covered_examples.sum() >= self.min_cover:
                            self.update_best(nr)
                            new_star.append(nr)
                    self.visited.add(rkey)

            # assign a rank to each rule in new star
            nrules = len(new_star)
            inst_quality = np.full((X.shape[0], nrules), initial_rule.quality)
            for ri, r in enumerate(new_star):
                inst_quality[r.covered_examples, ri] = r.quality
            sel_rules = min(nrules, 5)
            queues = np.argsort(inst_quality)[:, :sel_rules]

            # create new star from queues
            new_star_set = set()
            index = 0
            while len(new_star_set) < self.width:
                if index >= sel_rules:
                    break
                # pop one rule from each queue and put into a temporary counter
                cnt = Counter()
                for qi, q in enumerate(queues):
                    ri = q[index]
                    if inst_quality[qi, ri] > 0:
                        cnt[ri] += 1
                if not cnt:
                    break
                elts = cnt.most_common()
                for e, _ in elts:
                    if e in new_star_set:
                        continue
                    new_star_set.add(e)
                    if len(new_star_set) >= self.width:
                        break
                index += 1
            star = [new_star[ri] for ri in new_star_set]

        # select best rules
        rule_list = []
        self.visited = set()
        for r in self.bestr:
            # add r
            self.add_rule(rule_list, r)
        rule_list = sorted(rule_list, key=lambda r: r.quality)
        return rule_list


    def update_best(self, rule):
        """ Maintains a list of best rules for all learning instances. """
        indices = (rule.covered_examples) & (rule.quality + EPS < self.bestq)
        self.bestr[indices] = rule
        self.bestq[indices] = rule.quality

    def add_rule(self, rule_list, rule):
        """ Adds a rule to the final rule list if the rule is not yet in it. """
        if rule is None:
            return
        rkey = rule.covered_examples.tostring()
        if rkey not in self.visited:
            rule_list.append(rule)
        self.visited.add(rkey)

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

            new_rule = RRule(candidate_rule.global_avg,
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



class MeanEvaluator(Evaluator):
    """ Return blended mean value of class of examples 
    covered by rule.
    """
    def __init__(self, K):
        self.K = K

    def evaluate_rule(self, rule):
        return (rule.stats_in[0] + self.K * rule.global_avg) / (rule.covered + self.K)

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
        num_target_covered = rule.stats_in[5]
        return (num_target_covered >= self.min_covered_examples and
               rule.length <= self.max_rule_length)


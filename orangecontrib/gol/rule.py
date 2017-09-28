import numpy as np
from Orange.classification.rules import Rule, Validator, Evaluator

class RRule(Rule):
    """ Basic class describing a single rule. """
    def __init__(self, global_avg, **kwargs):
        Rule.__init__(self, **kwargs)
        self.global_avg = global_avg
        self.stats_in = None
        self.covered = None

    def filter_and_store(self, X, Y, W, target_class, predef_covered=None):
        """ Apply data and target class to a rule. """
        self.target_class = target_class
        if predef_covered is not None:
            self.covered_examples = predef_covered
        else:
            self.covered_examples = np.ones(X.shape[0], dtype=bool)
        for selector in self.selectors:
            self.covered_examples &= selector.filter_data(X)
        self.stats_in = RRule.compute_stats(Y[self.covered_examples])
        self.covered = self.stats_in[6]
        self.mean_prediction = self.stats_in[1]

    @staticmethod
    def compute_stats(vals):
        """  
        Returns a list of statistics: [mean, std, median, min, max, n] from
        the provided numpy array vals.
        """
        if vals.shape[0] == 0:
            return [-1] * 7
        return [np.sum(vals), np.mean(vals), np.std(vals), 
                np.median(vals), np.min(vals), np.max(vals), vals.shape[0]]

    def __str__(self):
        attributes = self.domain.attributes
        class_var = self.domain.class_var

        if self.selectors:
            cond = " AND ".join([attributes[s.column].name + s.op +
                   (str(attributes[s.column].values[int(s.value)])
                    if attributes[s.column].is_discrete
                    else str(s.value)) for s in self.selectors])
        else:
            cond = "TRUE"
        if self.stats_in:
            outcome = "stats({})=[ncovered:{}, mean:{:.4f}, std:{:.4f}, median:{:.4f}, min:{:.4f}, max:{:.4f}]".format(
                       class_var.name, self.stats_in[6], self.stats_in[1], self.stats_in[2],
                       self.stats_in[3], self.stats_in[4], self.stats_in[5])
        else:
            outcome = ""
        return "IF {} THEN {} ".format(cond, outcome)

    def copy(self):
        return RRule(self.global_avg, selectors=self.selectors, domain=self.domain,
                significance_validator = self.significance_validator,
                quality_evaluator = self.quality_evaluator,
                complexity_evaluator = self.complexity_evaluator,
                general_validator = self.general_validator)


        

class RuleLearner:
    def __init__(self, nrules=10, m=2, min_quality=0.9, implicit=False,
                 active=False, min_covered_examples=1, min_traces=5, unique_cov=1, unique_traces=0,
                 default_alpha=1.0, parent_alpha=1.0, max_rule_length=10):

        self.learner = rules.RulesStar(evc=False, width=100, m=m,
                default_alpha=default_alpha, parent_alpha=parent_alpha,
                max_rule_length=max_rule_length, min_covered_examples=min_covered_examples)
        self.learner.target_class = "yes"
        self.nrules = nrules
        self.min_quality = min_quality
        self.min_traces = min_traces
        self.learner.rule_finder.general_validator = TracesValidator(self.learner.rule_finder.general_validator, min_traces)

    def __call__(self, examples, goal_positives, 
                 example_traces, active_traces):
        self.learner.rule_finder.general_validator.goal_positives = goal_positives
        self.learner.rule_finder.general_validator.example_traces = example_traces
        self.learner.rule_finder.general_validator.active_traces = active_traces

        # learn rules
        rules = self.learner(examples).rule_list

        # select rules that:
        # a) have quality at least min_quality, and
        # b) cover at least min_unique_traces active traces
        selected_rules = []
        covered_traces = set()
        rules.sort(key = lambda r: len(r.traces), reverse=True)
        for r in rules:
            if r.quality < self.min_quality:
                continue
            if len(r.traces - covered_traces) < 1:
                continue

            selected_rules.append(r)
            covered_traces |= r.traces

        return selected_rules


class TracesValidator:
    def __init__(self, general_validator, min_traces):
        self.general_validator = general_validator
        self.max_rule_length = self.general_validator.max_rule_length
        self.min_traces = min_traces # each rule has to cover at least this number of traces
        self.goal_positives = None
        self.example_traces = None
        self.active_traces = None

    def validate_rule(self, rule):
        pos = rule.covered_examples & self.goal_positives
        cov_traces = set(self.example_traces[pos])
        rule.traces = cov_traces & self.active_traces
        if len(rule.traces) < self.min_traces:
            return False
        if self.general_validator:
            return self.general_validator.validate_rule(rule)
        return True


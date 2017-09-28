import numpy as np
from Orange.data import Table
from orangecontrib.gol.goal import GoalValidatorDepth, Goal
from orangecontrib.gol.rule_learner import RegressiveRuleLearner

class GraphConceptualizer:
    def __init__(self, rule_learner=None, goal_validator=None):
        self.rule_learner = rule_learner if rule_learner is not None else \
                            RegressiveRuleLearner()
        self.goal_validator = goal_validator if goal_validator is not None else \
                              GoalValidatorDepth()


    def __call__(self, learn_examples, learn_states, learn_traces, final_goal):
        # create initial goal graph
        covered = final_goal(learn_examples)
        complexities = self.goal_validator(final_goal, learn_examples, 
                                           learn_states, covered)
        ggraph = GoalNode(final_goal, [], complexities)
        while True:
            print("covered", np.sum(covered))
            if np.all(covered):
                break
            
            # learn a rule given current complexities and coverage
            indices = ~covered
            curr_complexities = complexities[indices]
            curr_examples = Table.from_table_rows(learn_examples, indices)
            curr_examples.Y = curr_complexities

            rule = self.rule_learner(curr_examples)
            
            goal = Goal(rule)
            goal_complexities = self.goal_validator(goal, learn_examples,
                                                    learn_states, covered)
            # calculate new covered
            covered[indices] |= rule.covered_examples
            # add to covered also those examples that are closer to new goal
            # than the distance between goal and previous goals
            close_goal = goal_complexities <= rule.mean_prediction
            close_before = complexities <= rule.mean_prediction
            covered |= close_goal & close_before
            print("new covered", np.sum(covered))

            # select parents
            parents = []
            nodes = ggraph.get_nodes()
            for gnode in nodes:
                closest = gnode.complexities == complexities
                if (closest[indices] & rule.covered_examples).any():
                    parents.append(gnode)
            
            # create new goal node
            new_node = GoalNode(goal, parents, rule.mean_prediction + goal_complexities)

            # update children
            for p in parents:
                p.children.append(new_node)

            # calculate new complexities
            print("prej", complexities[~covered].sum(), goal_complexities[~covered].sum())
            complexities = np.minimum(complexities, rule.mean_prediction + goal_complexities)
            print("potem", complexities[~covered].sum())

            print(ggraph)
            



class GraphConceptualizer__:
    def __init__(self, rule_learner=None, goal_validator=None):
        self.rule_learner = rule_learner if rule_learner is not None else \
                            RegressiveRuleLearner()
        self.goal_validator = goal_validator if goal_validator is not None else \
                              GoalValidatorDepth()
   
    def __call__(self, learn_examples, learn_states, learn_traces, final_goal):
        # create initial goal graph
        covered = final_goal(learn_examples)
        complexities = self.goal_validator(final_goal, learn_examples, 
                                      learn_states, covered)
        ggraph = GoalNode(final_goal, 0, complexities, [], set(learn_traces), covered)

        print(ggraph.complexities)
        # loop until there are unexpanded goals
        while True:
            # get a set of unexpanded goals, select goal with lowest complexity
            goals = ggraph.get_unexpanded()
            if not goals:
                break
            selected = min(goals, key=lambda x: x.avg_complexity)
            selected.expanded = True
            if np.all(selected.covered):
                continue
            print("selected", selected)

            # prepare learning examples for learning for selected
            complexities = np.array(selected.complexities)
            for g in goals:
                if g == selected:
                    continue
                complexities = np.minimum(complexities, g.complexities)
            # compute which examples goes to which goal
            best_per_goal = []
            for g in goals:
                best_per_goal.append(complexities == g.complexities)

            learn_examples.Y = complexities
           
            # select relevant examples = 
            # examples that are not covered by selected and
            # their trace is in active traces
            relevant = ~covered & np.in1d(learn_traces, list(selected.traces))

            curr_indices = ~covered
            curr_relevant = relevant[curr_indices]
            curr_traces = learn_traces[curr_indices]
            curr_complexities = complexities[curr_indices]
            curr_examples = Table.from_table_rows(learn_examples, curr_indices)
            curr_rules = [g.goal.rule.copy() for g in goals]

            # learn rules 
            rules = self.rule_learner(curr_examples, curr_relevant, curr_traces, curr_rules)
            for r in rules:
                print(r, 'quality: ', r.quality)
                goal = Goal(r)
                # compute new covered 
                new_covered = np.array(selected.covered)
                # new covered also includes all covered examples by rule
                new_covered[curr_indices] |= r.covered_examples
                # covered are also examples that are closer than avg_complexity
                # of covered examples
                new_covered |= selected.complexities < r.mean_prediction
                print("new_covered", new_covered.sum(), selected.covered.sum())
                # compute new complexities
                new_complexities = self.goal_validator(goal, learn_examples,
                                                   learn_states, new_covered)
                new_node = GoalNode(goal, r.mean_prediction, 
                                    new_complexities + r.mean_prediction,
                                    [], set(), new_covered)
                print("new complexities", new_node.avg_complexity, np.mean(new_node.complexities))

                # connect parent nodes and new node
                # if parents trace leeds thorugh covered and parent 
                # is actually the best option, add it
                for gi, g in enumerate(goals):
                    # is there a trace in parent goal that is the same 
                    gtraces = set(curr_traces[best_per_goal[gi][curr_indices] & r.covered_examples])
                    intersect = gtraces & g.traces
                    if intersect:
                        g.children.append(new_node)
                        new_node.parents.append(g)
                        # add traces
                        new_node.traces |= intersect
            print(ggraph)


class GraphConceptualizer_:
    def __init__(self, rule_learner=None, min_depth=5, max_depth=6):
        self.min_depth = min_depth
        self.max_depth = max_depth
        if rule_learner is not None:
            self.rule_learner = rule_learner
        else:
            self.rule_learner = RuleLearner()

    def __call__(self, learn_examples, learn_states, learn_traces, final_goal):
        # preprocessing of data to enable faster search in goal validation
        goal_validator = GoalValidator(self.max_depth)

        # create initial goal graph
        covered = final_goal(learn_examples)
        ggraph = GoalNode(final_goal, 0, [], set(learn_traces), covered)

        # loop until there are unexpanded goals
        while True:
            # get a set of unexpanded goals, select goal with lowest complexity
            goals = ggraph.get_unexpanded()
            if not goals:
                break
            selected = min(goals, key=lambda x: x.complexity)
            selected.expanded = True
            if np.all(selected.covered):
                continue
            
            # set active traces (all traces without solved ones from existing
            # children)
            active_traces = set(selected.traces)
            for c in selected.children:
                active_trace -= c.traces

            depth = self.min_depth
            while active_traces and depth < self.max_depth:
                # getting positive examples for selected goal
                # (memoizing as much as possible)
                if depth in selected.positives:
                    positives = selected.positives[depth]
                else:
                    positives = goal_validator(selected.goal, learn_examples, 
                                           learn_states, selected.covered, depth)
                    selected.positives[depth] = positives

                # now compute which other non expanded goals are achievable 
                # from these examples (again memoize as much as possible) 
                goal_positives = [] # one boolean array for each unexpanded goal
                other_positives = np.array(positives) # a union of all positives
                for gnode in goals:
                    if gnode == selected:
                        goal_positives.append(positives)
                        continue
                    if depth in gnode.positives:
                        gpos = gnode.positives[depth]
                    else:
                        gpos = goal_validator(gnode.goal, learn_examples, learn_states,
                                              selected.covered, depth)
                        gnode.positives[depth] = gpos
                    goal_positives.append(gpos)
                    other_positives = other_positives | gpos
            
                print("positives", positives.sum(), other_positives.sum())

                # create examples for learning
                indices = ~covered
                learn_examples.Y[:] = 0
                learn_examples.Y[other_positives] = 1
                curr_examples = Table.from_table_rows(learn_examples, indices)

                # get traces of learning examples
                curr_traces = learn_traces[indices]

                # get true positives for learning examples
                curr_positives = positives[indices]
                
                # learn rules
                rules = self.rule_learner(curr_examples, curr_positives, 
                                          curr_traces, active_traces)
                for r in rules:
                    print(r, r.quality, "depth: ", depth)
                    goal = Goal(r)
                    # compute new covered 
                    new_covered = selected.covered | (positives & np.in1d(learn_traces, list(r.traces)))
                    new_node = GoalNode(goal, [selected], r.traces, new_covered)
                    selected.children.append(new_node)
                    # iterate through all unexpanded nodes
                    for gnode in goals:
                        if gnode == selected:
                            continue
                        # check if gnode can be achieved from examples covered 
                        # by rule r


                    # update active traces
                    active_traces -= r.traces
                    print("active", len(active_traces))
                depth += 1
            print(ggraph)
            covered = new_covered
        return ggraph


class GoalNode:
    def __init__(self, goal, parents, complexities):
        """
        A goal-node contains the actual goal, links to its parent nodes,
        instances (an array of ones and zeros) covered by this particular strategy.
        Traces is a set of traces covered by the strategy.
        Covered are indexes of covered examples by this strategy.
        """
        self.goal = goal
        self.children = []
        self.parents = parents
        #self.expanded = False
        #self.traces = traces
        #self.covered = covered
        #self.avg_complexity = avg_complexity
        self.complexities = complexities

    def get_open_goals(self):
        if not self.expanded:
            return set([self])
        open_goals = set()
        for c in self.children:
            open_goals |= c.get_open_goals()
        return open_goals

    def get_unexpanded(self):
        if not self.expanded:
            return set([self])
        allun = set()
        for c in self.children:
            allun |= c.get_unexpanded()
        return allun

    def get_nodes(self):
        nodes = set([self])
        for c in self.children:
            nodes |= c.get_nodes()
        return nodes

    def __str__(self):
        return self.str_rec(0)

    def __eq__(self, other):
        return self.goal == other.goal

    def __hash__(self):
        return self.goal.__hash__()

    def str_rec(self, indent):
        gstr = ' ' * indent
        gstr += self.goal2str() + '\n'
        for c in self.children:
            gstr += c.str_rec(indent+2)
        return gstr

    def goal2str(self):
        return 'g: {}'.format(
            str(self.goal))


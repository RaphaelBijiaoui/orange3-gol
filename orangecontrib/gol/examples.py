""" Creates examples from states for goal-oriented learning.
"""
from Orange.data import Table, Domain, Instance, \
                        StringVariable, ContinuousVariable

def create_data(data_desc):
    example_states, example_traces = data_desc.get_example_states()
    return create_data_from_states(example_states, example_traces), example_states, example_traces

def create_data_from_states(example_states, example_traces):
    data_desc = example_states[0].domain
    attributes = data_desc.get_attributes()
    domain = Domain(attributes,
                    ContinuousVariable.make("complexity"),
                    metas = [StringVariable.make("id"), ContinuousVariable("trace")])
    data = Table.from_domain(domain)
    for si, s in enumerate(example_states):
        e = Instance(domain)
        for f in attributes:
            e[f] = s.get_attribute(f)
        e["id"] = s.get_id()
        e["trace"] = example_traces[si]
        data.append(e)
    return data

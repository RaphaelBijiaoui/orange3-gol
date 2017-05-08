from Orange.data import Table, Domain, Instance, DiscreteVariable, \
                        StringVariable

def create_data(states):
    features = states[0].features
    domain = Domain(features, 
                    DiscreteVariable.make("goal", values=["no","yes"]),
                    metas = [StringVariable.make("id")])
    data = Table.from_domain(domain)
    for s in states:
        e = Instance(domain)
        for f in features:
            e[f] = s.get_feature(f)
        e["id"] = s.get_id()
        data.append(e)
    return data

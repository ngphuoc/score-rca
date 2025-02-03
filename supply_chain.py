import pandas as pd
import secrets
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dowhy.gcm as gcm
import networkx


ASINS = [secrets.token_hex(5).upper() for i in range(1000)]

def get_supply_chain_dag(data_week1):
    # Causal Attribution Analysis
    causal_graph = nx.DiGraph([('demand', 'submitted'),
                               ('constraint', 'submitted'),
                               ('submitted', 'confirmed'),
                               ('confirmed', 'received')])
    # gcm.util.plot(causal_graph)
    causal_model = gcm.StructuralCausalModel(causal_graph)
    # Automatically assign appropriate causal models to each node in graph
    gcm.auto.assign_causal_mechanisms(causal_model, data_week1)
    return causal_model


def buying_data(alpha, beta, demand_mean, num_samples):
    unobserved = {}
    unobserved["constraint"] = np.random.gamma(1, scale=1, size=num_samples)
    unobserved["demand"] = np.random.gamma(demand_mean, scale=1, size=num_samples)
    unobserved["submitted"] = np.random.gamma(1, scale=1, size=num_samples)
    submitted = unobserved["demand"] - unobserved["constraint"] + unobserved["submitted"]
    unobserved["confirmed"] = np.random.gamma(0.1, scale=1, size=num_samples)
    confirmed = alpha * submitted + unobserved["confirmed"]
    unobserved["received"] = np.random.gamma(0.1, scale=1, size=num_samples)
    received = beta * confirmed + unobserved["received"]
    observed = pd.DataFrame(dict(asin=ASINS,
                              demand = np.round(unobserved["demand"]),
                              constraint=np.round(unobserved["constraint"]),
                              submitted = np.round(submitted),
                              confirmed = np.round(confirmed),
                              received = np.round(received)))
    return observed, pd.DataFrame(unobserved)


def supply_chain_data(num_samples):
    """
    Using linear ANMs, we generate data (or draw i.i.d. samples) from the distribution of each variable. We use the Gamma distribution for noise terms mainly to mimic real-world setting, where the distribution of variables often show heavy-tail behaviour. Between two weeks, we only change the data-generating process (causal mechanism) of demand and confirmed respectively by changing the value of demand mean from 2 to 4, and linear coefficient \alpha from 1 to 2.
    """
    # we change the parameters alpha and demand_mean between weeks
    observed_week1, unobserved_week1 = buying_data(1, 1, demand_mean=2, num_samples=num_samples)
    observed_week1['week'] = 'week1'
    observed_week2, unobserved_week2 = buying_data(3, 1, demand_mean=4, num_samples=num_samples)
    observed_week2['week'] = 'week2'
    observed = observed_week1.append(observed_week2, ignore_index=True)
    # unobserved = pd.concat(unobserved_week1, unobserved_week2, ignore_index=True)
    # write data to a csv file
    # data.to_csv('supply_chain_week_over_week.csv', index=False)

    normal_data, normal_noise = observed_week1, unobserved_week1
    causal_model = get_supply_chain_dag(observed_week1)
    # ground_truth_dag, target_node, ordered_nodes, normal_data, normal_noise = supply_chain_data(n_samples)
    # return observed_week1, unobserved_week1, observed_week2, unobserved_week2, causal_model
    gcm.fit(causal_model, normal_data)
    target_node = "received"
    ordered_nodes = networkx.topological_sort(causal_model.graph)
    return causal_model, target_node, ordered_nodes, normal_data, normal_noise


def example_analysis():

    # Our target of interest is the average value of received over those two weeks.
    data.groupby(['week']).mean()[['received']].plot(kind='bar', title='average received', legend=False);
    data_week2.received.mean() - data_week1.received.mean()
    # The average value of received quantity has increased from week w1 to week w2. Why?

    # Ad-hoc attribution analysis
    data.groupby(['week']).mean().plot(kind='bar', title='average', legend=True);

    # disabling progress bar to not clutter the output here
    gcm.config.disable_progress_bars()
    # setting random seed for reproducibility
    np.random.seed(10)

    causal_model = get_supply_chain_dag()
    # call the API for attributing change in the average value of `received`
    contributions = gcm.distribution_change(causal_model,
                                            data_week1,
                                            data_week2,
                                            'received',
                                            num_samples=2000,
                                            difference_estimation_func=lambda x1, x2 : np.mean(x2) - np.mean(x1))

    plt.bar(contributions.keys(), contributions.values())
    plt.ylabel('Contribution')
    plt.show()

    median_contribs, uncertainty_contribs = gcm.confidence_intervals(
        gcm.bootstrap_sampling(gcm.distribution_change,
                               causal_model,
                               data_week1,
                               data_week2,
                               'received',
                               num_samples=2000,
                               difference_estimation_func=lambda x1, x2 : np.mean(x2) - np.mean(x1)),
        confidence_level=0.95,
        num_bootstrap_resamples=10)

    yerr_plus = [uncertainty_contribs[node][1] - median_contribs[node] for node in median_contribs.keys()]
    yerr_minus = [median_contribs[node] - uncertainty_contribs[node][0] for node in median_contribs.keys()]
    plt.bar(median_contribs.keys(), median_contribs.values(), yerr=np.array([yerr_minus, yerr_plus]), ecolor='black')
    plt.ylabel('Contribution')
    plt.show()


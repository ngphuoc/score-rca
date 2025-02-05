import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
import matplotlib.pyplot as plt
from scipy.stats import truncexpon, halfnorm
import numpy as np
import networkx

# DAG
def get_micro_service_dag():
    causal_graph = nx.DiGraph([('www', 'Website'),
                               ('Auth Service', 'www'),
                               ('API', 'www'),
                               ('Customer DB', 'Auth Service'),
                               ('Customer DB', 'API'),
                               ('Product Service', 'API'),
                               ('Auth Service', 'API'),
                               ('Order Service', 'API'),
                               ('Shipping Cost Service', 'Product Service'),
                               ('Caching Service', 'Product Service'),
                               ('Product DB', 'Caching Service'),
                               ('Customer DB', 'Product Service'),
                               ('Order DB', 'Order Service')])

    causal_model = gcm.StructuralCausalModel(causal_graph)

    for node in causal_graph.nodes:
        if len(list(causal_graph.predecessors(node))) > 0:
            causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
        else:
            causal_model.set_causal_mechanism(node, gcm.ScipyDistribution(halfnorm))

    return causal_model


def create_observed_latency_data(unobserved_intrinsic_latencies, w_auth_api = 1, w_cus_prod = 1):
    observed_latencies = {}
    observed_latencies['Product DB'] = unobserved_intrinsic_latencies['Product DB']
    observed_latencies['Customer DB'] = unobserved_intrinsic_latencies['Customer DB']
    observed_latencies['Order DB'] = unobserved_intrinsic_latencies['Order DB']
    observed_latencies['Shipping Cost Service'] = unobserved_intrinsic_latencies['Shipping Cost Service']
    observed_latencies['Caching Service'] = np.random.choice([0, 1], size=(len(observed_latencies['Product DB']),),
                                                             p=[.5, .5]) * \
                                            observed_latencies['Product DB'] \
                                            + unobserved_intrinsic_latencies['Caching Service']
    observed_latencies['Product Service'] = np.maximum(np.maximum(observed_latencies['Shipping Cost Service'],
                                                                  observed_latencies['Caching Service']),
                                                       w_cus_prod * observed_latencies['Customer DB']) \
                                            + unobserved_intrinsic_latencies['Product Service']
    observed_latencies['Auth Service'] = observed_latencies['Customer DB'] \
                                         + unobserved_intrinsic_latencies['Auth Service']
    observed_latencies['Order Service'] = observed_latencies['Order DB'] \
                                          + unobserved_intrinsic_latencies['Order Service']
    observed_latencies['API'] = observed_latencies['Product Service'] \
                                + observed_latencies['Customer DB'] \
                                + w_auth_api * observed_latencies['Auth Service'] \
                                + observed_latencies['Order Service'] \
                                + unobserved_intrinsic_latencies['API']
    observed_latencies['www'] = observed_latencies['API'] \
                                + observed_latencies['Auth Service'] \
                                + unobserved_intrinsic_latencies['www']
    observed_latencies['Website'] = observed_latencies['www'] \
                                    + unobserved_intrinsic_latencies['Website']

    return pd.DataFrame(observed_latencies)


def unobserved_intrinsic_latencies_normal(num_samples):
    return {
        'Website': truncexpon.rvs(size=num_samples, b=3, scale=0.2),
        'www': truncexpon.rvs(size=num_samples, b=2, scale=0.2),
        'API': halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        'Auth Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Product Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Order Service': halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        'Shipping Cost Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Caching Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.1),
        'Order DB': truncexpon.rvs(size=num_samples, b=5, scale=0.2),
        'Customer DB': truncexpon.rvs(size=num_samples, b=6, scale=0.2),
        'Product DB': truncexpon.rvs(size=num_samples, b=10, scale=0.2)
    }


def unobserved_intrinsic_latencies_anomalous(num_samples):
    return {
        'Website': truncexpon.rvs(size=num_samples, b=3, scale=0.2),
        'www': truncexpon.rvs(size=num_samples, b=2, scale=0.2),
        'API': halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        'Auth Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Product Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Order Service': halfnorm.rvs(size=num_samples, loc=0.5, scale=0.2),
        'Shipping Cost Service': halfnorm.rvs(size=num_samples, loc=0.1, scale=0.2),
        'Caching Service': 2 + halfnorm.rvs(size=num_samples, loc=0.1, scale=0.1),
        'Order DB': truncexpon.rvs(size=num_samples, b=5, scale=0.2),
        'Customer DB': truncexpon.rvs(size=num_samples, b=6, scale=0.2),
        'Product DB': truncexpon.rvs(size=num_samples, b=10, scale=0.2)
    }


def micro_service_data(n_samples):
    normal_noise = unobserved_intrinsic_latencies_normal(n_samples)
    normal_data = create_observed_latency_data(normal_noise)
    causal_model = get_micro_service_dag()
    gcm.fit(causal_model, normal_data)
    target_node = "Website"
    ordered_nodes = networkx.topological_sort(causal_model.graph)
    return (causal_model, target_node, ordered_nodes, normal_data, normal_noise)


# normal_data = create_observed_latency_data(unobserved_intrinsic_latencies_normal(10000))
# outlier_data = create_observed_latency_data(unobserved_intrinsic_latencies_anomalous(1000))
# outlier_data = pd.read_csv("rca_microservice_architecture_anomaly.csv")
# plt.rcParams['figure.figsize'] = [13, 13] # Make plot bigger
# gcm.util.plot(causal_graph)


#############################################
# Scenario 1: Observing a single outlier_data

# outlier_data.iloc[0]['Website']-normal_data['Website'].mean()
# For this customer, Website was roughly 2 seconds slower than for other customers on average. Why?
# gcm.config.disable_progress_bars() # to disable print statements when computing Shapley values

# median_attribs, uncertainty_attribs = gcm.confidence_intervals(
#         gcm.bootstrap_training_and_sampling(gcm.attribute_anomalies,
#                                             causal_model,
#                                             normal_data,
#                                             target_node='Website',
#                                             anomaly_samples=outlier_data),
#         num_bootstrap_resamples=10)

# def bar_plot_with_uncertainty(median_attribs, uncertainty_attribs, ylabel='Attribution Score', figsize=(8, 3), bwidth=0.8, xticks=None, xticks_rotation=90):
#     fig, ax = plt.subplots(figsize=figsize)
#     yerr_plus = [uncertainty_attribs[node][1] - median_attribs[node] for node in median_attribs.keys()]
#     yerr_minus = [median_attribs[node] - uncertainty_attribs[node][0] for node in median_attribs.keys()]
#     plt.bar(median_attribs.keys(), median_attribs.values(), yerr=np.array([yerr_minus, yerr_plus]), ecolor='#1E88E5', color='#ff0d57', width=bwidth)
#     plt.xticks(rotation=xticks_rotation)
#     plt.ylabel(ylabel)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     if xticks:
#         plt.xticks(list(median_attribs.keys()), xticks)
#     plt.show()

# bar_plot_with_uncertainty(median_attribs, uncertainty_attribs)


##########################################################
# Scenario 2: Observing permanent degradation of latencies

# outlier_data = pd.read_csv("rca_microservice_architecture_anomaly_1000.csv")
# outlier_data['Website'].mean() - normal_data['Website'].mean()


###############################################################
# Scenario 3: Simulating the intervention of shifting resources

# median_mean_latencies, uncertainty_mean_latencies = gcm.confidence_intervals(
#         lambda : gcm.bootstrap_training_and_sampling(gcm.interventional_samples,
#                                                      causal_model,
#                                                      outlier_data,
#                                                      interventions = {
#                                                          "Caching Service": lambda x: x-1,
#                                                          "Shipping Cost Service": lambda x: x+2
#                                                          },
#                                                      observed_data=outlier_data)().mean().to_dict(),
#         num_bootstrap_resamples=10)

# avg_website_latency_before = outlier_data.mean().to_dict()['Website']
# bar_plot_with_uncertainty(dict(before=avg_website_latency_before, after=median_mean_latencies['Website']),
#                           dict(before=np.array([avg_website_latency_before, avg_website_latency_before]), after=uncertainty_mean_latencies['Website']),
#                           ylabel='Avg. Website Latency',
#                           figsize=(3, 2),
#                           bwidth=0.4,
#                           xticks=['Before', 'After'],
#                           xticks_rotation=45)


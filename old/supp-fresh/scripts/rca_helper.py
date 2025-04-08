import random

import networkx
import numpy
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score

from dowhy.gcm import InvertibleStructuralCausalModel, ScipyDistribution, AdditiveNoiseModel, is_root_node, \
    MeanDeviationScorer, draw_samples, PredictionModel
from dowhy.gcm._noise import compute_noise_from_data, noise_samples_of_ancestors
from dowhy.gcm.anomaly import anomaly_score_attributions
from dowhy.gcm.graph import PARENTS_DURING_FIT, get_ordered_predecessors
from dowhy.gcm.ml import SklearnRegressionModel
from dowhy.gcm.util.general import column_stack_selected_numpy_arrays, convert_to_data_frame


def random_linear_dag_generator(num_root_nodes, num_downstream_nodes):
    def sample_natural_number(init_mass) -> int:
        current_mass = init_mass
        probability = numpy.random.uniform(0, 1)
        k = 1

        is_searching = True

        while is_searching:
            if probability <= current_mass:
                return k
            else:
                k += 1
                current_mass += 1 / (k ** 2)

    causal_dag = InvertibleStructuralCausalModel(networkx.DiGraph())

    all_nodes = []

    for i in range(num_root_nodes):
        random_distribution_obj = ScipyDistribution(stats.norm, loc=0, scale=1)

        new_root = 'X' + str(i)
        causal_dag.graph.add_node(new_root)
        causal_dag.set_causal_mechanism(new_root, random_distribution_obj)
        causal_dag.graph.nodes[new_root][PARENTS_DURING_FIT] = get_ordered_predecessors(causal_dag.graph, new_root)

        all_nodes.append(new_root)

    for i in range(num_downstream_nodes):
        parents = numpy.random.choice(all_nodes,
                                      min(sample_natural_number(init_mass=0.6), len(all_nodes)),
                                      replace=False)

        new_child = 'X' + str(i + num_root_nodes)
        causal_dag.graph.add_node(new_child)

        linear_reg = LinearRegression()
        # Positive coefficients only to ensure clear ground truth
        linear_reg.coef_ = numpy.random.uniform(0, 5, len(parents))
        linear_reg.intercept_ = 0

        causal_mechanism = AdditiveNoiseModel(SklearnRegressionModel(linear_reg),
                                              ScipyDistribution(stats.norm, loc=0, scale=1))
        causal_dag.set_causal_mechanism(new_child, causal_mechanism)

        for parent in parents:
            causal_dag.graph.add_edge(parent, new_child)

        causal_dag.graph.nodes[new_child][PARENTS_DURING_FIT] = get_ordered_predecessors(causal_dag.graph, new_child)

        all_nodes.append(new_child)

    return causal_dag


def draw_anomaly_samples(causal_graph, num_samples, k, list_of_potential_anomaly_nodes):
    drawn_samples = {}
    drawn_noise_samples = {}
    lambdas = {}

    noises = numpy.random.uniform(3, 5, num_samples)
    anomaly_nodes = random.sample(list_of_potential_anomaly_nodes, k)

    for i, node in enumerate(networkx.topological_sort(causal_graph.graph)):
        causal_model = causal_graph.causal_mechanism(node)

        if is_root_node(causal_graph.graph, node):
            if node in anomaly_nodes:
                drawn_noise_samples[node] = numpy.array(noises)
            else:
                drawn_noise_samples[node] = numpy.zeros(num_samples)

            drawn_samples[node] = drawn_noise_samples[node]
        else:
            if node in anomaly_nodes:
                drawn_noise_samples[node] = numpy.array(noises)
            else:
                drawn_noise_samples[node] = numpy.zeros(num_samples)

            parent_samples = column_stack_selected_numpy_arrays(drawn_samples,
                                                                get_ordered_predecessors(causal_graph.graph, node))

            drawn_samples[node] = causal_model.evaluate(parent_samples, drawn_noise_samples[node])

    return convert_to_data_frame(drawn_samples), convert_to_data_frame(drawn_noise_samples), lambdas


def our_approach_rankings(causal_dag, target_node, anomaly_samples, target_prediction_method, nodes_order):
    noise_of_anomaly_samples = compute_noise_from_data(causal_dag, anomaly_samples)

    node_samples, noise_samples = noise_samples_of_ancestors(causal_dag, target_node, 5000)
    scorer = MeanDeviationScorer()
    scorer.fit(node_samples[target_node].to_numpy())

    attributions = anomaly_score_attributions(noise_of_anomaly_samples[nodes_order].to_numpy(),
                                              noise_samples[nodes_order].to_numpy(),
                                              lambda x: scorer.score(target_prediction_method(x)),
                                              attribute_mean_deviation=False)

    result = []

    for i in range(attributions.shape[0]):
        tmp = {}
        for j in range(attributions.shape[1]):
            tmp[nodes_order[j]] = attributions[i, j]

        result.append(tmp)

    return result


def naive_approach(causal_dag, nodes_order, anomaly_samples):
    node_samples = draw_samples(causal_dag, 5000)
    result = []
    scorers = {}

    for node in nodes_order:
        scorer = MeanDeviationScorer()
        scorer.fit(node_samples[node].to_numpy())
        scorers[node] = scorer

    for i in range(anomaly_samples.shape[0]):
        tmp = {}
        for node in nodes_order:
            tmp[node] = scorers[node].score(anomaly_samples.iloc[i][node])

        result.append(tmp)

    return result


class MySquaredRegressor(PredictionModel):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self,
            X: numpy.ndarray,
            Y: numpy.ndarray) -> None:
        self.model.fit(X ** 2, Y)

    def predict(self,
                X: numpy.ndarray) -> numpy.ndarray:
        return self.model.predict(X ** 2).reshape(-1, 1)


def get_noise_coefficient(causal_dag, target_node):
    result = {}
    for node in causal_dag.graph.nodes:
        all_paths = list(networkx.all_simple_paths(causal_dag.graph, node, target_node))
        if len(all_paths) == 0:
            continue

        noise_coef = 0
        for path in all_paths:
            path.reverse()
            tmp_coef = 1
            for i in range(0, len(path) - 1):
                current_node = path[i]
                upstream_node = path[i + 1]
                parent_coef_index = get_ordered_predecessors(causal_dag.graph, current_node).index(upstream_node)
                tmp_coef *= causal_dag.causal_mechanism(current_node).prediction_model.sklearn_model.coef_[
                    parent_coef_index]
            noise_coef += tmp_coef

        result[node] = noise_coef
    result[target_node] = 1

    return result


def evaluate_results_ndcg(method_scores, ground_truth_scores, overall_max_k):
    result = {}
    overall_max_k += 1
    for k in range(1, overall_max_k):
        result[k] = []

    for i in range(len(method_scores)):
        for k in range(1, overall_max_k):
            result[k].append(ndcg_score([ground_truth_scores[i]], [method_scores[i]], k=k))

    return result


def summarize_result(result_dict):
    mean, std = [], []

    for k in sorted(result_dict.keys()):
        mean.append(numpy.mean(result_dict[k]))
        std.append(numpy.std(result_dict[k]))

    return mean, std

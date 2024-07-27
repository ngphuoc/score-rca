import random
import networkx
import numpy
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score
import dowhy
from dowhy.gcm import InvertibleStructuralCausalModel, ScipyDistribution, AdditiveNoiseModel, is_root_node, \
    MeanDeviationScorer, draw_samples, PredictionModel
from dowhy.gcm._noise import compute_noise_from_data, noise_samples_of_ancestors
from dowhy.gcm.anomaly import anomaly_score_attributions
from dowhy.gcm.graph import PARENTS_DURING_FIT, get_ordered_predecessors
from dowhy.gcm.ml import SklearnRegressionModel
from dowhy.gcm.util.general import column_stack_selected_numpy_arrays, convert_to_data_frame
from scipy.stats import norm
import numpy as np
from dowhy.gcm.shapley import ShapleyApproximationMethods
import pickle

# Compute the z-score, and tail probability outlier score.
#     zscore(x) = |x - EX| / σ(X)
#     outlier score (x) = -log P(|X - EX| >= |x - EX|)
class ZOutlierScorePy:
    def __init__(self, X):
        self.loc = np.mean(X)
        self.scale = np.std(X)

    def score(self, X):
        return -norm.logcdf(-np.abs((X - self.loc)) / self.scale)


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

        new_root = 'X' + str(i).zfill(2)
        causal_dag.graph.add_node(new_root)
        causal_dag.set_causal_mechanism(new_root, random_distribution_obj)
        causal_dag.graph.nodes[new_root][PARENTS_DURING_FIT] = get_ordered_predecessors(causal_dag.graph, new_root)

        all_nodes.append(new_root)

    for i in range(num_downstream_nodes):
        parents = numpy.random.choice(all_nodes,
                                      min(sample_natural_number(init_mass=0.6), len(all_nodes)),
                                      replace=False)

        new_child = 'X' + str(i + num_root_nodes).zfill(2)
        causal_dag.graph.add_node(new_child)

        linear_reg = LinearRegression()
        # Positive coefficients only to ensure clear ground truth
        linear_reg.coef_ = numpy.random.uniform(0.5, 2, len(parents))
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

            parent_samples = column_stack_selected_numpy_arrays(drawn_samples, get_ordered_predecessors(causal_graph.graph, node))

            drawn_samples[node] = causal_model.evaluate(parent_samples, drawn_noise_samples[node])

    return convert_to_data_frame(drawn_samples), convert_to_data_frame(drawn_noise_samples), lambdas


def pickle_save(fname, dic):
    with open(fname, 'wb') as f:
        pickle.dump(dic, f)


def pickle_load(fname):
    with open(fname, 'rb') as f:
        dic = pickle.load(f)
    return dic


def draw_samples_2(causal_graph, num_samples, anomaly_nodes=[]):
    drawn_samples = {}
    drawn_noise_samples = {}
    lambdas = {}

    for i, node in enumerate(networkx.topological_sort(causal_graph.graph)):
        causal_model = causal_graph.causal_mechanism(node)

        if node in anomaly_nodes:
            noise = numpy.random.uniform(3, 5, num_samples)
        else:
            if is_root_node(causal_graph.graph, node):
                noise = causal_model.draw_samples(num_samples)
            else:
                noise = causal_model.draw_noise_samples(num_samples)
                # noise = numpy.zeros(num_samples)

        drawn_noise_samples[node] = noise

        if is_root_node(causal_graph.graph, node):
            drawn_samples[node] = noise

        else:
            parent_samples = column_stack_selected_numpy_arrays(drawn_samples, get_ordered_predecessors(causal_graph.graph, node))
            drawn_samples[node] = causal_model.evaluate(parent_samples, noise)

    return convert_to_data_frame(drawn_samples), convert_to_data_frame(drawn_noise_samples), lambdas


def draw_samples_2_bac2(causal_graph, num_samples, anomaly_nodes=[]):
    drawn_samples = {}
    drawn_noise_samples = {}
    lambdas = {}

    for i, node in enumerate(networkx.topological_sort(causal_graph.graph)):
        causal_model = causal_graph.causal_mechanism(node)

        if node in anomaly_nodes:
            noise = numpy.random.uniform(3, 5, num_samples)
        else:
            if is_root_node(causal_graph.graph, node):
                noise = causal_model.draw_samples(num_samples)
            else:
                noise = causal_model.draw_noise_samples(num_samples)

        drawn_noise_samples[node] = noise

        if is_root_node(causal_graph.graph, node):
            drawn_samples[node] = noise

        else:
            parent_samples = column_stack_selected_numpy_arrays(drawn_samples, get_ordered_predecessors(causal_graph.graph, node))
            drawn_samples[node] = causal_model.evaluate(parent_samples, noise)

    return convert_to_data_frame(drawn_samples), convert_to_data_frame(drawn_noise_samples), lambdas


def draw_samples_2_bac(causal_graph, num_samples):
    drawn_samples = {}
    drawn_noise_samples = {}
    lambdas = {}

    for i, node in enumerate(networkx.topological_sort(causal_graph.graph)):
        causal_model = causal_graph.causal_mechanism(node)

        if is_root_node(causal_graph.graph, node):
            noise = causal_model.draw_samples(num_samples)
            drawn_noise_samples[node] = noise
            drawn_samples[node] = noise

        else:
            noise = causal_model.draw_noise_samples(num_samples)
            drawn_noise_samples[node] = noise
            parent_samples = column_stack_selected_numpy_arrays(drawn_samples, get_ordered_predecessors(causal_graph.graph, node))
            drawn_samples[node] = causal_model.evaluate(parent_samples, noise)

    return convert_to_data_frame(drawn_samples), convert_to_data_frame(drawn_noise_samples), lambdas


def our_approach_rankings(causal_dag, target_node, anomaly_samples, target_prediction_method, nodes_order, zscorer, ref_samples, approximation_method):
    noise_of_anomaly_samples = compute_noise_from_data(causal_dag, anomaly_samples)

    # node_samples, noise_samples = noise_samples_of_ancestors(causal_dag, target_node, 1000)
    # scorer = MeanDeviationScorer()
    # scorer.fit(node_samples[target_node].to_numpy())
    scorer =  zscorer
    noise_samples = ref_samples
    shapley_config = dowhy.gcm.shapley.ShapleyConfig(approximation_method)
    attributions = anomaly_score_attributions(noise_of_anomaly_samples[nodes_order].to_numpy(),
                                              noise_samples[nodes_order].to_numpy(),
                                              lambda x: scorer.score(target_prediction_method(x)),
                                              attribute_mean_deviation=False,
                                              shapley_config=shapley_config,
                                              )

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


def get_ground_truth_rankings(ground_truth_dag, target_node, all_noise_samples, nodes_order, max_k, overall_max_k):
    ground_truth_rankings = []
    ground_truth_noise_coefficients = get_noise_coefficient(ground_truth_dag, target_node)
    ground_truth_scores = []
    for i in range(all_noise_samples.shape[0]):
        tmp = {node: ground_truth_noise_coefficients[node] * all_noise_samples.iloc[i][node] for node in
                ground_truth_noise_coefficients}
        ground_truth_rankings.append([k for k, v in sorted(tmp.items(), key=lambda item: -item[1])])
        ground_truth_scores.append([0] * len(nodes_order))
        for q in range(max_k):
            ground_truth_scores[i][nodes_order.index(ground_truth_rankings[i][q])] = overall_max_k - q
    return ground_truth_scores


def get_edge_coefficient(causal_dag, target_node):
    "Similar to get_noise_coefficient but the target_node is the arrow begin"
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

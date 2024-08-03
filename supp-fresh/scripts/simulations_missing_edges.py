import pickle
import random
from copy import deepcopy

import networkx
import numpy
import pandas
from tqdm import tqdm

from rca_helper import summarize_result, random_linear_dag_generator, get_noise_coefficient, evaluate_results_ndcg, \
    naive_approach, our_approach_rankings, draw_anomaly_samples
from dowhy.gcm import fit, is_root_node, EmpiricalDistribution, AdditiveNoiseModel, draw_samples
from dowhy.gcm._noise import get_noise_dependent_function
from dowhy.gcm.config import disable_progress_bars
from dowhy.gcm.ml import create_linear_regressor


# Mostly copied code from the other experiment.
def run_experiment():
    numpy.random.seed(0)

    disable_progress_bars()
    overall_max_k = 5

    result_our = {}
    result_naive = {}

    for percent in range(0, 100, 10):
        result_our[percent] = {}
        result_naive[percent] = {}
        for k in range(1, 10 + 1):
            result_our[percent][k] = []
            result_naive[percent][k] = []

    for run in tqdm(range(100)):
        max_k = int(numpy.random.choice(overall_max_k, 1)) + 1
        is_sufficiently_deep_graph = False

        while not is_sufficiently_deep_graph:
            # Generate DAG with random number of nodes and root nodes
            num_total_nodes = numpy.random.randint(20, 30)
            num_root_nodes = numpy.random.randint(1, 5)

            ground_truth_dag = random_linear_dag_generator(num_root_nodes,
                                                           num_downstream_nodes=num_total_nodes - num_root_nodes)

            # Make sure that the randomly generated DAG is deep enough.
            for node in random.sample(list(ground_truth_dag.graph.nodes), len(list(ground_truth_dag.graph.nodes))):
                target_node = node
                if len(list(networkx.ancestors(ground_truth_dag.graph, target_node))) + 1 >= 10:
                    is_sufficiently_deep_graph = True
                    break

        ground_truth_noise_coefficients = get_noise_coefficient(ground_truth_dag, target_node)
        nodes_order_ground_truth = list(ground_truth_noise_coefficients.keys())

        # Generate training samples
        training_data = draw_samples(ground_truth_dag, 2000)

        # Generate anomalous samples
        all_anomaly_samples = []
        all_noise_samples = []
        for i in range(10):
            anomaly_samples, noise_samples, _ = draw_anomaly_samples(ground_truth_dag, 1, max_k,
                                                                     nodes_order_ground_truth)

            all_anomaly_samples.append(anomaly_samples)
            all_noise_samples.append(noise_samples)

        all_anomaly_samples = pandas.concat(all_anomaly_samples)
        all_noise_samples = pandas.concat(all_noise_samples)

        # Get ground truth rankings
        ground_truth_rankings = []
        ground_truth_scores = []
        for i in range(all_noise_samples.shape[0]):
            tmp = {node: ground_truth_noise_coefficients[node] * all_noise_samples.iloc[i][node] for node in
                   ground_truth_noise_coefficients}
            ground_truth_rankings.append([k for k, v in sorted(tmp.items(), key=lambda item: -item[1])])
            ground_truth_scores.append([0] * len(nodes_order_ground_truth))
            for q in range(max_k):
                ground_truth_scores[i][nodes_order_ground_truth.index(ground_truth_rankings[i][q])] = overall_max_k - q

        # Run it for different percentages of missing edges.
        for percent in range(0, 100, 10):
            # Learn SCM from data
            learned_dag = deepcopy(ground_truth_dag)

            all_edges = list(learned_dag.graph.edges)
            to_remove = random.sample(all_edges, int(len(all_edges) * percent / 100))
            learned_dag.graph.remove_edges_from(to_remove)

            for node in learned_dag.graph.nodes:
                if is_root_node(learned_dag.graph, node):
                    learned_dag.set_causal_mechanism(node, EmpiricalDistribution())
                else:
                    learned_dag.set_causal_mechanism(node, AdditiveNoiseModel(create_linear_regressor()))

            fit(learned_dag, training_data)

            # Get the noise dependent model for the target node, i.e. Y = f(N0, ..., Nk).
            pred_method, nodes_order = get_noise_dependent_function(learned_dag, target_node)

            # Our approach
            contributions = our_approach_rankings(learned_dag,
                                                  target_node,
                                                  all_anomaly_samples,
                                                  target_prediction_method=pred_method,
                                                  nodes_order=nodes_order)

            rankings_our = []
            scores_our = []
            for tmp in contributions:
                rankings_our.append([k for k, v in sorted(tmp.items(), key=lambda item: -item[1])])
                scores_our.append([0] * len(nodes_order_ground_truth))
                for q in range(min(max_k, len(rankings_our[0]))):
                    tmp_node = rankings_our[-1][q]
                    tmp_index = nodes_order_ground_truth.index(tmp_node)
                    scores_our[-1][tmp_index] = overall_max_k - q

            # Naive approach
            contributions = naive_approach(learned_dag, nodes_order, all_anomaly_samples)
            rankings_naive = []
            scores_naive = []
            for tmp in contributions:
                rankings_naive.append([k for k, v in sorted(tmp.items(), key=lambda item: -item[1])])
                scores_naive.append([0] * len(nodes_order_ground_truth))
                for q in range(min(max_k, len(rankings_naive[0]))):
                    tmp_node = rankings_naive[-1][q]
                    tmp_index = nodes_order_ground_truth.index(tmp_node)
                    scores_naive[-1][tmp_index] = overall_max_k - q

            our_result = evaluate_results_ndcg(scores_our, ground_truth_scores, 10)
            naive_result = evaluate_results_ndcg(scores_naive, ground_truth_scores, 10)

            for k in range(1, 10 + 1):
                result_our[percent][k].extend(our_result[k])
                result_naive[percent][k].extend(naive_result[k])

    pickle.dump(result_our, open("OurResultMissing.p", "wb"))
    pickle.dump(result_naive, open("NaiveResultMissing.p", "wb"))


def summarize_results():
    result_our = pickle.load(open("OurResultMissing.p", "rb"))
    result_naive = pickle.load(open("NaiveResultMissing.p", "rb"))

    for percent in range(0, 100, 10):
        result = numpy.zeros((10, 5))
        result[:, 0] = numpy.arange(1, 11)

        mean, std = summarize_result(result_our[percent])
        result[:, 1] = mean
        result[:, 2] = std

        mean, std = summarize_result(result_naive[percent])
        result[:, 3] = mean
        result[:, 4] = std

        final_df = pandas.DataFrame(result, columns=['k', 'Our - Mean', 'Our - Std', 'Naive - Mean', 'Naive - Std'])

        final_df.to_csv(str(percent) + "PercentMissingEdgesResults.csv", index=False)

    for k in range(10):
        result = numpy.zeros((10, 5))
        result[:, 0] = numpy.arange(0, 100, 10)
        for percent in range(10):
            mean, std = summarize_result(result_our[percent * 10])
            result[percent, 1] = mean[k]
            result[percent, 2] = std[k]

            mean, std = summarize_result(result_naive[percent * 10])
            result[percent, 3] = mean[k]
            result[percent, 4] = std[k]

        final_df = pandas.DataFrame(result,
                                    columns=['Percentage', 'Our - Mean', 'Our - Std', 'Naive - Mean', 'Naive - Std'])

        final_df.to_csv("k=" + str(k) + "WithMissingEdgesResults.csv", index=False)


run_experiment()
summarize_results()

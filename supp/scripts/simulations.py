import pickle
import random
from copy import deepcopy

import networkx
import numpy
import pandas
from tqdm import tqdm

from rca_helper import random_linear_dag_generator, draw_anomaly_samples, our_approach_rankings, naive_approach, \
    MySquaredRegressor, get_noise_coefficient, evaluate_results_ndcg, summarize_result
from dowhy.gcm import draw_samples, is_root_node, EmpiricalDistribution, AdditiveNoiseModel, fit
from dowhy.gcm._noise import get_noise_dependent_function
from dowhy.gcm.config import disable_progress_bars
from dowhy.gcm.ml import create_linear_regressor

def run_experiment():
    numpy.random.seed(0)

    misspecified_fcm = False
    disable_progress_bars()
    overall_max_k = 5

    result_our = {}
    result_naive = {}

    for k in range(1, 10 + 1):
        result_our[k] = []
        result_naive[k] = []

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

        # Drawing samples from the DAG
        training_data = draw_samples(ground_truth_dag, 2000)

        # Learn causal models from scratch, given the graph structure.
        learned_dag = deepcopy(ground_truth_dag)
        for node in learned_dag.graph.nodes:
            if is_root_node(learned_dag.graph, node):
                learned_dag.set_causal_mechanism(node, EmpiricalDistribution())
            else:
                if misspecified_fcm:
                    # Purposely assigning a wrong model class.
                    learned_dag.set_causal_mechanism(node, AdditiveNoiseModel(MySquaredRegressor()))
                else:
                    learned_dag.set_causal_mechanism(node, AdditiveNoiseModel(create_linear_regressor()))

        # Fit causal mechanisms..
        fit(learned_dag, training_data)

        # Get the noise dependent model for the target node, i.e. Y = f(N0, ..., Nk).
        if not misspecified_fcm:
            pred_method, nodes_order = get_noise_dependent_function(learned_dag, target_node)
        else:
            pred_method, nodes_order = get_noise_dependent_function(learned_dag,
                                                                    target_node,
                                                                    approx_prediction_model=MySquaredRegressor())

        # Generate anomalous samples
        all_anomaly_samples = []
        all_noise_samples = []
        for i in range(10):
            anomaly_samples, noise_samples, _ = draw_anomaly_samples(ground_truth_dag, 1, max_k, nodes_order)

            all_anomaly_samples.append(anomaly_samples)
            all_noise_samples.append(noise_samples)

        all_anomaly_samples = pandas.concat(all_anomaly_samples)
        all_noise_samples = pandas.concat(all_noise_samples)

        # Get ground truth rankings
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

        # Ranking using our approach
        contributions = our_approach_rankings(learned_dag,
                                              target_node,
                                              all_anomaly_samples,
                                              target_prediction_method=pred_method,
                                              nodes_order=nodes_order)

        rankings_our = []
        scores_our = []
        for tmp in contributions:
            rankings_our.append([k for k, v in sorted(tmp.items(), key=lambda item: -item[1])])
            scores_our.append([0] * len(nodes_order))
            for q in range(max_k):
                tmp_node = rankings_our[-1][q]
                tmp_index = nodes_order.index(tmp_node)
                scores_our[-1][tmp_index] = overall_max_k - q

        # Ranking using the naive approach
        contributions = naive_approach(learned_dag, nodes_order, all_anomaly_samples)
        rankings_naive = []
        scores_naive = []
        for tmp in contributions:
            rankings_naive.append([k for k, v in sorted(tmp.items(), key=lambda item: -item[1])])
            scores_naive.append([0] * len(nodes_order))
            for q in range(max_k):
                tmp_node = rankings_naive[-1][q]
                tmp_index = nodes_order.index(tmp_node)
                scores_naive[-1][tmp_index] = overall_max_k - q

        our_result = evaluate_results_ndcg(scores_our, ground_truth_scores, 10)
        naive_result = evaluate_results_ndcg(scores_naive, ground_truth_scores, 10)

        for k in range(1, 10 + 1):
            result_our[k].extend(our_result[k])
            result_naive[k].extend(naive_result[k])

    pickle.dump(result_our, open("OurResult.p", "wb"))
    pickle.dump(result_naive, open("NaiveResult.p", "wb"))


def summarize_results():
    result_our = pickle.load(open("OurResult.p", "rb"))
    result_naive = pickle.load(open("NaiveResult.p", "rb"))

    result = numpy.zeros((10, 5))
    result[:, 0] = numpy.arange(1, 11)

    mean, std = summarize_result(result_our)
    result[:, 1] = mean
    result[:, 2] = std

    mean, std = summarize_result(result_naive)
    result[:, 3] = mean
    result[:, 4] = std

    final_df = pandas.DataFrame(result, columns=['k', 'Our - Mean', 'Our - Std', 'Naive - Mean', 'Naive - Std'])

    final_df.to_csv("Results.csv", index=False)


run_experiment()
summarize_results()

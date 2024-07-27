import pickle

import networkx
import numpy
import pandas

from dowhy.gcm import InvertibleStructuralCausalModel, EmpiricalDistribution, AdditiveNoiseModel, fit, \
    MeanDeviationScorer, ITAnomalyScorer, attribute_anomalies
from dowhy.gcm.anomaly import conditional_anomaly_score
from dowhy.gcm.graph import get_ordered_predecessors
from dowhy.gcm.ml import create_linear_regressor_with_given_parameters

variables = ['hodder_place', 'henthorn', 'new_jumbles_rock', 'whalley_weir']

data = pandas.read_csv("river.csv")

training_data = data[variables].iloc[:3287]
training_data = training_data.dropna()
test_data = data.iloc[3287:]

dag = InvertibleStructuralCausalModel(networkx.DiGraph(
    [('hodder_place', 'new_jumbles_rock'), ('henthorn', 'new_jumbles_rock'), ('whalley_weir', 'new_jumbles_rock')]))
dag.set_causal_mechanism('hodder_place', EmpiricalDistribution())
dag.set_causal_mechanism('henthorn', EmpiricalDistribution())
dag.set_causal_mechanism('whalley_weir', EmpiricalDistribution())
dag.set_causal_mechanism('new_jumbles_rock',
                         AdditiveNoiseModel(create_linear_regressor_with_given_parameters(numpy.array([1, 1, 1]),
                                                                                          intercept=0)))
fit(dag, training_data)

results = {}

for v in variables:
    results[v] = {}
    results[v]['z-score'] = {}
    results[v]['it-z-score'] = {}

    anomaly_scorer = MeanDeviationScorer()
    anomaly_scorer_it = ITAnomalyScorer(MeanDeviationScorer())
    anomaly_scorer.fit(training_data[v].to_numpy())
    anomaly_scorer_it.fit(training_data[v].to_numpy())

    z_scores = anomaly_scorer.score(test_data[v].to_numpy())
    it_z_scores = anomaly_scorer_it.score(test_data[v].to_numpy())

    for i in range(test_data.shape[0]):
        results[v]['z-score'][test_data['date'].iloc[i]] = z_scores[i]
        results[v]['it-z-score'][test_data['date'].iloc[i]] = it_z_scores[i]

results['new_jumbles_rock']['conditional-score'] = {}
results['new_jumbles_rock']['it-conditional-score'] = {}
results['new_jumbles_rock']['contributions'] = {}

tmp_anomaly_parent_samples = test_data[get_ordered_predecessors(dag.graph, 'new_jumbles_rock')].to_numpy()
tmp_anomaly_target_samples = test_data['new_jumbles_rock'].to_numpy()

contributions = attribute_anomalies(dag, 'new_jumbles_rock', test_data, anomaly_scorer=MeanDeviationScorer())
for i in range(test_data.shape[0]):
    results['new_jumbles_rock']['conditional-score'][
        test_data['date'].iloc[i]] = conditional_anomaly_score(
        tmp_anomaly_parent_samples[i].reshape(1, -1),
        tmp_anomaly_target_samples[i].reshape(1, 1),
        dag.causal_mechanism('new_jumbles_rock'),
        MeanDeviationScorer,
        10000)

    results['new_jumbles_rock']['it-conditional-score'][
        test_data['date'].iloc[i]] = conditional_anomaly_score(
        tmp_anomaly_parent_samples[i].reshape(1, -1),
        tmp_anomaly_target_samples[i].reshape(1, 1),
        dag.causal_mechanism('new_jumbles_rock'),
        lambda: ITAnomalyScorer(MeanDeviationScorer()),
        10000)

    results['new_jumbles_rock']['contributions'][test_data['date'].iloc[i]] = {}
    for v in variables:
        results['new_jumbles_rock']['contributions'][test_data['date'].iloc[i]][v] = contributions[v][i]

pickle.dump(results, open("RiverResults.p", "wb"))
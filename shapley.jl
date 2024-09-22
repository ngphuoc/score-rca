
function predict_leaf()

end

function scorer()

end

function shapley_method()

end

function compute_noise_from_data(causal_dag, anomaly_samples)

end

function anomaly_score_attributions()

end

function our_approach_rankings(causal_dag, target_node, anomaly_samples, target_prediction_method, nodes_order, zscorer, ref_samples, approximation_method)
    noise_of_anomaly_samples = compute_noise_from_data(causal_dag, anomaly_samples)
    # node_samples, noise_samples = noise_samples_of_ancestors(causal_dag, target_node, 1000)
    # scorer = MeanDeviationScorer()
    # scorer.fit(node_samples[target_node].to_numpy())
    scorer =  zscorer
    noise_samples = ref_samples
    attributions = anomaly_score_attributions(noise_of_anomaly_samples, noise_samples, scorer ∘ target_prediction_method)
    result = []
    for i in 1:size(attributions[1])
        tmp = Dict()
        for j in 1:size(attributions[1])
            tmp[nodes_order[j]] = attributions[i, j]
        end
        push!(result, tmp)
    end

    return result
end


contributions = our_approach_rankings(causal_dag, target_node, anomaly_samples, target_prediction_method, nodes_order, zscorer, ref_samples, approximation_method)



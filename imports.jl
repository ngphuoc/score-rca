using PythonCall
using Parameters: @unpack
include("lib/utils.jl")
include("lib/graph.jl")
include("lib/outlier.jl")
pyimport("sys").path.insert(0, "")
pyimport("sys").path.insert(0, "supp/scripts")
pyimport("sys").path

@py import pickle
@py import random
@py import networkx
@py import numpy
@py import pandas
@py import pybaseball as pyb
@py import scipy
@py import dowhy
@py import sklearn
@unpack stats = scipy
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")
tqdm = pyimport("tqdm").tqdm
pydeepcopy = pyimport("copy").deepcopy

@unpack random_nonlinear_dag_generator, random_linear_dag_generator, draw_anomaly_samples, draw_samples_2, our_approach_rankings, naive_approach, MySquaredRegressor, get_noise_coefficient, evaluate_results_ndcg, summarize_result, get_ground_truth_rankings, ZOutlierScorePy, ShapleyApproximationMethods, pickle_save, pickle_load = pyimport("rca_helper")
@unpack draw_samples, is_root_node, EmpiricalDistribution, InvertibleStructuralCausalModel, ScipyDistribution, AdditiveNoiseModel = dowhy.gcm
@unpack gcm = dowhy
@unpack disable_progress_bars = dowhy.gcm.config
@unpack create_linear_regressor = dowhy.gcm.ml
@unpack get_ordered_predecessors = dowhy.gcm.graph
@unpack get_noise_dependent_function = dowhy.gcm._noise

int(x::Py) = pyconvert(Int, x)
Base.float(x::Py) = pyconvert(Float64, x)

df2pd(df) = pandas.DataFrame(Base.eachrow(df), columns=names(df))

pd2df(pd) = DataFrame(PyTable(pd))

function evaluate_results(W, Ŵ)
    m = size(W, 1)
    ci = @> map(1:m) do i
        map(i+1:m) do j
            CartesianIndex(i, j)
        end
    end vcats
    W = (W + W')[ci]
    Ŵ = (Ŵ + Ŵ')[ci]
    roc_auc_score(W, Ŵ) |> float
end

function get_upper(W)
    m = size(W, 1)
    ci = @> map(1:m) do i
        map(i+1:m) do j
            CartesianIndex(i, j)
        end
    end vcats
    W = (W + W')[ci]
end

function evaluate_results_n(node_truths, node_scores)
    rn = roc_auc_score(node_truths, node_scores)
end

function evaluate_results_e(node_truths, edge_truths, node_scores, edge_scores)
    e_truth = get_upper(edge_truths)
    e_score = get_upper(edge_scores)
    re = roc_auc_score(e_truth, e_score)
    rn = roc_auc_score(node_truths, node_scores)
    r = roc_auc_score([node_truths..., e_truth...], [node_scores..., e_score...])
    @> [r, re, rn, (re + rn)/2] float.() round.(digits=2)
end

function evaluate_results_ne(node_truths, edge_truths, node_scores, edge_scores)
    e_truth = get_upper(edge_truths)
    e_score = get_upper(edge_scores)
    re = roc_auc_score(e_truth, e_score)
    rn = roc_auc_score(node_truths, node_scores)
    r = roc_auc_score([node_truths..., e_truth...], [node_scores..., e_score...])
    @> [r, re, rn, (re + rn)/2] float.() round.(digits=2)
end

function evaluate_results_2(node_truths, edge_truths, node_scores, edge_scores)
    e_truth = get_upper(edge_truths)
    e_score = get_upper(edge_scores)
    re = roc_auc_score(e_truth, e_score)
    rn = roc_auc_score(node_truths, node_scores)
    r = roc_auc_score([node_truths..., e_truth...], [node_scores..., e_score...])
    @> [r, re, rn, (re + rn)/2] float.() round.(digits=2)
end

function evaluate_results_3(node_truths, edge_truths, node_scores, edge_scores)
    re = roc_auc_score(edge_truths, edge_scores)
    rn = roc_auc_score(node_truths, node_scores)
    r = roc_auc_score([node_truths..., edge_truths...], [node_scores..., edge_scores...])
    @> [r, re, rn, (re + rn)/2] float.() round.(digits=2)
end

function dict_namedtuple(adict::PyDict)
    (; (Symbol(k) => v for (k, v) in adict)...)
end

function dict_namedtuple(adict::Dict)
    (; (Symbol(k) => v for (k, v) in adict)...)
end


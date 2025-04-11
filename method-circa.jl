@info "Step1: fit linear bayesnet"

@≥ z, x, l, s, z3, x3, l3, s3, za, xa, la, sa cpu.()
d = size(z, 1)

bn = copy_linear_dag(g)
bn.cpds = cpu.(bn.cpds)
Distributions.fit!(bn, x)
g = bn

@info "Step2: define residual function as outlier scores"

function get_residual(bn, x)
    μx = forward_1step(bn, x)
    x - μx
end

ẑx = get_residual(bn, x)
vx = abs.(ẑx)  # anomaly_measure

@info "Step3: Get ground truth rankings"

max_k = args.n_anomaly_nodes
overall_max_k = max_k + 1

function get_gt_z_ranking(za, ∇za)
    @assert size(za, 1) == d
    i = 1
    scores = Vector{Float64}[]  # 1 score vector for each outlier
    batchsize = size(za, 2)
    for i = 1:batchsize
        tmp = Dict(j => ∇za[j, i] * za[j, i] for j = 1:d)
        ranking = [k for (k, v) in sort(tmp, byvalue=true, rev=true)]   # used
        score = zeros(d)
        for q in 1:max_k
            iq = findfirst(==(ranking[q]), 1:d)
            score[iq] = overall_max_k - q
        end
        push!(scores, score)
    end
    @≥ scores hcats
    return scores
end

∇za, = Zygote.gradient(za, ) do za
    @> forward_leaf(g, za, ii) sum
end
gt_ranking = get_gt_z_ranking(za, ∇za)
@> gt_ranking mean(dims=2)
gt_manual = indexin(1:d, anomaly_nodes) .!= nothing
gt_manual = repeat(gt_manual, outer=(1, size(xa, 2)))

@info "Step5: Outlier ndcg ranking"

μa = forward_1step(g, xa)
ẑa = xa - μa
anomaly_measure = abs.(get_residual(bn, xa))  # anomaly_measure
anomaly_measure_full = anomaly_measure_half = anomaly_measure

using PythonCall
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")

@info "Step6: Save results"

k = 2
res = map(1:d) do k
    results = (;
                 map(round3 ∘ only ∘ PyArray, (;
                     ranking_full = ndcg_score(gt_ranking', anomaly_measure_full'; k),
                     manual_full = ndcg_score(gt_manual', anomaly_measure_full'; k),
                     ranking_half = ndcg_score(gt_ranking', anomaly_measure_half'; k),
                     manual_half = ndcg_score(gt_manual', anomaly_measure_half'; k),
                 ))...,
                 k, args.data_id, args.n_nodes, args.n_anomaly_nodes, method = "CIRCA", fpath,
              )
end
@> DataFrame(res) println

append!(results, res)
nothing


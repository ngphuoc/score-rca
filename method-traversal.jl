@info "Step1: fit linear bayesnet"

@≥ z, x, l, s, z3, x3, l3, s3, za, xa, la, sa cpu.()
d = size(z, 1)

bn = copy_linear_dag(g)
bn.cpds = cpu.(bn.cpds)
Distributions.fit!(bn, x)
g = bn

@info "Step2: define residual function as outlier scores"

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

function zscore(x, r)
    μ, σ = @> r mean(dims=2), std(dims=2)
    @. (x - μ) / σ
end

""" like matmul but reduce by maximum/minimum/f instead of sum
"""
function fmul(f, A::AbstractMatrix{Bool}, x1)
    d = size(A, 1)
    y = similar(x1)
    for j = 1:d
        a = A[j, :]
        z = (a .* x1)[a]
        y[j] = length(z) > 0 ? f(z) : 0f0
    end
    y
end

maxmul(A, x1) = fmul(maximum, A, x1)
minmul(A, x1) = fmul(minimum, A, x1)

""" Subgraph connecting node j and node l"""
function path_graph(dag, j, l)
    g = DiGraph(nv(dag))
    for e = a_star(dag, j, l)
        add_edge!(g, e.src, e.dst);
    end
    g
end

function traversal_measure(g, xa, x)
    d = nv(g.dag)
    v = Distributions.cdf(Normal(0, 1), -abs.(zscore(xa, x)))
    B = @> adjacency_matrix(g.dag) Matrix{Bool}
    A = @> I(d) + B Matrix{Bool}
    x1 = v[:, 2]
    ys = []
    for x1 = eachcol(v)
        # traversal algorithm - no abnormal before j, all abnormal after j: minscore(>=j) - maxscore(<j)
        y = similar(x1)
        j = 2
        @≥ x1 Array
        for j = 1:d
            a1, b1 = @> copy(x1), copy(x1) Array{Float32}.()
            a1[j] = 0
            as = [a1]  # before j
            # no abnormal before j: propagate max from root to j
            for i = 1:j-1
            # for i = 1:1
                xj = copy(as[end])
                push!(as, maxmul(A', xj))
            end

            # all abnormal after j: propagate min from j to leaf d, path at most length j+1:d
            P = @> I(d) + adjacency_matrix(path_graph(g.dag, j, d)) Matrix{Bool}
            bs = [b1]  # after j
            for i = j+1:d
            # for i = 1:1
                push!(bs, minmul(P', bs[end]))
            end
            # minscore(>=j) - maxscore(<j)
            y[j] = bs[end][d] - as[end][j]
        end
        hcat(x1, y)
        A
        push!(ys, y)
    end
    hcats(ys)
end

function naive_measure(g, xa, x)
    d = nv(g.dag)
    v = Distributions.cdf(Normal(0, 1), -abs.(zscore(xa, x)))
end

anomaly_measure = traversal_measure(g, xa, x)
anomaly_measure_full = anomaly_measure_half = anomaly_measure

using PythonCall
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")

@info "Step6: Save results traversal measure"

k = 2
res = map(1:d) do k
    results = (;
                 map(round3 ∘ only ∘ PyArray, (;
                     ranking_full = ndcg_score(gt_ranking', anomaly_measure_full'; k),
                     manual_full = ndcg_score(gt_manual', anomaly_measure_full'; k),
                     ranking_half = ndcg_score(gt_ranking', anomaly_measure_half'; k),
                     manual_half = ndcg_score(gt_manual', anomaly_measure_half'; k),
                 ))...,
                 k, args.data_id, args.n_nodes, args.n_anomaly_nodes, method = "Traversal", fpath,
              )
end
@> DataFrame(res) println

append!(results, res)

@info "Step:7: Save results naive measure"

anomaly_measure = naive_measure(g, xa, x)
anomaly_measure_full = anomaly_measure_half = anomaly_measure

k = 2
res = map(1:d) do k
    results = (;
                 map(round3 ∘ only ∘ PyArray, (;
                     ranking_full = ndcg_score(gt_ranking', anomaly_measure_full'; k),
                     manual_full = ndcg_score(gt_manual', anomaly_measure_full'; k),
                     ranking_half = ndcg_score(gt_ranking', anomaly_measure_half'; k),
                     manual_half = ndcg_score(gt_manual', anomaly_measure_half'; k),
                 ))...,
                 k, args.data_id, args.n_nodes, args.n_anomaly_nodes, method = "Naive", fpath,
              )
end
@> DataFrame(res) println

append!(results, res)

nothing


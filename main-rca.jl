using Revise
using Flux
include("./data-rca.jl")
include("./lib/utils.jl")
include("./score-matching.jl")
include("./plot-dsm.jl")

g, ε, x, ε′, x′, εa, xa = get_data(; args)

#-- normalise data
X = @> hcat(x, x′);
μX, σX = @> X mean(dims=2), std(dims=2);
normalise(x) = @. (x - μX) / σX
@≥ ε, x, ε′, x′, εa, xa, X normalise.();

# include("./attribution.jl")
function get_ref(x, r)
    dist_xr = pairwise(Euclidean(), x, r)  # correct
    _, j = @> findmin(dist_xr, dims=2)
    @≥ j getindex.(2) vec
    r[:, j]
end

function get_score(dnet, x)
    t = fill!(similar(x, size(x)[end]), 0.01) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    # σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
    @> dnet(x, t)
end

r = x
# r = get_ref(xa, x)
# pairwise(Euclidean(), xa, r)

@≥ x, r gpu.();
inputdim = size(x, 1)
dnet = @> DSM(inputdim; args) gpu
dnet = train_dsm(dnet, x; args)

dx = get_score(dnet, x)
dr = get_score(dnet, r)
# anomaly_measure = norm2(dx) .* norm2(r - x)
# norm2(dr)
anomaly_measure = abs.(dx)

max_k = n_anomaly_nodes
overall_max_k = max_k + 1

g
adjmat = @> g.dag adjacency_matrix Matrix{Bool}
d = size(adjmat, 1)
ii = @> adjmat eachcol findall.()
∇εa, = Zygote.gradient(εa, ) do εa
    @> forward_leaf(g, εa, ii) sum
end

function get_ε_rankings(εa, ∇εa)
    @assert size(εa, 1) == d
    i = 1
    scores = Vector{Float64}[]  # 1 score vector for each outlier
    batchsize = size(εa, 2)
    for i = 1:batchsize
        tmp = Dict(j => ∇εa[j, i] * εa[j, i] for j = 1:d)
        ranking = [k for (k, v) in sort(tmp, byvalue=true, rev=true)]  # used
        score = zeros(d)
        for q in 1:max_k
            iq = findfirst(==(ranking[q]), 1:d)
            score[iq] = overall_max_k - q
        end
        push!(scores, score)
    end
    return scores
end

@≥ x, xa cpu.();
xr = get_ref(xa, x);
εr
gt_value = @> get_ε_rankings(εa, ∇εa) hcats

μa = forward_1step_mean(g, xa, ii)
ε̂a = xa - μa
μr = forward_1step_mean(g, xr, ii)
ε̂r = xr - μr
∇xa = @> get_score(dnet, gpu(xa)) cpu
anomaly_measure = abs.(∇xa)
dr = get_score(dnet, r)
ndcg_score(gt_value', abs.((ε̂a - ε̂r) .* ∇xa)', k=n_anomaly_nodes)


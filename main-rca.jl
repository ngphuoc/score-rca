include("./data-rca.jl")
include("./score-matching.jl")
include("./plot-dsm.jl")

n_nodes = round(Int, min_depth^2 / 3 + 1)
n_root_nodes = 1
n_downstream_nodes = n_nodes - n_root_nodes
scale, hidden = 0.5, 100
g = random_mlp_dag_generator(n_root_nodes, n_downstream_nodes, scale, hidden)

ε, x, ε′, x′, εa, xa = draw_normal_perturbed_anomaly(g; args)

#-- normalise data
X = @> hcat(x, x′);
μX, σX = @> X mean(dims=2), std(dims=2);
normalise(x) = @. (x - μX) / σX
@≥ ε, x, ε′, x′, εa, xa normalise.();

# include("./attribution.jl")
function get_ref(x, r)
    dxr = pairwise(Euclidean(), x, r)
    _, j = @> findmin(dxr, dims=2)
    @≥ j getindex.(2) vec
    n = r[:, j]
end

function get_score(dnet, x)
    t = fill!(similar(x, size(x)[end]), 0.01) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
    J = @> dnet(x, t)
end

r = get_ref(xa, x)

@≥ x Array gpu;
@≥ r Array gpu;
@≥ dnet gpu;
dx = get_score(dnet, x)
dr = get_score(dnet, r)
# anomaly_measure = norm2(dx) .* norm2(r - x)
# norm2(dr)
anomaly_measure = abs.(dx)

max_k = n_outliers
overall_max_k = max_k + 1

function get_ϵ_rankings()
    ws = get_noise_coefficient(ground_truth_dag, target_node) |> PyDict
    ϵ = infer_ϵ(bn, Xo)
    j = 1
    scores = Vector{Float64}[]
    for j = 1:size(ϵ, 1)
        tmp = Dict(node => w * ϵ[j, i_(node)] for (node, w) in ws)
        ranking = [k for (k, v) in sort(tmp, byvalue=true, rev=true)]
        score = zeros(length(sub_nodes))
        for q in 1:max_k
            iq = findfirst(==(Symbol(ranking[q])), sub_nodes)
            score[iq] = overall_max_k - q
        end
        push!(scores, score)
    end
    return scores
end

gt_value = @> get_ϵ_rankings() transpose.() vcats
ndcg_score(gt_value', ((ξ .- ξ0) .* ξ_value)', k=n_outliers)


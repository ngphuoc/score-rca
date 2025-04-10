
@info "Step4: Integrate scores along the trajectories and multiply by the distance \frac{1}{2}(x_{j}-x'_{j}) for each j using Eq. 16"

# TODO: recheck gt ranking scores
∇za, = Zygote.gradient(za, ) do za
    @> forward_leaf(g, za, ii) sum
end
gt_value = @> get_z_rankings(za, ∇za) hcats
@> gt_value mean(dims=2)
anomaly_nodes

@≥ ẑa, ẑr to_device
∇xa = @> get_scores(diffusion_model, ẑa)
anomaly_measure = abs.(∇xa)

using PythonCall
@unpack ndcg_score, classification_report, roc_auc_score, r2_score = pyimport("sklearn.metrics")

gt_manual = indexin(1:d, anomaly_nodes) .!= nothing
gt_manual = repeat(gt_manual, outer=(1, size(xa, 2)))
ndcg_score(gt_manual', abs.((ẑa - ẑr) .* ∇xa)', k=args.n_anomaly_nodes)

df = DataFrame(
               n_nodes = Int[],
               n_anomaly_nodes = Int[],
               method = String[],
               noise_dist  = String[],
               data_id = Int[],
               ndcg_ranking = Float64[],
               ndcg_manual = Float64[],
               k = Int[],
              )

k = 1
anomaly_measure = abs.((ẑa - ẑr) .* ∇xa)'
# a = anomaly_measure
# @≥ a reshape(:, 5, 4)
# i = @> a std(dims=3) squeeze(3) argmax(dims=2) vec
# a[i, :]
for k=1:args.min_depth
    ndcg_ranking = ndcg_score(gt_value', anomaly_measure; k)
    ndcg_manual = ndcg_score(gt_manual', anomaly_measure; k)
    @≥ ndcg_ranking, ndcg_manual PyArray.() only.()
    push!(df, [args.n_nodes, args.n_anomaly_nodes, "DSM", string(args.noise_dist), args.data_id, ndcg_ranking, ndcg_manual, k])
end

println(df);

fname = "results/random-graphs.csv"
# CSV.write(fname, df, header=!isfile(fname), append=true)


function get_sensitivities_minibatch(g, zs, ii)
    ∇fs = similar(zs)
    size_zs = size(zs)
    for t = 1:size(zs)[end]
        ∇f, = Zygote.gradient(gpu(zs[:, :, t], )) do z
            @> forward_leaf(g, z, ii) sum
        end;
        ∇fs[:, :, t] .= cpu(∇f)
    end
    ∇fs
end


using BenchmarkTools

# Return jacobian w.r.t. input
function get_sensitivities(g, zs, ii)
    zs_gpu = gpu(reshape(zs, size(zs, 1), :))
    ys, ss = forward(g, zs_gpu, ii)
    ys, back = Zygote.pullback(zs_gpu) do zs_gpu
        ys, ss = forward(g, zs_gpu, ii)
        ys
    end
    J = zeros(Float32, (d, d, size(zs_gpu)[end]))
    for i = 1:d
        seed = zeros_like(ys)
        seed[i, :] .= 1f0
        b1 = back(seed)[1]  # back(seed) returns a tuple, @show i, sum(b1)
        J[i, :, :] .= @> b1 cpu
    end
    J
end

J = get_sensitivities(g, zs, ii);
J[end, :, :] ≈ ∇f

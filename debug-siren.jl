using Flux, CUDA, MLUtils, PyPlot
include("data-rca.jl")
include("lib/utils.jl")
include("dsm-model.jl")

"""
n_samples = 5
fname = "fig/paths.png"
"""
function plot_path_zx(zs, xs, n_samples = 5, fname="fig/paths.png")
    M, N, T, = size(zs)
    @show M, N, T
    fig, axs = subplots(M, 2, figsize=(16, 4*M))
    # Plot original histograms
    i = j = 1
    vars = [zs, xs]
    for k in 1:2
        v = vars[k]
        for i in 1:M
            @show i, k
            ax = axs[i, k]
            for j in 1:n_samples
                ax.plot(1:T, v[i, j, :])
            end
            ax.set_title("$(k % 2 == 1 ? "Noise" : "Observation") $i")
        end
    end
    tight_layout()
    savefig(fname)
    println("Saved fig to file $fname")
end

forwardg(x) = forward(g, x)
xs = @> zs reshape(d, :) forwardg reshape(size(zs))
plot_path_zx(zs, xs)

function various_path_plots(xs, zs, ms, ∇zs, anomaly_nodes; args, n_path_samples = 10, fname)
    d, n, T = size(xs)
    n_rows = 9
    fig, axs = subplots(d, n_rows, figsize=(8*n_rows, 6*d))
    for i = 1:d
        for k = 1:n_rows
            plot_name, data = [("Diffusion Noise", zs), ("Diffusion Mean", ms), ("Diffusion Observation", xs),
                               ("Initial Noise", zs[i, :, 1]), ("Initial Mean", ms[i, :, 1]), ("Initial Observation", xs[i, :, 1]),
                               ("Final Noise", zs[i, :, end]), ("Final Mean", ms[i, :, end]), ("Final Observation", xs[i, :, end])][k]
            ax = axs[i, k]
            i in anomaly_nodes && k > 3 && (plot_name *= " Outlier")
            if k <= 3
                ax.set_title("$plot_name $i")
                for j = 1:n_path_samples
                    ax.plot(1:T, data[i, j, :])
                end
            else
                ax.set_title("$plot_name $i")
                ax.hist(data, bins=50, alpha=0.7)
            end
        end
    end
    tight_layout()
    fname = "fig/$fname-$(args.data_id).png"
    savefig(fname)
end

forwardg(x) = forward(g, x)

init_z = z
zs, ms, ∇zs = langevine_ref(diffusion_model, init_z, n_steps = 100) |> cpu
xs = @> zs reshape(d, :) forwardg reshape(size(zs))
fname = "generated_data_skewed_diffusion_normal"
various_path_plots(xs, zs, ms, ∇zs, anomaly_nodes; args, fname)

init_z = za
zs, ms, ∇zs = langevine_ref(diffusion_model, init_z, n_steps = 100) |> cpu
xs = @> zs reshape(d, :) forwardg reshape(size(zs))
fname = "generated_data_skewed_diffusion_OUTLIER"
various_path_plots(xs, zs, ms, ∇zs, anomaly_nodes; args, fname)

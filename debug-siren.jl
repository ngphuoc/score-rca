using Flux, CUDA, MLUtils, PyPlot
include("data-rca.jl")
include("lib/utils.jl")
include("dsm-model.jl")

function various_diffusion_plots(xs, ss, zs, ms, ∇zs, anomaly_nodes; args, n_path_samples = 10, pname)
    d, n, T = size(xs)
    n_rows = 3 * 4
    fig, axs = subplots(d, n_rows, figsize=(4*n_rows, 5*d))
    for i = 1:d
        for k = 1:n_rows
            plot_name, data = [("Diffusion Noise", zs), ("Diffusion Mean", ms), ("Diffusion Scale", ss), ("Diffusion Observation", xs),
                               ("Initial Noise", zs[i, :, 1]), ("Initial Mean", ms[i, :, 1]), ("Initial Scale", ss[i, :, 1]), ("Initial Observation", xs[i, :, 1]),
                               ("Final Noise", zs[i, :, end]), ("Final Mean", ms[i, :, end]), ("Final Scale", ss[i, :, end]), ("Final Observation", xs[i, :, end])][k]
            ax = axs[i, k]
            i in anomaly_nodes && k > 3 && (plot_name *= " Outlier")
            if k <= 4
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
    @info "Saving plots to $pname"
    savefig(pname)
end

forwardg(x) = forward(g, x)

init_z = z
zs, ms, ∇zs = langevine_ref(diffusion_model, init_z, n_steps = 100) |> cpu
xs, ss = @> zs reshape(d, :) forwardg reshape.(Ref(size(zs)))
pname = replace(fpath, ".bson" => "-diffusion-plots.png")
various_diffusion_plots(xs, ss, zs, ms, ∇zs, anomaly_nodes; args, pname)

# init_z = za
# zs, ms, ∇zs = langevine_ref(diffusion_model, init_z, n_steps = 100) |> cpu
# xs = @> zs reshape(d, :) forwardg reshape(size(zs))
# fname = "generated_data_skewed_diffusion_OUTLIER"
# various_diffusion_plots(xs, zs, ms, ∇zs, anomaly_nodes; args, fname)


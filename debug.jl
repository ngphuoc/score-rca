using PythonCall, PyPlot, Random, Distributions

# include("./plot-dsm.jl")

function test_plot()
    # Generate original data: 5 variables, 100 observations
    data = randn(100, 5)
    # Generate random noise with random means (variance = 1)
    noise_means = rand(Uniform(-3, 3), 5)
    noise = randn(100, 5) .+ noise_means'
    # Add noise to original data
    data_noisy = data .+ noise
    # Set up 2x5 subplot
    fig, axes = subplots(2, 5, figsize=(20, 8))
    # Plot original histograms
    for i in 1:5
        ax = axes[1, i]
        ax.hist(data[:, i], bins=20, alpha=0.7)
        ax.set_title("Original Var $i")
    end
    # Plot noisy histograms
    for i in 1:5
        ax = axes[2, i]
        ax.hist(data_noisy[:, i], bins=20, alpha=0.7, color="orange")
        ax.set_title("Noisy Var $i (μ=$(round(noise_means[i], digits=2)))")
    end
    # Adjust layout and save to file
    tight_layout()
    savefig("fig/histogram_grid.png")
end

function plot_zx(a, fname)
    w, h, = size(a)
    @show w, h
    fig, axs = subplots(h, w, figsize=(20, 16))
    # Plot original histograms
    i = j = 1
    for j in 1:h
        for i in 1:w
            @show i, j
            ax = axs[j, i]
            ax.hist(a[i, j, :], bins=20, alpha=0.7)
            ax.set_title("$(j % 2 == 1 ? "Noise" : "Observation") $i")
        end
    end
    tight_layout()
    savefig(fname)
    println("Saved fig to file $fname")
end

a = reshape(vcat(z[:, 1:500], x[:, 1:500], za, xa), d, 4, :)
fname = "fig/zx$d.png"
plot_zx(a, fname)

function get_score(dnet, x)
    t = fill!(similar(x, size(x)[end]), 0.01) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    # σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
    @> dnet(x, t)
end

dz = get_score(dnet, z)


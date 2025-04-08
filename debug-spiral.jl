using PythonCall, PyPlot, Random, Distributions

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

function various_plots(X_val)
    # Define plot limits
    xlims = (-15, 15)
    ylims = (-15, 15)

    # ---------------------------
    # 1. Plot Original Data
    # ---------------------------
    # Get a CPU copy of the data
    x_data = X_val[1, :]
    y_data = X_val[2, :]

    # Create a 2×2 grid figure
    fig, axs = subplots(2, 4, figsize=(10*4, 10*2))

    # First subplot: Original Data
    ax1 = axs[1, 1]
    ax1.set_title("Data (x, y)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    ax1.scatter(x_data, y_data, s=10, alpha=0.5)

    # ---------------------------
    # 2. Plot Perturbed Data
    # ---------------------------
    # Move X_val to GPU (if needed)
    x_gpu = X_val |> gpu
    # Generate random time t for each sample
    t = rand(Float32, size(x_gpu, 2)) .* (1f0 - 1f-5) .+ 1f-5  # vector of size N
    # Compute σ_t using your marginal probability function (assumed defined)
    σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1) |> gpu
    # Add noise (using same shape as x_gpu)
    z = randn(Float32, size(x_gpu)) |> gpu
    # Perturb data: x̃ = x + σ_t .* noise
    x_tilde = x_gpu .+ σ_t .* z
    # Bring perturbed data back to CPU
    x_tilde_cpu = cpu(x_tilde)
    x_data2 = x_tilde_cpu[1, :]
    y_data2 = x_tilde_cpu[2, :]

    # Second subplot: Perturbed Data
    ax2 = axs[1, 2]
    ax2.set_title("Perturbed score matching (x, y)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    ax2.scatter(x_data2, y_data2, s=10, alpha=0.5)

    # ---------------------------
    # 3. Plot Gradients
    # ---------------------------
    # Create a grid of points over the plot limits

    x_grid = range(xlims[1], stop=xlims[2], length=20)
    y_grid = range(ylims[1], stop=ylims[2], length=20)
    X_grid, Y_grid = meshgrid(x_grid, y_grid)
    # Flatten the grid into 2×(20*20) array (each column is a point)
    xy = vcat(vec(X_grid)', vec(Y_grid)')
    # Convert to Float32 and move to GPU
    xy_gpu = gpu(Float32.(xy))
    # Create a constant time vector (e.g. 0.001 for all grid points)
    t_grid = fill(0.001f0, size(xy_gpu, 2)) |> gpu
    # (Optional) Compute σ_t on the grid if needed:
    σ_t_grid = expand_dims(marginal_prob_std(t_grid; args.σ_max), 1)
    # Evaluate your diffusion model at these grid points (multiplied by a factor, e.g. 0.2)
    J = 0.2f0 .* diffusion_model(xy_gpu, t_grid) |> cpu
    # Extract grid point coordinates and gradient components
    x_grid = xy[1, :]
    y_grid = xy[2, :]
    u = J[1, :]  # gradient in x-direction
    v = J[2, :]  # gradient in y-direction

    # Third subplot: Gradients (using quiver)
    ax3 = axs[1, 3]
    ax3.set_title("Gradients")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_xlim(xlims)
    ax3.set_ylim(ylims)
    ax3.quiver(x_grid, y_grid, u, v)

    # ---------------------------
    # 4. Plot Stream Plot
    # ---------------------------
    # For a stream plot we need to reshape the gradients back to a grid shape.
    U = reshape(u, size(X_grid))
    V = reshape(v, size(Y_grid))
    ax4 = axs[1, 4]
    ax4.set_title("Stream Plot")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_xlim(xlims)
    ax4.set_ylim(ylims)
    ax4.streamplot(X_grid, Y_grid, U, V)

    # ---------------------------
    # 5. Plot generated samples from diffusion_model
    # ---------------------------

    init_x = args.σ_max .* randn_like(X_val) |> gpu
    xs, ms = langevine_ref(diffusion_model, init_x, n_steps = 100) |> cpu

    ax5 = axs[2, 1]
    ax5.set_title("Generated samples (x, y)")
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_xlim(xlims)
    ax5.set_ylim(ylims)
    ax5.scatter(xs[1, :, end], xs[2, :, end], s=10, alpha=0.5)

    # ---------------------------
    # 6. Plot diffusion paths of generated samples
    # ---------------------------

    ax6 = axs[2, 2]
    ax6.set_title("Diffusion paths x1")
    ax6.set_xlabel("t")
    ax6.set_ylabel("x")
    # ax6.set_xlim(xlims)
    # ax6.set_ylim(ylims)
    ax7 = axs[2, 3]
    ax7.set_title("Diffusion paths x2")
    ax7.set_xlabel("t")
    ax7.set_ylabel("x")
    # ax7.set_xlim(xlims)
    # ax7.set_ylim(ylims)
    ts = 1:size(xs, 3)
    for j = 1:100
        ax6.plot(ts, xs[1, j, :])
        ax7.plot(ts, xs[2, j, :])
    end

    # Adjust layout and display/save the figure
    tight_layout()
    savefig("fig/combined-spiral-2d.png")

    n_frames = size(xs)[end]
    filenames = String[]

    # Generate frames: each frame is a scatter plot of 100 random 2D points scaled to [0,10]
    i = 1
    for (i_, i) in enumerate(vcat(1:n_frames-1, fill(n_frames-1, 10)))
        fig, axs = subplots(1, 2, figsize=(6*2, 6))
        data = xs[:, :, i]
        ax1 = axs[1]
        ax1.set_title("Frame $i")
        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims)
        ax1.scatter(data[1,:], data[2,:], color="blue", alpha=0.7)
        data = ms[:, :, i]
        ax2 = axs[2]
        ax2.set_title("Frame $i")
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)
        ax2.scatter(data[1,:], data[2,:], color="blue", alpha=0.7)
        filename = @sprintf("figs/frame_%03d.png", i_)
        savefig(filename)
        close("all")
        push!(filenames, filename)
    end
    # Create an animated GIF using ImageMagick's 'convert'
    # The -delay option sets the delay between frames (in 1/100ths of a second),
    # and -loop 0 makes the GIF loop infinitely.
    run(`convert -delay 5 -loop 0 "figs/*.png" fig/scatter_animation.gif`)

    # # Optionally, remove the temporary frame files
    # for filename in filenames
    #     rm(filename)
    # end
    # println("Animated GIF saved as scatter_animation.gif")
end

"""
WIP
init_x = args.σ_max .* randn_like(X_val) |> gpu
n_steps = 10
time_steps = @> LinRange(1.0f0, 1f-2, n_steps) collect
time_step = time_steps[1]
snr = 0.16f0
eps = 1e-3
This is a single iteration of Langevin Dynamics:
    x_k+1 = x_k + ϵ * ∇_x logp(x) + 2ϵ ⋅ ξ_k
with ϵ being the step size, and ξk ∼ N(0,I) Gaussian noise. But rather than using a fixed ϵ, the code adaptively sets it based on the SNR (signal-to-noise ratio).
"""
function pc_sampler(diffusion_model, init_x, time_steps, Δt, σ_max, snr = 0.16f0, eps = 1e-3)
    x = x_mean = init_x
    for time_step in time_steps
        batch_time_step = ones_like(init_x, size(init_x)[end]) .* time_step
        ∇x = @> diffusion_model(x, batch_time_step)
        grad_norm = mean([norm(v) for v in eachcol(∇x)])
        # Compute the noise norm (product of dimensions excluding batch)
        noise_norm = sqrt(prod(size(x)[1:end-1]))
        langevin_step_size = 2 * (snr * noise_norm / grad_norm)^2
        x = x .+ langevin_step_size .* ∇x .+ sqrt(2 * langevin_step_size) .* randn_like(x)
        ## Predictor step (Euler–Maruyama)
        g = diffusion_coeff(batch_time_step, args.σ_max)  # g has shape (batch_size,)
        # Reshape g^2 to broadcast over spatial dims: (batch_size, 1, 1, 1)
        g2 = reshape(g .^ 2, batch_size, 1, 1, 1)
        x_mean = x .+ g2 .* diffusion_model(x, batch_time_step) .* Δt
        # Compute sqrt(g^2 * Δt) per sample and reshape
        g_step = reshape(sqrt.(g .^ 2 .* Δt), batch_size, 1, 1, 1)
        x = x_mean .+ g_step .* randn(Float32, size(x))
    end
    # Return the final mean (without the added noise on the last step)
    return x_mean
end


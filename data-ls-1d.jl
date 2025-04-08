using Random
using Distributions
using LinearAlgebra
using Distances
using PyPlot  # For plotting with pyplot

using Random, Distributions

"""
    generate_ls_data(num_samples)

Generate synthetic data for a causal model \( X \to Y \) using a location-scale noise model.

# Steps
1. **Generate Parent \( X \):**
   \( X \) is constructed with heteroscedastic noise by drawing a standard normal value scaled by a random variance factor.

2. **Nonlinear Transformation \( f(X) \):**
   We apply a sigmoid-like transformation to \( X \) with a random horizontal shift, a scale factor (which may be positive or negative), and a vertical scaling factor.

3. **Location-Scale Noise on \( f(X) \):**
   Additional noise is added to the transformed \( X \). This noise is scaled by the difference between \( f(X) \) and its minimum, so that the noise magnitude depends on the local "location" of \( f(X) \).

# Returns
A two-column array where the first column is \( X \) (the parent) and the second column is \( Y \) (the child).
"""

function generate_ls_data(num_samples::Int)
    # Generate parent variable X with heteroscedastic noise.
    noise_variance_X = rand(Uniform(1, 2), num_samples)
    noise_for_X = randn(num_samples)
    parent_X = @. sqrt(noise_variance_X) * noise_for_X
    # Define parameters for the sigmoid-like transformation.
    shift_param = rand(Uniform(-2, 2))         # horizontal shift
    b_selector  = rand(Binomial(1, 0.5))
    b_scale_param = b_selector == 1 ? rand(Uniform(0.5, 2)) : rand(Uniform(-2, -0.5))
    vertical_scale_param = rand(Exponential(4)) + 1  # vertical scaling factor
    # Apply the sigmoid-like transformation f(X)
    transformed_X = @. vertical_scale_param * (b_scale_param * (parent_X + shift_param)) /
                         (1 + abs(b_scale_param * (parent_X + shift_param)))
    # Generate noise for the child variable Y.
    noise_variance_Y = rand(Uniform(1, 2), num_samples)
    noise_for_Y = randn(num_samples)
    noise_Y = @. sqrt(noise_variance_Y) * noise_for_Y
    # Add location-scale noise to f(X) to obtain Y.
    child_Y = @. transformed_X + (transformed_X - mean(transformed_X) + 0.5) * noise_Y
    # Combine parent and child into a two-column array.
    return hcat(parent_X, child_Y)
end

# Example usage: Generate 500 samples.
data = generate_ls_data(500)

close("all")
figure()
scatter(data[:, 1], data[:, 2])
title("Location-Scale Causal Model (X → Y)")
xlabel("X")
ylabel("Y")
savefig("fig/ls.png")

using Distributions, Random
using PyPlot

function generate_ls_data_2d(num_samples::Int)
    Random.seed!(1)
    # Generate 2D parent variable X with heteroscedastic noise
    noise_variance_X = rand(Uniform(1, 2), num_samples, 2)
    noise_for_X = randn(num_samples, 2)
    parent_X = sqrt.(noise_variance_X) .* noise_for_X
    # Define transformation parameters
    shift_param = rand(Uniform(-2, 2), 2)
    b_selector = rand(Binomial(1, 0.5), 2)
    b_scale_param = [b == 1 ? rand(Uniform(0.5, 2)) : rand(Uniform(-2, -0.5)) for b in b_selector]
    vertical_scale_param = rand(Exponential(4), 2) .+ 1
    # Apply sigmoid-like transformation
    shifted_scaled_X = b_scale_param .* (parent_X .+ shift_param')
    transformed_X = vertical_scale_param .* shifted_scaled_X ./ (1 .+ abs.(shifted_scaled_X))
    combined_transformed = sum(transformed_X, dims=2)
    # Location-scale noise
    noise_variance_Y = rand(Uniform(1, 2), num_samples)
    noise_for_Y = randn(num_samples)
    noise_Y = sqrt.(noise_variance_Y) .* noise_for_Y
    noise_multiplier = combined_transformed .- mean(combined_transformed) .+ 0.5
    child_Y = combined_transformed[:, 1] .+ noise_multiplier[:, 1] .* noise_Y
    return parent_X[:, 1], parent_X[:, 2], child_Y, shift_param, b_scale_param, vertical_scale_param
end

function f_noiseless_surface(X1, X2, shift_param, b_scale_param, vertical_scale_param)
    # Apply same transformation as in generate_ls_data_2d
    shifted_scaled_x1 = b_scale_param[1] .* (X1 .+ shift_param[1])
    shifted_scaled_x2 = b_scale_param[2] .* (X2 .+ shift_param[2])
    fx1 = vertical_scale_param[1] .* shifted_scaled_x1 ./ (1 .+ abs.(shifted_scaled_x1))
    fx2 = vertical_scale_param[2] .* shifted_scaled_x2 ./ (1 .+ abs.(shifted_scaled_x2))
    return fx1 .+ fx2
end

# Generate data
x1, x2, y, shift_param, b_scale_param, vertical_scale_param = generate_ls_data_2d(500)

# Prepare surface grid
x1_range = range(minimum(x1), maximum(x1), length=40)
x2_range = range(minimum(x2), maximum(x2), length=40)
X1_grid = repeat(reshape(x1_range, :, 1), 1, 40)
X2_grid = repeat(reshape(x2_range, 1, :), 40, 1)
Z = f_noiseless_surface(X1_grid, X2_grid, shift_param, b_scale_param, vertical_scale_param)

# Plot 3D scatter and surface
fig = figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Surface
ax.plot_surface(X1_grid, X2_grid, Z, alpha=0.6, color="orange", edgecolor="none", label="Noiseless f(X)")

# Noisy points
ax.scatter(x1, x2, y, alpha=0.4, label="Noisy Y", s=10)

ax.set_title("Location-Scale Causal Model (2D X → Y)", fontsize=12)
ax.set_xlabel("X₁")
ax.set_ylabel("X₂")
ax.set_zlabel("Y")
tight_layout()
savefig("fig/ls-2d.png")


"""
n = 100
noise_type = 3
"""
function sample_generative_model(n::Int; noise_type::Int=1)
    noise_exp = 1
    ran = randn(n)
    noise_var = rand(Uniform(1, 2), n)
    # Element-wise operations: take square root, absolute value, then raise to noise_exp and multiply by sign.
    noisetmp = (sqrt.(noise_var) .* abs.(ran)).^noise_exp .* sign.(ran)
    x_pa = copy(noisetmp)
    noise_var_ch = rand(Uniform(1, 2), n)

    # Injective mechanism using a sigmoid-like function.
    a_sig = rand(Uniform(-2, 2))
    bern = rand(Binomial(1, 0.5))
    b_sig = bern == 1 ? rand(Uniform(0.5, 2)) : rand(Uniform(-2, -0.5))
    c_sig = rand(Exponential(4)) + 1  # Exponential with rate 4; note mean=1/4.
    # Apply a sigmoid-like transformation.
    x_child = c_sig .* (b_sig .* (x_pa .+ a_sig)) ./ (1 .+ abs.(b_sig .* (x_pa .+ a_sig)))

    # Generate noise to modify x_child.
    ran = randn(n)
    noisetmp = (0.2 * sqrt.(noise_var_ch) .* abs.(ran)).^noise_exp .* sign.(ran)

    x_child .= x_child .+ (x_child .- minimum(x_child)) .* noisetmp

    if noise_type == 1
        # Additive noise.
        x_child .= x_child .+ noisetmp
    elseif noise_type == 2
        # Multiplicative noise.
        x_child .= x_child .* rand(Uniform(0, 1), n)
    elseif noise_type == 3
        # Location-scale noise.
        x_child .= x_child .+ (x_child .- minimum(x_child)) .* noisetmp
    elseif noise_type == 4
        # Alternative noise formulation.
        ran = randn(n)
        sd = x_child .- minimum(x_child)
        x_child .= (0.2 * sqrt.(sd) .* abs.(ran)).^noise_exp .* sign.(ran)
    else
        error("Noise type not implemented.")
    end

    # Return a matrix with the parent variable in the first column and child variable in the second.
    return hcat(x_pa, x_child)
end

# Convenience wrapper functions:
sample_ANs(n::Int)  = sample_generative_model(n; noise_type=1)
sample_MN_u(n::Int) = sample_generative_model(n; noise_type=2)
sample_LSs(n::Int)  = sample_generative_model(n; noise_type=3)

# Generate data using the location-scale (LS) model.
data = sample_LS(500)

# Plotting with PyPlot.
scatter(data[:, 1], data[:, 2])
title("Location-Scale Causal Model (X → Y)")
xlabel("X")
ylabel("Y")
savefig("fig/ls.png")


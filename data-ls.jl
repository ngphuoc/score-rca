using LinearAlgebra
using Distances
using PyPlot  # For plotting with pyplot
using Random, Distributions, Statistics

# function generate_ls_data_2d(num_samples::Int)
#     Random.seed!(1)
#     # Generate 2D parent variable X with heteroscedastic noise
#     noise_variance_X = rand(Uniform(1, 2), 2, num_samples)
#     noise_for_X = randn(2, num_samples)
#     parent_X = sqrt.(noise_variance_X) .* noise_for_X
#     # Define transformation parameters
#     shift_param = rand(Uniform(-2, 2), 2)
#     b_selector = rand(Binomial(1, 0.5), 2)
#     b_scale_param = [b == 1 ? rand(Uniform(0.5, 2)) : rand(Uniform(-2, -0.5)) for b in b_selector]
#     vertical_scale_param = rand(Exponential(4), 2) .+ 1
#     # Apply sigmoid-like transformation
#     shifted_scaled_X = b_scale_param .* (parent_X .+ shift_param)
#     transformed_X = vertical_scale_param .* shifted_scaled_X ./ (1 .+ abs.(shifted_scaled_X))
#     combined_transformed = sum(transformed_X, dims=2)
#     # Location-scale noise
#     noise_variance_Y = rand(Uniform(1, 2), num_samples)
#     noise_for_Y = randn(num_samples)
#     noise_Y = sqrt.(noise_variance_Y) .* noise_for_Y
#     noise_multiplier = combined_transformed .- mean(combined_transformed) .+ 0.5
#     child_Y = combined_transformed[:, 1] .+ noise_multiplier[:, 1] .* noise_Y
#     return parent_X[:, 1], parent_X[:, 2], child_Y, shift_param, b_scale_param, vertical_scale_param
# end

function generate_ls_data_2d(num_samples::Int)
    # --- Generate 2D parent variable X with heteroscedastic noise ---
    # For each feature (2 features) and each sample, we generate a random variance from Uniform(1,2).
    noise_variance_X = rand(Uniform(1, 2), 2, num_samples)
    # Generate standard normal noise for both features and samples.
    noise_for_X = randn(2, num_samples)
    # Parent variable X: each element is scaled by the square root of the corresponding noise variance.
    parent_X = @. sqrt(noise_variance_X) * noise_for_X

    # --- Define transformation parameters (each is a 2-element vector, one per feature) ---
    # Shift parameter: a random horizontal offset for each feature.
    shift_param = rand(Uniform(-2, 2), 2)
    # Binary selector: for each feature, choose 0 or 1 with equal probability.
    b_selector = rand(Binomial(1, 0.5), 2)
    # Scaling parameter for the nonlinearity for each feature.
    b_scale_param = [b == 1 ? rand(Uniform(0.5, 2)) : rand(Uniform(-2, -0.5)) for b in b_selector]
    # Vertical scaling parameter (adds vertical stretch) for each feature.
    vertical_scale_param = rand(Exponential(4), 2) .+ 1

    # --- Apply the sigmoid-like transformation to each feature ---
    # For each feature (row) and each sample (column), add the shift parameter to X,
    # then multiply by the corresponding scale parameter.
    # Broadcasting makes sure that the 2-element vectors are applied feature-wise.
    shifted_scaled_X = @. b_scale_param * (parent_X + shift_param)
    transformed_X = @. vertical_scale_param * shifted_scaled_X / (1 + abs(shifted_scaled_X))

    # Combine the two features per sample into a single value.
    # Since each column is a sample, sum the two features (rows) for each sample.
    combined_transformed = sum(transformed_X, dims=1)  # 1×num_samples array

    # --- Generate location-scale noise and add it to the combined transformation to get Y ---
    # Generate noise variance and standard normal noise for Y, per sample.
    noise_variance_Y = rand(Uniform(1, 2), num_samples)
    noise_for_Y = randn(num_samples)
    # Noise for Y is scaled by 0.2 and the square root of noise variance.
    noise_Y = @. 0.2 * sqrt(noise_variance_Y) * noise_for_Y

    # Create a noise multiplier that adjusts with the level of f(X)
    # Here we center the combined transformation by subtracting its mean and add 0.5.
    noise_multiplier = @. combined_transformed - mean(combined_transformed) + 0.5

    # For each sample (each column), add the scaled noise to the combined transformed signal.
    # Convert the 1×num_samples array to a vector with [:] for elementwise operations.
    child_Y = combined_transformed[:] .+ noise_multiplier[:] .* noise_Y

    # --- Return the results ---
    # Note: parent_X is 2×num_samples. Each row is one feature.
    # We return the first feature, second feature, and the generated child variable Y,
    # along with the transformation parameters.
    return parent_X[1, :], parent_X[2, :], child_Y, shift_param, b_scale_param, vertical_scale_param
end

# Example usage: generate data for num_samples samples.
feature1, feature2, child_Y, shift_param, b_scale_param, vertical_scale_param = generate_ls_data_2d(500)


function f_noiseless_surface(X1, X2, shift_param, b_scale_param, vertical_scale_param)
    # Apply same transformation as in generate_ls_data_2d
    shifted_scaled_x1 = b_scale_param[1] .* (X1 .+ shift_param[1])
    shifted_scaled_x2 = b_scale_param[2] .* (X2 .+ shift_param[2])
    fx1 = vertical_scale_param[1] .* shifted_scaled_x1 ./ (1 .+ abs.(shifted_scaled_x1))
    fx2 = vertical_scale_param[2] .* shifted_scaled_x2 ./ (1 .+ abs.(shifted_scaled_x2))
    return fx1 .+ fx2
end

# Generate data
num_samples = 500
x1, x2, y, shift_param, b_scale_param, vertical_scale_param = generate_ls_data_2d(num_samples)

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


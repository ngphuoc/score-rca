include("lib/plot-utils.jl")

function plot_data_gradients(normal_df, perturbed_df, regressor, oracle, unet, σX, μX; xlim, ylim)
    @≥ regressor, oracle, unet gpu.();

    #-- defaults
    default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)

    #-- plot data
    x, y = eachcol(normal_df)
    pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Data $(x, y)$")

    #-- plot r2
    x = @> vcat(normal_df, perturbed_df) Array transpose Array gpu;
    @> x[2, :] minimum, maximum, mean, std
    # x = @> normal_df Array transpose Array gpu;
    d = size(x, 1)
    total_loss = 0.0
    xj = unsqueeze(x, 1)
    @≥ x unsqueeze(2) repeat(1, d, 1)
    x̂ = regressor(x)
    @assert size(xj) == size(x̂)
    y, ŷ = @> xj, x̂ getindex.(1,2,:) cpu.() vec.();
    pl_r2 =  scatter(y, ŷ; xlab=L"y", ylab=L"\hat{y}", title=L"Regression $R^2=%$(round(float(r2_score(y, ŷ)), digits=2))$")

    #-- plot oracle r2, need to save μX, σX for oracle regressor
    # x = @> vcat(normal_df, perturbed_df) Array transpose Array gpu;
    x = @> normal_df Array transpose Array gpu;
    d = size(x, 1)
    total_loss = 0.0
    xj = unsqueeze(x, 1);
    @≥ x unsqueeze(2) repeat(1, d, 1)
    @≥ σX, μX gpu.()
    x0 = @. x * σX + μX
    x̂ = (oracle(x0) .- μX') ./ σX'
    @assert size(xj) == size(x̂)
    y, ŷ = @> xj, x̂ getindex.(1,2,:) cpu.() vec.();
    # m = dag.cpds[2].mlp
    # ŷ = @> m(cpu(x[[1], :])) vec
    # y = @> x[2, :] cpu
    pl_oracle =  scatter(y, ŷ; xlab=L"y", ylab=L"\hat{y}", title=L"Oracle Regression $R^2=%$(round(float(r2_score(y, ŷ)), digits=2))$")

    #-- plot perturbations
    x, y = eachcol(perturbed_df)
    pl_perturbed_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Perturbed $3\sigma$ $(x, y)$")

    x = @> normal_df Array transpose Array gpu;
    d = size(x, 1)
    X, batchsize = size(x, 1), size(x)[end]
    j_mask = @> I(X) Matrix{Float32} gpu;
    xj = x[[2], :]
    t = rand!(similar(xj)) .* (1f0 - 1f-5) .+ 1f-5
    σ_t = marginal_prob_std(t)
    z = 2rand!(similar(x)) .- 1;
    x̃ = x .+ σ_t .* z
    x, y = eachrow(cpu(x̃))
    pl_sm_data = scatter(x, y; xlab=L"x", ylab=L"y", title="Perturbed score matching")

    #-- plot gradients
    x = @> Iterators.product(range(xlim..., length=30), range(ylim..., length=30)) collect vec;
    x = @> reinterpret(reshape, Float64, x) Array{Float32} gpu;
    d = size(x, 1)
    xj = unsqueeze(x, 1);
    n = norm2(xj, dims=2)
    n = n ./ maximum(n)
    @> n vec minimum, maximum, mean, std
    t = fill!(similar(xj), 0.1)
    # t = 0.5f0n
    # t = repeat(t, outer=(1,2,1))
    σ_t = marginal_prob_std(t)
    @≥ x unsqueeze(2) repeat(1, d, 1)
    J = @> unet(x, t)[:, 2, :]
    x = @> squeeze(xj, 1)
    @≥ J, x cpu.()
    x, y = eachrow(x);
    u, v = eachrow(0.05J);
    _, _, _, s = @> norm2(J, dims=1) maximum, minimum, mean, std

    pl_gradient = scatter(x, y, markersize=0, lw=0, color=:white);
    arrow0!.(x, y, u, v; as=1.0, lw=1.0);
    # quiver!(x, y, quiver=(u, v), aspect_ratio=:equal)
    @> Plots.plot(pl_data, pl_perturbed_data, pl_sm_data, pl_r2, pl_oracle, pl_gradient; xlim, ylim, size=(1000, 800)) savefig("fig/$datetime_prefix-2d.png")
end

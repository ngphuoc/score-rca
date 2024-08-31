
function plot_gradients(normal_df, perturbed_df, unet; xlim, ylim, args)
    @≥ unet gpu()

    #-- defaults
    default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)

    #-- plot data
    x, y = eachcol(normal_df)
    pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Data $(x, y)$")

    #-- plot perturbations
    x, y = eachcol(perturbed_df)
    pl_perturbed_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Perturbed $3\sigma$ $(x, y)$")

    x = @> normal_df Array transpose Array gpu;
    d = size(x, 1)
    X, batchsize = size(x, 1), size(x)[end]
    t = rand!(similar(x, size(x)[end])) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
    z = 2rand!(similar(x)) .- 1;
    x̃ = x .+ σ_t .* z

    x, y = eachrow(cpu(x̃))
    pl_sm_data = scatter(x, y; xlab=L"x", ylab=L"y", title="Perturbed score matching")

    #-- plot gradients
    x = @> Iterators.product(range(xlim..., length=20), range(ylim..., length=20)) collect vec;
    x = @> reinterpret(reshape, Float64, x) Array{Float32} gpu;
    d = size(x, 1)
    t = fill!(similar(x, size(x)[end]), 0.1) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
    J = @> unet(x, t)
    @≥ J, x cpu.()
    x, y = eachrow(x);
    u, v = eachrow(0.2J);
    pl_gradient = scatter(x, y, markersize=0, lw=0, color=:white);
    arrow0!.(x, y, u, v; as=0.2, lw=1.0);

    ##-- plot slop estimate
    #x = @> range(xlim..., length=20) Array gpu
    #d = size(x, 1)
    #t = fill!(similar(x, size(x)[end]), 0.1) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    #σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
    #J = @> unet(x, t)
    #@≥ J, x cpu.()
    #x, y = eachrow(x);
    #u, v = eachrow(0.2J);
    #pl_gradient = scatter(x, y, markersize=0, lw=0, color=:white);
    #arrow0!.(x, y, u, v; as=0.2, lw=1.0);

    @> Plots.plot(pl_data, pl_perturbed_data, pl_sm_data, pl_gradient; xlim, ylim, size=(1000, 800)) savefig("fig/$datetime_prefix-2d.png")
end

function plot_data_gradients(normal_df, perturbed_df, regressor, oracle, unet, σX, μX; xlim, ylim)
    @≥ regressor, oracle, unet gpu.();

    #-- defaults
    default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)

    #-- plot data
    x, y = eachcol(normal_df)
    pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Data $(x, y)$")

    #-- plot r2
    x = @> vcat(normal_df, perturbed_df) Array transpose Array gpu;
    # x = @> normal_df Array transpose Array gpu;
    d = size(x, 1)
    total_loss = 0.0
    y = x[2, :]
    @≥ x unsqueeze(2) repeat(1, d, 1)
    x̂ = regressor(x)
    ŷ = x̂[1, 2, :]
    @≥ ŷ, y cpu.()
    pl_r2 =  scatter(y, ŷ; xlab=L"y", ylab=L"\hat{y}", title=L"Regression $R^2=%$(round(float(r2_score(y, ŷ)), digits=2))$")

    #-- plot oracle r2, need to save μX, σX for oracle regressor
    # x = @> vcat(normal_df, perturbed_df) Array transpose Array gpu;
    x = @> normal_df Array transpose Array gpu;
    d = size(x, 1)
    total_loss = 0.0
    y = x[2, :]
    @≥ σX, μX gpu.()
    x0 = @. x * σX + μX
    @≥ x unsqueeze(2) repeat(1, d, 1)
    @≥ σX, μX gpu.()
    x0 = @. x * σX + μX
    x̂ = (oracle(x0) .- μX') ./ σX'
    ŷ = x̂[1, 2, :]
    @≥ ŷ, y cpu.()
    pl_oracle =  scatter(y, ŷ; xlab=L"y", ylab=L"\hat{y}", title=L"Oracle Regression $R^2=%$(round(float(r2_score(y, ŷ)), digits=2))$")

    #-- plot perturbations
    x, y = eachcol(perturbed_df)
    pl_perturbed_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Perturbed $3\sigma$ $(x, y)$")

    x = @> normal_df Array transpose Array gpu;
    d = size(x, 1)
    X, batchsize = size(x, 1), size(x)[end]
    t = rand!(similar(x, size(x)[end])) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    σ_t = expand_dims(marginal_prob_std(t), 1)
    z = 2rand!(similar(x)) .- 1;
    @show size(x) size(σ_t) size(z)
    x̃ = x .+ σ_t .* z

    x, y = eachrow(cpu(x̃))
    pl_sm_data = scatter(x, y; xlab=L"x", ylab=L"y", title="Perturbed score matching")

    #-- plot gradients
    x = @> Iterators.product(range(xlim..., length=30), range(ylim..., length=30)) collect vec;
    x = @> reinterpret(reshape, Float64, x) Array{Float32} gpu;
    d = size(x, 1)
    t = fill!(similar(x, size(x)[end]), 1.0) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    σ_t = expand_dims(marginal_prob_std(t), 1)
    J = @> unet(x, t)
    @≥ J, x cpu.()
    x, y = eachrow(x);
    u, v = eachrow(10J);

    pl_gradient = scatter(x, y, markersize=0, lw=0, color=:white);
    arrow0!.(x, y, u, v; as=1.0, lw=1.0);
    @> Plots.plot(pl_data, pl_perturbed_data, pl_sm_data, pl_r2, pl_oracle, pl_gradient; xlim, ylim, size=(1000, 800)) savefig("fig/$datetime_prefix-2d.png")
end


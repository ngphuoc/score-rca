include("lib/plot-utils.jl")

using LaTeXStrings

function plot_3data(train_df, perturbed_df, anomaly_df; xlim, ylim, fig_path="fig/3data-2d.png")
    #-- defaults
    default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)
    #-- plot data
    x, y = eachcol(train_df)
    pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"data $(x, y)$")
    x, y = eachcol(perturbed_df)
    pl_perturbed = scatter(x, y; xlab=L"x", ylab=L"y", title=L"perturbed data $(x, y)$")
    x, y = eachcol(anomaly_df)
    pl_anomaly = scatter(x, y; xlab=L"x", ylab=L"y", title=L"anomaly data $(x, y)$")
    @> Plots.plot(pl_data, pl_perturbed, pl_anomaly; xlim, ylim, size=(1000, 800)) savefig(fig_path)
end

function plot_dsm()
    xlim = ylim = (-3, 3)

    #-- defaults
    default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, color=:seaborn_deep, markersize=2, leg=nothing)

    dsm_loss(dnet, x; args.σ_max)
    X_val = X[1:100:end, :]' |> cpu
    x, y = eachrow(X_val)
    pl_data = scatter(x, y; xlab=L"x", ylab=L"y", title=L"Data $(x, y)$")

    #-- plot perturbed data
    x = @> X_val gpu;
    d = size(x, 1)
    X, batchsize = size(x, 1), size(x)[end]
    t = rand!(similar(x, size(x)[end])) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
    z = 2rand!(similar(x)) .- 1;
    x̃ = x .+ σ_t .* z
    x, y = eachrow(cpu(x̃))
    pl_sm_data = scatter(x, y; xlab=L"x", ylab=L"y", title="Perturbed score matching")

    #-- plot gradients
    μx, σx = @> X_val mean(dims=2), std(dims=2)
    x = @> Iterators.product(range(xlim..., length=50), range(ylim..., length=50)) collect vec;
    x = @> reinterpret(reshape, Float64, x) Array{Float32} gpu;
    t = fill!(similar(x, size(x)[end]), 0.01) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
    J = @> dnet(x, t)
    @≥ J, x cpu.()
    mesh, scores = @> x', J' Array.()
    plot_score_field(mesh, scores, width=0.002, vis_path="fig/score-field.png")

    #-- plot outlier gradients
    X = @> anomaly_df Array;
    X_val = X' |> cpu
    x = @> X_val gpu;
    t = fill!(similar(x, size(x)[end]), 0.01) .* (1f0 - 1f-5) .+ 1f-5  # same t for j and paj
    σ_t = expand_dims(marginal_prob_std(t; args.σ_max), 1)
    J = @> dnet(x, t)
    @≥ J, x cpu.()
    mesh, scores = @> x', J' Array.()
    plot_score_field(mesh, scores, width=0.002, vis_path="fig/score-field-outlier.png")

    @> Plots.plot(pl_data, pl_sm_data; xlim, ylim, size=(1000, 800)) savefig("fig/main-rca-2d.png")
end


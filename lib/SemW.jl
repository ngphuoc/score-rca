include("using.jl")
include("args.jl")


struct Sem
    B::Matrix{Bool}
    pas::Vector{Vector{Int}}
end

init_p() = (W = (μ = zeros(4, 4) , ρ = log.(1e-9ones(4, 4))),
            b  = (μ = zeros(4) , ρ = log.(1e-9ones(4))), )

function reparam(μ, σ)
    μ .+ σ .* randn(size(σ))
end

function forward(p, X)
    μ = p.W.μ'X .+ p.b.μ
    σ² = exp.(p.W.ρ)' * X .^ 2 .+ exp.(p.b.ρ)
    Y = reparam(μ, .√σ²)
    Y, μ, σ²
end

"""
Convert parameters W and p(ϵ) of the SEM X=W'X + ϵ into a list of fcms in topo order
and observation matrix into observation list
"""
function dag2fcm(W, p_ϵ, X)
    d = size(W, 1)
    B = W .!= 0
    fs = map(1:d) do j
        pa = findall(W[:, j] .!= 0)
        f(x) = W[pa, j]'x
        pa -> w
    end
    return B, p_ϵ, fcms,, xs
end

function forward(sem, xs, ps, st)
end

# Data
function generate(p, n)
    X = ones(d, n)
    i = 1
    for i = 1:4
        Y, = forward(p, X)
        X[[i], :] .= Y[[i], :]  # only use time step i
    end
    X
end

function get_root_cause_data(rng, d, e, n; graph_type, sem_type, noise_type, noise_scale)
    rng, rng, rng, Y_rng, Z_rng = split(rng, 5);
    A = simulate_dag(rng, d, e, graph_type)
    W = simulate_parameters(rng, A)
    X = simulate_sem(rng, W, n, sem_type, noise_type, noise_scale)
    Y = rand(Y_rng, 1:d)
    Z = simulate_sem(Z_rng, W, n, sem_type, noise_type, noise_scale, fault="drift", faulty_node=Y)
    @≥ X, Z transpose.() Array{Float64}.()
    @≥ A Int.()
    return Dict(
            "A" => A,
            "W" => W,
            "X" => X,
            "Y" => Y,
            "Z" => Z,
           )
end

function leaf_nodes(A)
    @assert sum(tril(A, -1)) == 0
    a = @> triu(A, 1) sum(dims=2) vec
    findall(a .== 0)
end

function leaves_ancestors(A)
    d = size(A, 2)
    C = A .+ zeros(d, d, d)
    for i = 2:d
        C[:, :, i] .= A^i
    end
    links = @> sum(C, dims=3) .> 0 dropdims(dims=3)
    leaves = leaf_nodes(links)
    ancestors = map(leaves) do l
        findall(links[:,l] .> 0)
    end
    leaves, ancestors, links
end

function at_leat_10_ancestors(A)
    leaves, ancestors = leaves_ancestors(A)
    any(length.(ancestors) .>= 10)
end

""" Get the deterministic part
    Assume correct order
"""
function get_fcm(W, sem_type)
    d = size(W, 2)
    map(1:d) do j
        pa = findall(W[:, j] .!= 0)
        f = (x) -> begin
            if sem_type ∈ ["linear", "mix"]
                length(x) == 0 ? [0.0] : W[pa, j]'x
            elseif sem_type == "gp"
                gp = gaussian_process.GaussianProcessRegressor()
                x = vec(gp.sample_y(x', random_state=rand(rng, UInt32)))'
            elseif sem_type == "logistic"
                # rand(rng, Binomial(1, sigmoid.(w' * x)) * 1.0)
            elseif sem_type == "poisson"
                # rand(rng, Poisson(exp.(w' * x)) * 1.0)
            elseif sem_type == "mlp"
                # H = 100
                # w1 = rand(rng, Uniform(0.5, 2.0), (P, H))
                # w1[rand(rng, size(w1)...) .< 0.5] .*= -1
                # w2 = rand(rng, Uniform(0.5, 2.0), H)
                # w2[rand(rng, H) .< 0.5] .*= -1
                # w2 * s(w1 * x)
            elseif sem_type == "mim"
                # w1 = rand(rng, Uniform(0.5, 2.0), P)
                # w1[rand(rng, P) .< 0.5] .*= -1
                # w2 = rand(rng, Uniform(0.5, 2.0), P)
                # w2[rand(rng, P) .< 0.5] .*= -1
                # w3 = rand(rng, Uniform(0.5, 2.0), P)
                # w3[rand(rng, P) .< 0.5] .*= -1
                # tanh(w1 * x) + cos(w2 * x) + sin(w3 * x)
            elseif sem_type == "gp-add"
                # gp = GaussianProcessRegressor()
                # x = sum([gp.sample_y(x[:, i, nothing], random_state=nothing).flatten() for i in range(size(x)[1])])
            else
                error("unknown sem type")
            end
        end
        j, pa, f
    end
end

"""2-layer MLP for nonlinear connection, not trainable"""
struct MLP
    σ
    W1
    b1
    W2
    b2
end

function Base.show(io::IO, m::MLP)
    @extract m σ, W1, W2
    h, i = size(m.W1)
    o, h = size(m.W2)
    print(io, "MLP($i -> $σ($h) -> $σ($o))")
end

function MLP(rng::AbstractRNG, input_size, w_rng=(-5, 5), σ=sigmoid)
    hidden_size = rand(rng, 2:100)
    l, u = w_rng
    MLP(σ,
        (u-l) .* rand(rng, hidden_size, input_size) .- l,
        (u-l) .* rand(rng, hidden_size) .- l,
        (u-l) .* rand(rng, 1, hidden_size) .- l,
        (u-l) .* rand(rng, 1) .- l,
       )
end

function (m::MLP)(x)
    @extract m W1, b1, W2, b2, σ
    h = σ.(W1*x .+ b1)
    # return σ.(W2*h .+ b2)
    return W2*h .+ b2
end

zero_(_) = [0.0]

""" Get the deterministic part
    Assume correct order
"""
function get_mix_fcm(rng, W, fcm)
    d = size(W, 2)
    map(1:d) do j
        j, pa, f = fcm[j]
        f = if length(pa) == 0
            zero_
        elseif rand(rng) > p_nonlinear  # 20% linear
            f
        else
            j, pa, f = fcm[j]
            MLP(rng, length(pa), (-5, 5))
        end
        return j, pa, f
    end
end

# (i, pa, f) = fcm[4]

# function generate(fcm::Vector, ϵ)
#     X = copy(ϵ)
#     for (i, pa, f) in fcm
#         X[i, :] .= vec(f(X[pa, :])) .+ ϵ[i, :]
#     end
#     return X
# end

function generate(W::Matrix{Float64}, ϵ)
    X = copy(ϵ)
    d = size(W, 1)
    for i = 1:d
        X[[i], :] .= (W'X)[[i], :] .+ ϵ[[i], :]
    end
    return X
end

function get_noise(rng, noise_type, s, n)
    ϵ = noise_type == "gaussian" ? rand(rng, Normal(0, s), d, n) :
        noise_type == "exponential" ? rand(rng, Exponential(s), d, n) :
        noise_type == "gumbel" ? rand(rng, Gumbel(s), d, n) :
        noise_type == "uniform" ? rand(rng, Uniform(-s, s), d, n) :
                                  rand(rng, Normal(0, s), d, n)
end

# rng = SplittableRandom(seed)
function grad_data(rng, d, e, n, m=10; graph_type, sem_type, noise_type="gauss", noise_scale=1.0)
    rng, rng, rng, Y_rng, Z_rng = split(rng, 5);
    A = il = leaves = ancestors = nothing
    for _=1:100
        A = simulate_dag(rng, d, e, graph_type)
        leaves, ancestors = leaves_ancestors(A)
        il = findfirst(length.(ancestors) .>= 10)  # a leaf with ≥10 ancestors
        il != nothing && break
        # any(length.(ancestors) .>= 10) && break
    end
    il == nothing && error("no 10 ancestors node be found, d=$d, e=$e")
    l = leaves[il]
    # model
    w_ranges=((-5.0, 5.0), )
    W = simulate_parameters(rng, A, w_ranges)
    fcm = get_fcm(W, sem_type)
    if sem_type == "mix"
        fcm = get_mix_fcm(rng, W, fcm)
    end
    ϵ = get_noise(rng, noise_type, noise_scale, n)
    X = generate(fcm, ϵ)
    # outliers
    # n_faulty = rand(rng, 2:5)
    n_faulty = 3
    faulty = @> rand(rng, ancestors[il] ∪ [l], n_faulty) unique sort  # faulty nodes
    λ = 3 .+ 2*rand(rng, length(faulty), m)  # uniform(3, 5)
    ϵ = randn(rng, d, m)
    ϵ[faulty, :] .= λ
    Z = generate(fcm, ϵ)
    @≥ A Int.()
    return Dict(
            "A" => A,
            "W" => W,
            "fcm" => fcm,
            "X" => X,
            "faulty" => (leaf=l, ancestors=ancestors[il], faulty=faulty),
            "Z" => Z,
           )
end

# rng = SplittableRandom(seed)
""" Generate normal and outlier data for linear SEM
    Don't use get_fcm() and generate() since there are outliers in mechanisms.
    Instead, work directly on W

n = 1000: #observations
d = 20: #nodes
e = 40: #edges
_a = 7: minimum number of ancestors in one of the leaf
m : #faulty observations
_d : #faulty nodes
_e : #faulty edges
"""
function faulty_data_linear(rng, d, e=2d, n; m=10, _a=round(Int, d/3), _d=0, _e=2; graph_type, sem_type, noise_type="gauss", noise_scale=1.0)
    # graph
    A = il = leaves = ancestors = links = nothing
    for _=1:100
        A = simulate_dag(rng, d, e, graph_type)
        leaves, ancestors, links = leaves_ancestors(A)
        il = findfirst(length.(ancestors) .>= _a)  # a leaf with ≥10 ancestors
        il != nothing && break
        # any(length.(ancestors) .>= 10) && break
    end
    il == nothing && error("no 10 ancestors node be found, d=$d, e=$e")
    leaf, ancestors, links = leaves[il], ancestors[il], findall(links[:, il:il])

    # normal SEM
    w_ranges=((0.5, 2.0), )
    W = simulate_parameters(rng, A, w_ranges)
    ϵ = get_noise(rng, noise_type, noise_scale, n)
    X = generate(W, ϵ)

    # outlier SEM
    _W = copy(W)
    w_ranges=(-2.0, -0.5)
    _i = @> rand(links, _a) CartesianIndex.()
    _W[_i] .= rand(Uniform(w_ranges...), length(_i))

    # outlier observations
    ϵ = get_noise(rng, noise_type, noise_scale, m)
    _X = generate(_W, ϵ)
    @≥ A Int.()
    return (; A, W, X, _W, _X, leaf, ancestors, links, _i)
end

function fit_lr(A, X; max_iter=100, learning_rate=1e-1)
    loss(X, ps) = mean(abs2, X - ps.Ŵ'X)

    ps = (W=zeros(size(A)), )
    opt = Adam(learning_rate)
    opt_st = Optimisers.setup(opt, ps)
    for i=1:max_iter
        # l, back = pullback(ps -> loss(X, ps), ps)
        # gs = back((one(l), nothing))[1]
        gs = gradient(ps -> loss(X, ps), ps)[1]
        opt_st, ps = Optimisers.update(opt_st, ps, gs)
        ps.W .*= A
        @show loss(X, ps), sum(gs.W)
    end
    return ps.W
end

function inverse_noise(W, X)
    ϵ = X - W'X
    return ϵ
end

rand_subset(s) = rand(s, rand(length(s)))

binary_indices(s, j) = indexin(s, j) .!= nothing


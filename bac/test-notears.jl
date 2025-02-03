include("data.jl")
include("log.jl")
flog = "notears-mlp.flux.log"
using Flux, Random, Optimisers, Zygote, EvalMetrics, Optim, FileIO, ComponentArrays
using Parameters: @unpack
using Zygote
using Functors
sopt = pyimport("scipy.optimize")
using Flux: glorot_uniform, Dense, Chain

using Flux, Random, NNlib, Zygote, OMEinsum

struct NotearsMLP
    dims::Vector{Int}
    pos
    neg
    chain
end

Functors.@functor NotearsMLP (pos, neg, chain)

function Base.show(io::IO, m::NotearsMLP)
    print(io, "NotearsMLP($(m.dims))")
end

using Flux: Dense

# dims = [d, 10, 1]

function NotearsMLP(dims::Vector{Int})
    D, H = dims[1:2]
    pos = Dense(D, H*D, bias=false, init=Constant(0.0))
    neg = Dense(D, H*D, bias=false, init=Constant(0.0))
    chain = Chain(
        [BatchDense(dims[i], dims[i+1], D, Flux.sigmoid; init=Constant(0.0), bias=Constant(0.0)) for i=2:length(dims)-2]...,
        BatchDense(dims[end-1], dims[end], D; init=Constant(0.0), bias=Constant(0.0)),
    )
    NotearsMLP(dims, pos, neg, chain)
end

function (m::NotearsMLP)(X)
    d, h = m.dims[1:2]
    z = m.pos(X) - m.neg(X)
    z = reshape(z, (h, d, :))
    z = Flux.sigmoid(z)  # sigmoid
    z = m.chain(z)
    x̂ = dropdims(z, dims=1)
    return x̂
end

# {"n": n, "d": d, "s0": s0, "graph_type": graph_type, "sem_type": sem_type, "B": B_true, "W": W_true}
# dict = FileIO.load("data-notears.jld2")
# B, X = dict["B"], dict["X"]

dict = FileIO.load("data-notears-nonlinear.jld2")
dict
n, d, s0, graph_type, sem_type, B, X = dict["n"], dict["d"], dict["s0"], dict["graph_type"], dict["sem_type"], dict["B"], dict["X"]
B
X
n, d = size(X)
# X = X .- mean(X, dims=1)
λ1 = 0.01
λ2 = 0.01
ρ, α, h = 1.0, 0.0, Inf

dims = [d, 10, 1]
m = NotearsMLP(dims)
ps, re = destructure(m)

function _loss(X, m)
    M = m(X')
    R = X - M'
    n = size(X, 1)
    l = 0.5 / n * sum(abs2, R)
    l
end

"WX' -> XW'"
function adjmat(m)
    d, h = m.dims[1:2]
    W = m.pos.weight - m.neg.weight
    W = @> W .* W reshape(h, d, d) sum(dims=1) dropdims(dims=1) .√
    W'
end

"""
tr[(I - 1/d W∘W)^d] - d = 0 (α=1/d)
"""
function _h(m)
    d, h = m.dims[1:2]
    W = m.pos.weight - m.neg.weight
    A = @> W .* W reshape(h, d, d) sum(dims=1) dropdims(dims=1) transpose
    # M = I + W .* W / d  # (Yu et al. 2019)
    # E = M ^ (d - 1)
    # h = sum(E' .* M) - d
    h = tr(exp(A)) - d
    return h
end

function _func(ps)
    """
    Evaluate value and gradient of augmented Lagrangian for
    doubled variables ([2 d^2] array).
    """
    function closure(ps)
        m = re(ps)
        loss = _loss(X, m)
        h = _h(m)
        penalty = 0.5 * ρ * h * h + α * h
        l12 = λ1 * (sum(m.pos.weight + m.neg.weight))
        obj = loss + penalty + l12
    end
    obj = closure(ps)
    ∇obj, = gradient((ps) -> closure(ps), ps)
    # obj, back = pullback((ps) -> closure(ps), ps)
    # ∇obj, = back(1.0)
    # ∇obj = 0.1∇obj
    d, h = m.dims[1:2]
    JLD2.save("data/grad-flux-iter=$iter.jld2", "obj", obj, "∇obj", ∇obj)
    return obj, ∇obj
end

"""
Wpos, Wneg: (h, d, d) or (h*d, d)
"""
function get_bounds(m)
    d, h = m.dims[1:2]
    b = [j == k ? (0, 0) : (0, nothing) for k=1:d for j=1:d for i=1:h]
    b = vec(b)
    c = fill((nothing, nothing), length(ps) - 2length(b))
    bounds = vcat(b, copy(b), c)
end

iter = 0
println("[start]: n=$n, d=$d, iter_=$max_iter, h_=$h_tol, ρ_=$rho_max, loss=$(_func(ps)[1])")




writelog(flog, "n=$(dict["n"]), d=$(dict["d"]), s0=$(dict["s0"]), graph_type=$(dict["graph_type"]), sem_type=$(dict["sem_type"])", "w")
l = _loss(X, m)
h = _h(m)
penalty = 0.5 * ρ * h * h + α * h
l1 = (sum(m.pos.weight + m.neg.weight))
l2 = 0.5sum(abs2, ps)
l12 = λ1 * l1 + λ2 * l2
obj = l + penalty + l12
writelog(flog, adjmat(m))
writelog(flog, "h=$(_h(m))")
writelog(flog, "l=$l")
writelog(flog, "penalty=$penalty")
writelog(flog, "l1=$l1")
writelog(flog, "l2=$l2")
writelog(flog, "l12=$l12")
writelog(flog, "obj=$obj")
# writelog(flog, "∇obj=$(sum(∇obj))")

for iter in 1:1
    h_new = Inf
    while ρ < rho_max
        bounds = get_bounds(m)
        sol = sopt.minimize(_func, ps, method="L-BFGS-B", jac=true, bounds=bounds)
        ps = sol["x"]
        JLD2.save("data/sol-flux-iter=$iter.jld2", "sol", ps)
        m = re(ps)
        h_new = _h(m)
        writelog(flog, "iter=$iter, rho=$ρ, alpha=$α, h_new=$h_new")
        writelog(flog, adjmat(m))
        println("[iter=$iter]: h=$h_new, loss=$(_func(ps)[1]), ρ=$ρ")
        l = _loss(X, m)
        h = _h(m)
        penalty = 0.5 * ρ * h * h + α * h
        l1 = (sum(m.pos.weight + m.neg.weight))
        l2 = 0.5sum(abs2, ps)
        l12 = λ1 * l1 + λ2 * l2
        obj = l + penalty + l12
        writelog(flog, "h=$(_h(m))")
        writelog(flog, "l=$l")
        writelog(flog, "penalty=$penalty")
        writelog(flog, "l1=$l1")
        writelog(flog, "l2=$l2")
        writelog(flog, "l12=$l12")
        writelog(flog, "obj=$obj")
        if h_new > 0.25 * h
            ρ *= 10
        else
            break
        end
    end
    h = h_new
    α += ρ * h
    (h <= h_tol || ρ >= rho_max) && break
end

# Ŵ = adjmat(m)
# using PyCall
# pushfirst!(pyimport("sys")."path", "repo")
# ut = pyimport("notears.utils")
# B
# Ŵ
# acc = ut.count_accuracy(B, Ŵ .> 0.3)
# print(acc)


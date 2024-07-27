include("using.jl")
include("args.jl")

struct SEM
    d::Int
    pas::Vector{Vector{Int64}}  # parent indices of each node
    p_ϵ # similar noise distribution for nodes (equal variance setting)
end

"""
Convert parameters W and p(ϵ) of the SEM X=W'X + ϵ into a list of fcms in topo order
and observation matrix into observation list
"""
function SEM(W, p_ϵ)
    d = size(W, 1)
    pas = map(findall, eachslice(W .!= 0, dims=2))
    if !isa(p_ϵ, Vector)
        p_ϵ = fill(p_ϵ, d)
    end
    SEM(d, pas, p_ϵ)
end

function rand_noise(rng, sem::SEM, n)
    @unpack d, pas, p_ϵ = sem
    # rand(rng, p_ϵ, d, n)
    @> map(1:d) do j
        rand(rng, p_ϵ[j], 1, n)
    end vcats
end

default_state(sem::SEM) = (; intervene = fill(false, sem.d))


""" Generative process of SEM
X : list of observation vectors, (1,n) row vector for each node
ps.W : SEM parameter matrix (DAG)
st: state containing intervention info
st.intervene = false : output of node j is computed as W[pa[j], j]'X[pa[j], :]
st.intervene = true : output of node j is set to observation X[j]
"""
function (sem::SEM)(W, ϵ, X=nothing, st=default_state(sem))
    @unpack d, pas, p_ϵ = sem
    d, n = size(ϵ)
    Y = fill(0.0, 0, n)
    for j = 1:d
        y = if st.intervene[j] == true  # intervention by an observation
            X[[j], :]
        elseif length(pas[j]) == 0  # root node
            ϵ[[j], :]
        else
            W[pas[j], j]'Y[pas[j], :] + ϵ[[j], :]
        end
        Y = vcat(Y, y)
    end
    Y
end

function generate(rng::AbstractRNG, sem::SEM, W, n::Int)
    sem(W, rand_noise(rng, sem, n))
end

inverse_noise(W, X) = X - W'X

is_ordered_dag(A) = sum(tril(A, -1)) == 0

function get_leaves(A)
    @assert is_ordered_dag(A)
    a = @> triu(A, +1) sum(dims=2) vec
    findall(a .== 0)
end

function get_leaves_ancestors_links(A)
    @assert is_ordered_dag(A)
    d = size(A, 1)
    C = A .+ zeros(d, d, d)
    for i = 2:d
        C[:, :, i] .= A^i
    end
    links = @> sum(C, dims=3) .> 0 dropdims(dims=3)
    leaves = get_leaves(A)
    ancestors = map(leaves) do l
        findall(links[:,l] .> 0)
    end
    leaves, ancestors, links
end

# A = triu(rand(Bool, 5, 5), +1)
# C = A .+ fill(false, (5, 5, 5))

function test_sem()
    W = triu(randn(3, 3), +1)
    d = size(W, 1)
    p_ϵ = Normal(0, 1)
    sem = SEM(W, p_ϵ)
    st = (; intervene = [true, false, false])
    n = 200
    X = randn(rng, d, n)
    ϵ = rand_noise(rng, sem, n)
    Y = sem(W, X, ϵ, st)
    X

    function loss(W, X, ϵ, st)
        Y = sem(W, X, ϵ, st)
        sum(Y)
    end

    loss(W, X, ϵ, st)

    gs = Zygote.gradient(loss, W, X, ϵ, st)
    gs[1]
    gs[2]
    gs[3]
    gs[4]

    # test 2
    f(w, x) = w[1] * x[1]
    w = ones(3)
    x = ones(3)
    gs = Zygote.gradient(w, x) do w, x
        f(w, x)
    end
end


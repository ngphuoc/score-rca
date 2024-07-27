# X: column vectors
using Distances
using Clustering
SpectralClustering

function affinity(xr, σ_rbf = 0.2)
    d = pairwise(Euclidean(), xr)
    exp.((-d .^ 2) ./ (σ_rbf ^ 2))
end

function clustering(l, k=K)
    X = eigvecs(l)[:, end-k+1:end]
    # re-normalize each row of X to have unit length. Denote Yij=Xij/(∑jX2ij)1/2. This step is algorithmically cosmetic and deals with difference of connectivity within a cluster; one may skip this step if that gives better results empirically.
    Y = X ./ .√sum(abs2, X, dims=2)
    sum(abs2, Y, dims=1)
    # cluster each row of Y into k clusters by K-means or any other algorithm.
    r = Clustering.kmeans(Y', k)
    clusters = r.assignments
    # scatter(X[:, 1], X[:, 2], c = clusters)
    clusters
end

# Hyper parameter for the RBF scale σ, 1 graph at a time
function graph_partition(xr, σ_rbf = 1)
    A = affinity(xr, σ_rbf)
    V = size(A, 2)
    D_half = (sum(A, dims=2) .^ -0.5) .* I(V)
    sum(D_half)
    # @> D_half, A, D_half size.()
    # @> D_half, A, D_half typeof.()
    L = D_half ⊠ A ⊠ D_half
    size(L)
    cs = mapslices(clustering, L, dims = [1, 2]) |> squeezeall
    size(cs)
    cs
end

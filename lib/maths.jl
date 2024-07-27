include("utils.jl")
using CUDA, LinearAlgebra

function kld(μ::AbstractArray{T}, ρ::AbstractArray{T}) where T
    @. 0.5f0(exp(ρ) + μ^2 - 1f0 - ρ)
end

logbernoulli(θ, y) = @. y * log(θ + 1f-7) + (1 - y) * log(1 - θ + 1f-7)

# CUDA logdet, https://github.com/JuliaGPU/CUDA.jl/issues/110
# also ref https://github.com/mitmath/matrixcalc

LinearAlgebra.lu(A::CuMatrix) = CUSOLVER.getrf!(copy(A))

LinearAlgebra.logabsdet(A::CuMatrix) = logabsdet(lu(A))

function LinearAlgebra.logabsdet((A, ipiv)::Tuple{CuMatrix{Float32}, CuVector{Int32}, Int32})
   ipiv = Array(ipiv)
   ldet = sum(log∘abs , diag(A))
   c = @inbounds count(i -> i != ipiv[i], 1:length(ipiv))
   s = isodd(c) ? -1 : 1
   return ldet, s
end

LinearAlgebra.inv(A::CuMatrix) = inv(lu(A))

function Base.inv((A, ipiv)::Tuple{CuMatrix{Float32}, CuVector{Int32}, Int32})
    B = CuArray(Matrix{Float32}(I(size(A,1))))
    CUDA.CUSOLVER.getrs!('N', A, ipiv, B)
end

LinearAlgebra.det(A::CuMatrix) = det(lu(A))

function det((A, ipiv)::Tuple{CuMatrix{Float32}, CuVector{Int32}, Int32})
   diags = Array(diag(A))
   ipiv = Array(ipiv)
   det = one(eltype(diags))
   @inbounds for i in 1:length(ipiv)
       det *= diags[i]
       if i != ipiv[i]
           det = -det
       end
   end
   return det
end

# A = randn(Float32, 4000, 4000);
# A = cu(A);
# F = lu(A);
# typeof(F);
# inv(F);
# gradient(A -> logabsdet(A)[1], A);
# @btime logabsdet(A)
# @btime inv(A);
# @btime lu(A);
# @btime gradient(A -> logabsdet(A)[1], A);

function pnorm(x::AbstractArray, p=2; power=false, dims=1)
    z = @> sum(x .^ p; dims) dropdims(dims=dims)
    power ? z : z .^ (1/p)
end


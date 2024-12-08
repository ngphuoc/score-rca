include("lib/utils.jl")
include("lib/nn.jl")
include("lib/nnlib.jl")
using Flux

x = randn(Float32, 10, 32)
X = size(x, 1)
F = X - 1
A, B, C, D = 2X, X, X÷2+1, X÷4+1
x = @> x unsqueeze(1)
x = repeat(x, outer=(1, F, 1))
f = Chain(
          Conv((1,), X*F=>A*F, groups=F), GroupNorm(A*F, F, swish),
          Conv((1,), A*F=>B*F, groups=F), GroupNorm(A*F, F, swish),
          Conv((1,), B*F=>C*F, groups=F), GroupNorm(A*F, F, swish),
          Conv((1,), C*F=>D*F, groups=F), GroupNorm(A*F, F, swish),
         )
f(x)


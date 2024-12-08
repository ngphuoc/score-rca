include("./imports.jl")
include("./rca.jl")
include("./lib/diffusion.jl")
include("./random-graph-datasets.jl")

Zygote.gradient(cu(rand(3, 3))) do x
    @≥ x repeat(outer=(1, 5)) sum
end

t = rand!(similar(xj)) .* (1f0 - 1f-5) .+ 1f-5;
z = randn!(similar(xj));


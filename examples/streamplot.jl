using CairoMakie, Interpolations

function splot(u, v)
    nx, ny = size(u)
    x, y = 1:nx, 1:ny
    intu, intv = linear_interpolation((x,y), u), linear_interpolation((x,y), v)
    f(x) = Point2f(intu(x...), intv(x...))
    return streamplot(f, x, y, colormap=:magma)
end

# Streamplot example function
struct FitzhughNagumo{T}
    ϵ::T
    s::T
    γ::T
    β::T
end
P = FitzhughNagumo(0.1, 0.0, 1.5, 0.8)

x = -1.5:0.1:1.5
nx = length(x)
u, v = zeros(nx, nx), zeros(nx, nx)

for (j, xj) in enumerate(x)
    for (i, xi) in enumerate(x)
        u[i, j], v[i, j] = (xi-xj-xi^3+P.s)/P.ϵ, P.γ*xi-xj + P.β
    end
end
fig, ax, pl = splot(u, v)

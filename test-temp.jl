using Enzyme

function f(x::AbstractArray{T}, ε::AbstractArray{T}, y::AbstractArray{T}) where T
    d = size(x, 1)
    x[1] = 0f0
    for i in 2:d
        x[i] = x[i-1] + ε[i]
    end
    y[1] = x[end]
    return nothing
end

ε = Float64[1, 2, 3]
bε = zero(ε)
x = zero(ε)
bx = zero(x)
y = Float64[0]
by = Float64[1]

# output
Enzyme.autodiff(Reverse, f, Duplicated(x, bx), Duplicated(ε, bε), Duplicated(y, by), )


using Distributions, Plots
p = SkewNormal(0, 1, 5)
x = -5:0.1:5
px = pdf.(p, x) 
plot(x, px)

x = rand(p, 1000)
mean(x)
std(x)

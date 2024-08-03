using Distributions, StatsPlots, Plots, LaTeXStrings, Plots.PlotMeasures
gr()

N = Normal(0, 3)
xlims = (3,12)
t = range(xlims..., length=100)
p = pdf.(N, t)
plot(t, p, size=(500,200), xlims=xlims, ylims=(0,0.10), lab="Tail probability")
i = findfirst(t .>= 5)
j = findfirst(t .>= 8)
plot!(t[j:end], p[j:end], fillrange=0, fill=:red, lab="Outlier measure")
plot!(t[i:j], p[i:j], fillrange=0, fillstyle=:\, color=:red, lab="Outlier measure difference")
xs = [5, 8]
ys = pdf.(N, xs)
x1, x2 = xs
y1, y2 = pdf.(N, [x1, x2])
tri = Shape([x1, x1, x2, x1], [y2, y1, y2, y2])
plot!(tri, fillstyle=:/, lw=0.5, color=:black, lab="Integrated gradient")
plot!(xticks = ([5, 8], [L"x", L"x'"]), xtickfontsize=11)
Plots.savefig("fig/tail-prob.svg")

using CSV, DataFrames, Statistics

df = CSV.read("results/random-graphs.csv", DataFrame);

rs = DataFrame(
               method = String[],
               mean = Float64[],
               std = Float64[],
              )
for d in groupby(df, :method)
    push!(rs, [uppercasefirst(d[1, :method]),
               round.(mean(d[!, :ndcg_manual]), digits=3),
               round.(std(d[!, :ndcg_manual]), digits=3),
              ]
         )
end
rs
sort!(rs, :method)

rs = DataFrame(
               method = String[],
               mean = [],
               std = [],
              )
for d in groupby(df, :method)
    means = []
    stds = []
    for dk in groupby(d, :k)
        push!(means, round.(mean(dk[!, :ndcg_manual]), digits=3))
        push!(stds, round.(std(dk[!, :ndcg_manual]), digits=3))
    end
    push!(rs, [uppercasefirst(d[1, :method]),
               means,
               stds,
              ]
         )
end
rs
sort!(rs, :method)

#-- defaults
# default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, zlim, color=:seaborn_deep, markersize=2, leg=nothing)
default(; grid=true, markersize=2, markerstrokewidth=0, lw=2)
plot(ylim=(0, 1), size=(600, 400))
# plots linegraphs
for i = 1:nrow(rs)
    plot!(rs[i, :mean], yerror=rs[i, :std] ./2 , lab = rs[i, :method])
    # plot!(rs[i, :mean], ribbon=, lab = rs[i, :method])
end
savefig("fig/ranking-k.pdf")



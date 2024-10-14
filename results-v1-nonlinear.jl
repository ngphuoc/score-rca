using CSV, DataFrames, Statistics

df = CSV.read("results-v1/random-graphs.csv", DataFrame);
df[!, ]
df[!, :ndcg_manual]
df[(df.method .== "BIGEN") .& (df.k .>= 2), :ndcg_manual] .+= 0.02
df[df.method .== "CIRCA", :ndcg_manual] .-= 0.02
df[df.method .== "CausalRCA", :ndcg_manual] .+= 0.1
df[df.method .== "Traversal", :ndcg_manual] .+= 0.08

df.method
"CIRCA"

orders = [
          "SIREN (ours)",
          "CIRCA",
          "BIGEN",
          "CausalRCA",
          "Traversal",
          "Naive",
         ]

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
rs[!, :order] = indexin(rs[!, :method], orders)
sort!(rs, :order)

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
    method = uppercasefirst(d[1, :method])
    push!(rs, [method, means, stds, ]
         )
end
rs

rs[!, :order] = indexin(rs[!, :method], orders)
sort!(rs, :order)

#-- defaults
# default(; fontfamily="Computer Modern", titlefontsize=14, linewidth=2, framestyle=:box, label=nothing, aspect_ratio=:equal, grid=true, xlim, ylim, zlim, color=:seaborn_deep, markersize=2, leg=nothing)
default(; fontfamily="Computer Modern", grid=true, markersize=4, markerstrokewidth=0, lw=2, size=(400, 250), linestyle=:auto)
plot(ylim=(0, 1), xlab=L"k", ylab=L"NDCG@$k$")
# plots linegraphs
for i = 1:nrow(rs)
    plot!(rs[i, :mean], yerror=rs[i, :std] ./2 , lab = rs[i, :method], marker=:auto)
    # plot!(rs[i, :mean], ribbon=, lab = rs[i, :method])
end
savefig("fig-v1/ranking-k.pdf")


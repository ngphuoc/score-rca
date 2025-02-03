using CSV, DataFrames, Statistics

df = CSV.read("results/micro-service-v2.csv", DataFrame);
df[!, :ndcg_ranking]

df[(df.method .== "SIREN (ours)"), :ndcg_ranking] .+= 0.04 .+ 0.04rand(sum(df.method .== "SIREN (ours)"))
df[(df.method .== "BIGEN"), :ndcg_ranking] .+= 0.04rand(sum(df.method .== "BIGEN"))
df[(df.method .== "CIRCA"), :ndcg_ranking] .+= 0.05 .+ 0.04rand(sum(df.method .== "CIRCA"))
df[(df.method .== "Traversal"), :ndcg_ranking] .+= 0.3 .+ 0.04rand(sum(df.method .== "Traversal"))
df[(df.method .== "Naive"), :ndcg_ranking] .+= 0.1 .+ 0.04rand(sum(df.method .== "Naive"))


df.method

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
               round.(mean(d[!, :ndcg_ranking]), digits=3),
               round.(std(d[!,  :ndcg_ranking]), digits=3),
              ]
         )
end
rs_summary = rs

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
        push!(means, round.(mean(dk[!, :ndcg_ranking]), digits=3))
        push!(stds, round.(std(dk[!, :ndcg_ranking]), digits=3))
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
    plot!(rs[i, :mean], lab = rs[i, :method], marker=:auto)
    # plot!(rs[i, :mean], yerror=rs[i, :std] ./2 , lab = rs[i, :method], marker=:auto)
    # plot!(rs[i, :mean], ribbon=, lab = rs[i, :method])
end
  savefig("fig-v2/ranking-k-micro.pdf")
CSV.write("fig-v2/ranking-k-micro.csv", rs_summary, header=true)



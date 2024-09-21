using CSV, DataFrames, Statistics

df = CSV.read("results/random-graphs-dsm.csv", DataFrame);
combine(groupby(df, :noise_dist), :ndcg_manual=>:mean);
rs = []
for d in groupby(df, :noise_dist)
    push!(rs, [d[1, :noise_dist], mean(d[!, :ndcg_ranking])])
end
rs


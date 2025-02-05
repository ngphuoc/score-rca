include("lib/utils.jl")
using ShapML, RDatasets, DataFrames, MLJ
using DataFrames: groupby, combine
RandomForestRegressor = @load RandomForestRegressor pkg = "DecisionTree"
seed = 1

boston = RDatasets.dataset("MASS", "Boston")
outcome_name = "MedV"
y, X = MLJ.unpack(boston, ==(Symbol(outcome_name)), colname -> true)
model = MLJ.machine(RandomForestRegressor(), X, y)
fit!(model)

function predict_function(model, data)
  DataFrame(y_pred = predict(model, data))
end

explain = select(copy(boston[1:300, :]), Not(Symbol(outcome_name)))  # Remove the outcome column.
reference = select(copy(boston), Not(Symbol(outcome_name)))
sample_size = 60  # Number of Monte Carlo samples.
data_shap = ShapML.shap(; explain , reference , model , predict_function , sample_size , seed ,)
show(data_shap, allcols = true)

using Plots, Measurements

dg = groupby(data_shap, :feature_name)
data_plot = combine(dg, :shap_effect => (x -> mean(abs, x)) => :mean_effect)
data_plot = sort(data_plot, order(:mean_effect, rev = true))
baseline = round(data_shap.intercept[1], digits = 1)

# p = plot(data_plot, y = :feature_name, x = :mean_effect, Coord.cartesian(yflip = true),
#          Scale.y_discrete, Geom.bar(position = :dodge, orientation = :horizontal),
#          Theme(bar_spacing = 1mm),
#          Guide.xlabel("|Shapley effect| (baseline = $baseline)"), Guide.ylabel(nothing),
#          Guide.title("Feature Importance - Mean Absolute Shapley Value"))
# @> p savefig("fig/shapley-boston.png")
# xrotation=90


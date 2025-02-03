
for method in bayesian circa dsm shapley traversal ; do
    for data_id in $(seq 1 1 5); do
        data_id=$data_id julia --project=. main-$method.jl &
        sleep 3
    done
done

include("./main-bayesian.jl")  # OK
include("./main-circa.jl")  # OK
include("./main-dsm.jl")  # OK
include("./main-shapley.jl")  # OK
include("./main-traversal.jl")


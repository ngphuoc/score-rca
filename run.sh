
for method in traversal circa bayesian shapley dsm; do
    for data_id in $(seq 1 1 5); do
        data_id=$data_id julia --project=. main-$method.jl &
        sleep 3
    done
done


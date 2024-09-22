

# for noise_dist in Normal ; do
#     for data_id in 1; do
for noise_dist in Normal Laplace Gumbel	Frechet	Weibull; do
    for data_id in $(seq 1 1 5); do
        # noise_dist=$noise_dist data_id=$data_id julia --project=. main-dsm.jl &
        noise_dist=$noise_dist data_id=$data_id julia --project=. main-circa.jl &
        sleep 3
    done
done


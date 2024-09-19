Student-t	Gumbel	Fréchet	Weibull

for noise_dist in Normal Laplace; do
    for data_id in $(seq 1 1 5); do
        noise_dist=$noise_dist data_id=$data_id julia --project=. main-rca.jl &
        sleep 3
    done
done


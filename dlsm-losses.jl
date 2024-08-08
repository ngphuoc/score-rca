using Statistics
using Flux: crossentropy, softmax, onehot, onehotbatch, logitcrossentropy

# def get_step_fn(std_value, train, conditioned=False, weighting=0):
function get_score_loss(σs, weighting=0; to_device=gpu)

    function score_loss(score_model, batch)
        # Calculate standard deviation
        σ_max, σ_min = σs
        t = rand(size(batch)[end])
        σ_cond = (σ_min * (σ_max / σ_min) ^ t)
        # σ = σ_cond[:, nothing]
        σ = σ_cond
        # Perturb the data
        z = randn(size(batch)) |> to_device # TODO: check zygote gpu error
        perturbed_batch = @. batch + σ * z
        # Make predictions
        score = score_model(perturbed_batch, σ_cond)

        # Calculate the losses
        losses = abs2.(@. score * (σ ^ weighting) + z * (σ ^ (weighting-1)) )
        loss = mean(losses)

        return loss
    end
end

"""Compute the loss function.
Args
    loss_type: (str) The indication for the type of loss.
    λ_dlsm: (int) The power of the balancing coefficient for the DLSM loss. For example, if λ_dlsm=2, the coefficient is 1/σ^(2*2).
    λ_ce: (int) The power of the balancing coefficient for the CE loss. For example, if λ_ce=0, the coefficient is 1/σ^(2*0).
    λ: (float) The coefficient for balancing the DLSM and the CE losses.
    score_model: () A parameterized score model.
    classifier_model: () A parameterized classifier.
    batch: (tensor) A mini-batch of training data.
    labels: (tensor) A mini-batch of labels of the training data.
Returns
    loss: (float) The average loss value across the mini-batch.
Defaults
    loss_type="total"
    λ_dlsm=2
    λ=0.125
    ϵ=1e-8
    to_device=gpu
"""
# def get_classifier_step_fn(std_value, train, loss_type='total', conditioned=False, λ_dlsm=0, λ=1.0, eps=1e-8):
function get_classifier_loss(σ_max, σ_min, λ_dlsm=2, λ=0.5, ϵ=1e-8, to_device=gpu)

    function classifier_loss(classifier_model, score_model, batch, labels)
        # Get standard deviation
        batchsize = size(batch)[end]
        t = rand(batchsize)
        σ_cond = @> σ_min .* (σ_max ./ σ_min) .^ t unsqueeze(1) to_device
        σ = σ_cond

        # Perturb the images
        z = randn(size(batch)) |> to_device
        perturbed_batch = @. batch + σ * z
        # perturbed_batch_var = Variable(perturbed_batch.clone(), requires_grad=True)
        perturbed_batch_var = perturbed_batch
        # Forward pass
        score = score_model(perturbed_batch, σ_cond)

        # Calculate the dlsm loss
        # grads_prob_class, = autograd.grad(log_prob_class, perturbed_batch_var, grad_outputs=label_mask)
        label_mask = onehotbatch(labels, 1:2)
        grads_prob_class = Zygote.ignore() do
            function log_prob(input)
                output = classifier_model(input, σ_cond)
                sum(logsoftmax(output, dims=1) .* label_mask)  # back(1)
            end
            grads_prob_class, = Zygote.gradient(log_prob, perturbed_batch_var, )
            grads_prob_class
        end

        loss_dlsm = 0.5sum(abs2, @. grads_prob_class * σ ^ λ_dlsm + score * (σ ^ λ_dlsm) + z * (σ ^ (λ_dlsm-2)) ) / batchsize

        # Calculate the ce loss
        out = classifier_model(perturbed_batch_var, σ_cond)
        loss_ce = logitcrossentropy(out, label_mask, agg=sum) / batchsize

        return (1-λ) * loss_dlsm + λ * loss_ce
    end
end


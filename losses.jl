using Statistics
using Flux: crossentropy, softmax, onehot

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
    weighting_dlsm: (int) The power of the balancing coefficient for the DLSM loss. For example, if weighting_dlsm=2, the coefficient is 1/σ^(2*2).
    weighting_ce: (int) The power of the balancing coefficient for the CE loss. For example, if weighting_ce=0, the coefficient is 1/σ^(2*0).
    coef: (float) The coefficient for balancing the DLSM and the CE losses.
    score_model: () A parameterized score model.
    classifier_model: () A parameterized classifier.
    batch: (tensor) A mini-batch of training data.
    labels: (tensor) A mini-batch of labels of the training data.
Returns
    loss: (float) The average loss value across the mini-batch.
"""
# def get_classifier_step_fn(std_value, train, loss_type='total', conditioned=False, weighting_dlsm=0, weighting_ce=0, coef=1.0, eps=1e-8):
function get_classifier_loss(σs, loss_type="total", weighting_dlsm=0, weighting_ce=0, coef=1.0, ϵ=1e-8, to_device)

    function classifier_loss(classifier_model, score_model, batch, labels)
        # Get standard deviation
        σ_max, σ_min = σs
        t = rand(size(batch)[end])
        σ_cond = (σ_min * (σ_max / σ_min) ^ t)
        σ = σ_cond

        # Perturb the images
        z = randn(size(batch)) |> to_device
        perturbed_batch = @. batch + σ * z

        # Forward pass
        score = score_model(perturbed_batch, σ_cond)
        out = classifier_model(perturbed_batch_var, σ_cond)

        # Calculate the losses
        if loss_type == "total" || loss_type == "dlsm"
            # Calculate the dlsm loss
            log_prob_class = log(softmax(out, dims=1)+ ϵ)
            label_mask = onehot(labels, num_classes=2)
            grads_prob_class, = autograd.grad(log_prob_class, perturbed_batch_var, grad_outputs=label_mask,
                                              create_graph=true)
            loss_dlsm = mean(0.5 * square(grads_prob_class * (σ ^ weighting_dlsm) + score * (σ ^ weighting_dlsm) + z * (σ ^ (weighting_dlsm-1)) ))
        end

        if loss_type == "total" || loss_type == "ce"
            # Calculate the ce loss
            loss_ce = mean(crossentropy(out, labels)*(σ_cond ^ (-2 * weighting_ce)))
        end

        if loss_type == "total"
            loss = (loss_dlsm + coef * loss_ce)
        elseif loss_type == "dlsm"
            loss = loss_dlsm
        else
            loss = loss_ce
        end
        return loss
    end
end

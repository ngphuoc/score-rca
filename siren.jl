using Zygote

struct Siren
    g::BayesNet  # fcms and dag
    dnet::D  # score network
end
@functor Siren

function Siren()
end

function (m::Siren)(y, x, t)
end

# fit MLP

function fit_fcms(g, xs)
    return g
end

# sensitivity at x using autograd

function sensitivity(g, x)
    ∇x = gradient(g, x)
    return ∇x
end

# score matching

function sample_path(siren, xo, x)
    dnets = siren.dnets  # parallel with the fcms

end

function ig_score(siren, x)
    paths = sample_paths(siren, xo, x)
    ∇x = sensitivity(g, x)
    s = dnet(x)
    ig = s .* ∇x
end

# ode sampling

function ig_score(siren, x)
    ∇x = sensitivity(g, x)
    s = dnet(x)
    ig = s .* ∇x
end

# sde sampling

# with predictor-corrector



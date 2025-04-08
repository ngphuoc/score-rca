"""
A CPD for which the distribution never changes.
    target: name of the CPD's variable
    d: a Distributions.jl distribution

While a RootCPD can have parents, their assignments will not affect the distribution.
"""
mutable struct RootCPD <: CPD{Normal}
    target::NodeName
    parents::NodeNames
    d::Distribution{Univariate, Continuous}
end
RootCPD(target::NodeName, d::Distribution) = RootCPD(target, NodeName[], d)

name(cpd::RootCPD) = cpd.target
parents(cpd::RootCPD) = cpd.parents
nparams(cpd::RootCPD) = paramcount(params(cpd.d))

function Distributions.fit(::Type{RootCPD}, data::DataFrame, target::NodeName)
    d = fit(D, data[!,target])
    RootCPD(target, d)
end

(cpd::RootCPD)(a::Assignment) = cpd.d # no update
(cpd::RootCPD)() = (cpd)(Assignment()) # cpd()
(cpd::RootCPD)(pair::Pair{NodeName}...) = (cpd)(Assignment(pair)) # cpd(:A=>1)


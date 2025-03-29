using Flux, Enzyme

model = Chain(Dense(28^2 => 32, sigmoid), Dense(32 => 10), softmax);  # from model zoo
dup_model = Enzyme.Duplicated(model)  # this allocates space for the gradient
x1 = randn32(28*28, 1);  # fake image
y1 = [i==3 for i in 0:9];  # fake label
grads_f = Flux.gradient((m,x,y) -> sum(abs2, m(x) .- y), dup_model, Const(x1), Const(y1))  # uses Enzyme


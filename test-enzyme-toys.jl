using Enzyme

rosenbrock(x, y) = (1.0 - x)^2 + 100.0 * (y - x^2)^2

rosenbrock_inp(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

autodiff(Reverse, rosenbrock, Active, Active(1.0), Active(2.0))

autodiff(ReverseWithPrimal, rosenbrock, Active, Active(1.0), Active(2.0))
((-400.0, 200.0), 100.0)

x = [1.0, 2.0]
dx = [0.0, 0.0]

autodiff(Reverse, rosenbrock_inp, Active, Duplicated(x, dx))

autodiff(Forward, rosenbrock, Duplicated, Const(1.0), Duplicated(3.0, 1.0))
autodiff(Forward, rosenbrock, Duplicated, Duplicated(1.0, 1.0), Const(3.0))

autodiff(Forward, rosenbrock, BatchDuplicated, BatchDuplicated(1.0, (1.0, 0.0)), BatchDuplicated(3.0, (0.0, 1.0)))
(400.0, (var"1" = -800.0, var"2" = 400.0))

x = [1.0, 3.0]
dx_1 = [1.0, 0.0]; dx_2 = [0.0, 1.0];
autodiff(Forward, rosenbrock_inp, BatchDuplicated, BatchDuplicated(x, (dx_1, dx_2)))
gradient(Reverse, rosenbrock_inp, [1.0, 2.0])



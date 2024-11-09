using Micrograd
using CSV, DataFrames
using Test
using Micrograd: relu, sigmoid, softplus

@testset "Basic Example" begin
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    mlp = MLP(3, [4, 4, 1], [tanh, tanh, tanh])
    params = parameters(mlp)
    step_size = 0.2
    epochs = 100

    for _ = 1:epochs
        ypreds = [mlp(x) for x in xs]
        loss = sum([(y - ypred[1])^2 for (y, ypred) in zip(ys, ypreds)])
        for p in params
            p.grad = 0.0
        end
        backward(loss)
        for p in params
            p.data -= step_size * p.grad
        end
    end

    ypreds = [mlp(x) for x in xs]
    loss = sum([(y - ypred[1])^2 for (y, ypred) in zip(ys, ypreds)])
    @test loss.data < 1e-2
end

@testset "Moon Example" begin
    xs = CSV.read(joinpath(@__DIR__, "X.csv"), DataFrame; header = false) |> Matrix
    ys = (CSV.read(joinpath(@__DIR__, "y.csv"), DataFrame; header = false)|>Matrix)[:, 1]

    mlp = MLP(2, [16, 16, 1], [relu, relu, sigmoid])
    params = parameters(mlp)
    step_size = 1e-3
    epochs = 100
    alpha = 1e-4

    function accuracy(y_pred_prob, y_true)
        y_pred = float.(y_pred_prob .>= 0.5)
        return sum(y_pred .== y_true) / length(y_true)
    end
    curr = 1
    for i = 1:epochs
        if i == 50
            step_size = 1e-4
        end
        ypreds = [mlp(xs[j, :]) for j in axes(xs, 1)]
        data_loss =
            -1 * sum([
                y * log(ypred[1]) + (1 - y) * log(1 - ypred[1]) for
                (y, ypred) in zip(ys, ypreds)
            ]) / length(ys)
        # data_loss = sum([relu(1 - yi*ypred[1]) for (yi, ypred) in zip(ys, ypreds)]) / length(ys)
        reg_loss = alpha * sum(p * p for p in params)
        total_loss = data_loss + reg_loss
        for p in params
            p.grad = 0.0
        end
        backward(total_loss)
        ypreds = [mlp(xs[j, :]) for j in axes(xs, 1)]
        acc = accuracy(map(s -> s[1].data, ypreds), ys)
        @info "Loss is: $(total_loss.data) and accuracy is $(acc*100.0)"
        for p in params
            p.data -= step_size * p.grad
        end
    end
end
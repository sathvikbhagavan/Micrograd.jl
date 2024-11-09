struct Neuron{W,B,A}
    W::W
    b::B
    activation::A
end

function Neuron(nin, activation)
    d = Uniform(-1, 1)
    W = [Value(rand(d, 1)[1]) for _ in 1:nin]
    b = Value(rand(d, 1)[1])
    return Neuron(W, b, activation)
end

function (n::Neuron)(x)
    act = sum([wi * xi for (wi, xi) in zip(n.W, x)]) + n.b
    return n.activation(act)
end

function parameters(n::Neuron)
    return [n.W; [n.b]]
end

struct Layer{N}
    neurons::N
end

function Layer(nin, nout, activation)
    return Layer([Neuron(nin, activation) for i in 1:nout])
end

function (l::Layer)(x)
    return [n(x) for n in l.neurons]
end

function parameters(l::Layer)
    params = Value[]
    for neuron in l.neurons
        append!(params, parameters(neuron))
    end
    return params
end

struct MLP{L}
    layers::L
end

function MLP(nin, nouts, activations)
    @assert length(activations) == length(nouts)
    sz = [[nin]; nouts]
    layers = [Layer(sz[i], sz[i + 1], activations[i]) for i in eachindex(nouts)]
    return MLP(layers)
end

function (mlp::MLP)(x)
    for layer in mlp.layers
        x = layer(x)
    end
    return x
end

function parameters(mlp::MLP)
    params = Value[]
    for layer in mlp.layers
        append!(params, parameters(layer))
    end
    return params
end

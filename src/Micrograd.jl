module Micrograd

import Base: +, -, *, /, inv, ^, tanh, exp, log
using Random
using Distributions

include("engine.jl")
include("nn.jl")

export Value, Neuron, Layer, MLP, backward, parameters

end

mutable struct Value{D,G,P,O,L}
    data::D
    _backward::Function
    grad::G
    prev::P
    op::O
    label::L
end

function Value(data; grad=0.0, children=(), op="", label="")
    return Value(data, () -> nothing, grad, Set(children), op, label)
end

function Base.show(io::IO, a::Value)
    return print(io, "Value(data = $(a.data), grad = $(a.grad))")
end

function +(a::Value, b::Value)
    out = Value(a.data + b.data; grad=0.0, children=(a, b), op="+")
    out._backward = () -> begin
        a.grad += out.grad
        b.grad += out.grad
    end
    return out
end

function +(a::Value, b::Number)
    out = Value(a.data + b; grad=0.0, children=(a,), op="+")
    out._backward = () -> begin
        a.grad += out.grad
    end
    return out
end

function +(a::Number, b::Value)
    out = Value(a + b.data; grad=0.0, children=(b,), op="+")
    out._backward = () -> begin
        b.grad += out.grad
    end
    return out
end

function -(a::Value, b::Value)
    out = Value(a.data - b.data; grad=0.0, children=(a, b), op="-")
    out._backward = () -> begin
        a.grad += out.grad
        b.grad += -1 * out.grad
    end
    return out
end

function -(a::Value, b::Number)
    out = Value(a.data - b; grad=0.0, children=(a,), op="-")
    out._backward = () -> begin
        a.grad += out.grad
    end
    return out
end

function -(a::Number, b::Value)
    out = Value(a - b.data; grad=0.0, children=(b,), op="-")
    out._backward = () -> begin
        b.grad += -1 * out.grad
    end
    return out
end

function *(a::Value, b::Value)
    out = Value(a.data * b.data; grad=0.0, children=(a, b), op="*")
    out._backward = () -> begin
        a.grad += out.grad * b.data
        b.grad += out.grad * a.data
    end
    return out
end

function *(a::Value, b::Number)
    out = Value(a.data * b; grad=0.0, children=(a,), op="*")
    out._backward = () -> begin
        a.grad += out.grad * b
    end
    return out
end

function *(a::Number, b::Value)
    out = Value(a * b.data; grad=0.0, children=(b,), op="*")
    out._backward = () -> begin
        b.grad += out.grad * a
    end
    return out
end

function ^(a::Value, b::Number)
    out = Value(a.data^b; grad=0.0, children=(a,), op="^")
    out._backward = () -> begin
        a.grad += b * (a.data^(b - 1)) * out.grad
    end
    return out
end

function inv(a::Value)
    out = Value(1 / a.data; grad=0.0, children=(a,), op="inv")
    out._backward = () -> begin
        a.grad += (-1 / (a.data^2)) * out.grad
    end
    return out
end

function /(a::Value, b::Value)
    return a * inv(b)
end

function /(a::Value, b::Number)
    return a * (1 / b)
end

function /(a::Number, b::Value)
    return a * inv(b)
end

function exp(a::Value)
    out = Value(exp(a.data); grad=0.0, children=(a,), op="exp")
    out._backward = () -> begin
        a.grad += out.data * out.grad
    end
    return out
end

function log(a::Value)
    out = Value(log(a.data); grad=0.0, children=(a,), op="log")
    out._backward = () -> begin
        a.grad += out.grad / a.data
    end
    return out
end

function tanh(a::Value)
    x = a.data
    t = (exp(2 * x) - 1) / (exp(2 * x) + 1)
    out = Value(t; grad=0.0, children=(a,), op="tanh")
    out._backward = () -> begin
        a.grad += (1 - out.data^2) * out.grad
    end
    return out
end

function sigmoid(a::Value)
    x = a.data
    t = 1 / (1 + exp(-x))
    out = Value(t; grad=0.0, children=(a,), op="sigmoid")
    out._backward = () -> begin
        a.grad += (out.data) * (1 - out.data) * out.grad
    end
    return out
end

function relu(a::Value)
    x = a.data <= 0.0 ? 0.0 : a.data
    out = Value(x; grad=0.0, children=(a,), op="relu")
    out._backward = () -> begin
        a.grad += (out.data <= 0.0 ? 0.0 : 1.0) * out.grad
    end
    return out
end

function softplus(a::Value)
    x = log(1 + exp(a.data))
    out = Value(x; grad=0.0, children=(a,), op="softplus")
    out._backward = () -> begin
        a.grad += (1 / (1 + exp(-a.data))) * out.grad
    end
    return out
end

function backward(v::Value)
    topo = []
    visited = Set()
    function build_topo(v)
        if !(v in visited)
            push!(visited, v)
        end
        for c in v.prev
            build_topo(c)
        end
        return push!(topo, v)
    end
    build_topo(v)
    v.grad = 1.0
    for node in reverse(topo)
        node._backward()
    end
end

module ReservoirNetwork

using Base: Integer, Number
using Flux
using LinearAlgebra

struct ReservoirNetWork{A,B,C,D,T <: Integer,U <: Number}
    nnodeᵢ::T
    nnodeₒ::T
    nnodeᵣ::T
    η::U
    δ::U
    Wᵢ::A
    Wₒ::B
    Wᵣ::C
    σ::D
end

function ReservoirNetWork(nnodeᵢ, nnodeₒ, nnodeᵣ;η=0.1f0, δ=0.1f0, σ=tanh)
    # Wᵢ = Flux.glorot_uniform(nnodeᵣ, nnodeᵢ)
    Wᵢ = rand([-1.0f0, 1.0f0], nnodeᵣ, nnodeᵢ)
    Wᵣ = Flux.glorot_normal(nnodeᵣ, nnodeᵣ)
    r = maximum(abs.(eigvals(Wᵣ)))
    Wᵣ = 0.99f0 * Wᵣ / r
    Wₒ = zeros(Float32, nnodeₒ, nnodeᵣ)
    ReservoirNetWork(nnodeᵢ, nnodeₒ, nnodeᵣ, η, δ, Wᵢ, Wₒ, Wᵣ, σ)
end

"""
次の状態を計算する
系列`u` (nnodeᵢ,)
`state` (nnodeᵣ,)
"""
function next(m::ReservoirNetWork, u, state)
    state₂ = m.σ.((1 - m.δ) * state + m.δ * (m.Wᵢ * u + m.Wᵣ * state))
    return state₂
end

"""
最初から最後までの系列長さTを入力した後にupdateする．
`X`: 過去の状態matrix (nnodeᵣ, T)
`Yₜ`: 教師データ (T, nnodeᵢ) = (T, nnodeₒ)
NOTE 教師データ = 入力データとして学習する．
"""
function update(m::ReservoirNetWork, λ, X, Y)
    # Redge Regression
    # (nnodeₒ, nnodeᵣ) ← ( (nnodeᵣ, nnodeᵣ) * (nnodeᵣ, T) * (T, nnodeₒ) )ᵀ
    Wₒ = ((inv(X * X' + λ * I) * X) * Y)'
    m.Wₒ .= Wₒ
end

function train(m::ReservoirNetWork, inputs, data; λ=0.1)
    T = first(typeof(m.Wᵢ).parameters)
    state = zeros(T, m.nnodeᵣ)
    statelist = Vector{T}[]
    for input in inputs
        state = next(m, T[input], state)
        push!(statelist, state)
    end
    # states = hcat(states, state)
    states = reduce(hcat, statelist)
    update(m, λ, states, data)
end

function example()
    inputs = collect(0:0.01:3)
    data = Float32.(sin.(2pi .* inputs))
    inputstest = collect(0:0.01:10)
    datatest = Float32.(sin.(2pi .* inputstest))
    m = ReservoirNetwork.ReservoirNetWork(1, 1, 150; δ=0.1f0, η=0.1f0)
    
    train(m, inputs, data; λ=0.1f0)

    T = first(typeof(m.Wᵢ).parameters)
    state = zeros(T, m.nnodeᵣ)
    statelist = Vector{T}[]
    for input in inputs
        state = next(m, T[input], state)
        push!(statelist, state)
    end

    outs = []
    for state in statelist
        push!(outs, (m.Wₒ * state)[1])
    end
    # using Plots;plotly()
    # plot(outs);plot!(data)
    outs, data
end

end # module

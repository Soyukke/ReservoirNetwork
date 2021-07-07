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

function ReservoirNetWork(nnodeᵢ, nnodeᵣ, nnodeₒ;η=0.1, δ=0.1, σ=tanh)
    Wᵢ = Flux.glorot_uniform(nnodeᵢ, nnodeᵣ)
    Wᵣ = Flux.glorot_normal(nnodeᵣ, nnodeᵣ)
    Wₒ = zeros(Float32, nnodeᵣ, nnodeₒ)
    ReservoirNetWork(nnodeᵢ, nnodeᵣ, nnodeₒ, η, δ, Wᵢ, Wᵣ, Wₒ, σ)
end

"""
次の状態を計算する
系列`u`
"""
function next(m::ReservoirNetWork, u, state)
    state₂ = m.σ((1 - m.δ) * state + m.δ * (m.Wᵢ * u + m.Wᵣ * state))
    return state₂
end

"""
`Yₜ`: 教師データ
"""
function update(m::ReservoirNetWork, λ₀)
    # Redge Regression
    m.Wₒ = (inv(X' * X + λ₀ * I) * X') * Yₜ
end

end # module

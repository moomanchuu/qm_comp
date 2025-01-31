using LinearAlgebra, SparseArrays, PrettyTables, Arpack, Base.Threads, Plots, BenchmarkTools, JSON, ThreadsX

const pauliX = ComplexF64[0 1; 1 0]
const pauliY = ComplexF64[0 -im; im 0]
const pauliZ = ComplexF64[1 0; 0 -1]
const I2 = ComplexF64[1 0; 0 1]

# Kron helper
function kronN(mats...)
    result = mats[1]
    for i in 2:length(mats)
        result = kron(result, mats[i])
    end
    return result
end


function transverse_ising_H(J, h, N)
    # Build a 2^N×2^N sparse Hamiltonian
    H = spzeros(ComplexF64, 2^N, 2^N)
    
    # Interaction term
    for i in 1:(N-1)
        z_term = kronN([(j == i || j == i+1) ? pauliZ : I2 for j in 1:N]...)
        H .-= J .* z_term
    end

    # Transverse field term
    for i in 1:N
        x_term = kronN([(j == i) ? pauliX : I2 for j in 1:N]...)
        H .-= h .* x_term
    end

    return H
end

# Define single-qubit jump operators as sigma_x on each site
function single_spin_x_operator(N::Int, site::Int)
    # 2^N × 2^N operator: X on 'site', identity on others
    ops = [site == j ? pauliX : I2 for j in 1:N]
    return kronN(ops...)
end

function local_jump_ops(N::Int)
    # Return a vector of size N, each = X on a different site
    jump_ops = Matrix{ComplexF64}[]
    for site in 1:N
        push!(jump_ops, single_spin_x_operator(N, site))
    end
    return jump_ops
end


function local_jump_ops_sparse(N::Int)
    ops = Vector{SparseMatrixCSC{ComplexF64,Int}}()
    for site in 1:N
        denseX = single_spin_x_operator(N, site)
        push!(ops, sparse(denseX))
    end
    return ops
end

function energy(rho::AbstractMatrix{ComplexF64}, H::SparseMatrixCSC{ComplexF64,Int})
    # Compute E = tr(H * rho)
    # If H is sparse and rho is dense, we do Matrix(H)*rho
    return real(tr(Matrix(H)*rho))
end

"""
We define:
  E_current = tr(H*rho)
  E_new     = tr(H*(L_i * rho * L_i^dagger))
  dE        = E_current - E_new
  gᵢ        = min(1, exp(-beta * dE))
  
Note the sign: flipping it to E_new - E_current might or might not yield the correct
detailed balance sign. Here we do E_current - E_new => 'prefer' lower E_new.
"""
function compute_metropolis_rates(
    rho::Matrix{ComplexF64},
    H::SparseMatrixCSC{ComplexF64,Int},
    jump_ops::Vector{<:AbstractMatrix{ComplexF64}},
    beta::Real
)::Vector{Float64}
    Ecurr = energy(rho, H)
    gammas = zeros(Float64, length(jump_ops))

    for (i, L) in pairs(jump_ops)
        rho_new = L * rho * L'
        Enew    = real(tr(Matrix(H)*rho_new))
        dE      = Ecurr - Enew
        gammas[i] = min(1.0, exp(-beta*dE))
    end
    return gammas
end


function lindbladian(
    rho::Matrix{ComplexF64},
    jump_ops::Vector{<:AbstractMatrix{ComplexF64}},
    gammas::Vector{Float64}
)
    Lrho = zeros(ComplexF64, size(rho,1), size(rho,2))
    for (i, L) in enumerate(jump_ops)
        γ = gammas[i]
        cross = L * rho * L'
        deco  = 0.5 .* ((L' * L)*rho .+ rho*(L' * L))
        Lrho .+= γ .* (cross .- deco)
    end
    return Lrho
end

function evolve_density_matrix(
    rho::Matrix{ComplexF64},
    H::SparseMatrixCSC{ComplexF64,Int},
    jump_ops::Vector{<:AbstractMatrix{ComplexF64}},
    beta::Real; dt=0.001, steps=5000
)
    for _ in 1:steps
        gammas = compute_metropolis_rates(rho, H, jump_ops, beta)
        Lrho   = lindbladian(rho, jump_ops, gammas)
        rho_new = rho .+ dt .* Lrho

        # Enforce Hermiticity & trace=1
        rho_new = 0.5 .* (rho_new .+ adjoint(rho_new))
        tr_rho  = real(tr(rho_new))
        if tr_rho < 1e-14
            error("Trace(ρ) ~ 0 => unstable. Try smaller dt or fewer steps.")
        end
        rho_new ./= tr_rho
        rho = rho_new
    end
    return rho
end

# 5) test_gibbs_stationarity for a *sparse* H
function test_gibbs_stationarity(
    H::SparseMatrixCSC{ComplexF64,Int}, 
    beta::Real,
    jump_ops::Vector{<:AbstractMatrix{ComplexF64}}
)
    # Convert H to dense for exponentiation
    Hdense = Matrix(H)
    exp_neg_beta_H = exp(-beta .* Hdense)
    Z = tr(exp_neg_beta_H)
    rho_beta = exp_neg_beta_H ./ Z

    gammas = compute_metropolis_rates(rho_beta, H, jump_ops, beta)
    Lrho   = lindbladian(rho_beta, jump_ops, gammas)
    println("||L(rho_beta)|| = ", norm(Lrho))
    return rho_beta
end


function main_example()
    N = 3
    J = 1.0
    h = 1.0
    beta = 1.0

    # Build the sparse Hamiltonian
    H = transverse_ising_H(J, h, N)
    println("Built transverse Ising H. size=", size(H))

    # Build local jump ops => X on each site
    # We'll keep them dense for demonstration
    jump_ops = local_jump_ops(N)

    # Check stationarity of the naive quantum "Gibbs"
    _ = test_gibbs_stationarity(H, beta, jump_ops)

    # Evolve from random ρ
    dim = 2^N
    M = randn(ComplexF64, dim, dim)
    rho_init = M*adjoint(M)
    rho_init ./= tr(rho_init)

    steps = 5000
    dt    = 0.001
    println("Evolving from random state for steps=$steps, dt=$dt ...")
    rho_final = evolve_density_matrix(rho_init, H, jump_ops, beta; dt=dt, steps=steps)
    E_final   = energy(rho_final, H)

    # Compare with exact e^{-βH}/Z
    Hdense = Matrix(H)
    exp_neg_beta_H = exp(-beta .* Hdense)
    Z = tr(exp_neg_beta_H)
    rho_beta = exp_neg_beta_H ./ Z

    E_beta = real(tr(Hdense * rho_beta))
    dist   = 0.5 * sum(svdvals(rho_final - rho_beta))

    println("\nFinal energy   = $E_final")
    println("Exact Gibbs E  = $E_beta")
    println("Trace distance from exact Gibbs = $dist")
end

main_example()

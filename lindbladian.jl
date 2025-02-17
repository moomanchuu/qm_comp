using LinearAlgebra, SparseArrays, Printf, PrettyTables, Dates

###############################################
# Set up
###############################################
const pauliX = ComplexF64[0 1; 1 0]
const pauliY = ComplexF64[0 -im; im 0]
const pauliZ = ComplexF64[1 0; 0 -1]
const I2     = ComplexF64[1 0; 0 1]

###############################################
# Helper function to embed a 2×2 operator into the full Hilbert space for a spin chain.
# Places operator op at site pos in an N-spin chain.
###############################################
function embed_op(op::AbstractMatrix, pos::Int, N::Int)
    op_list = [I2 for _ in 1:N]
    op_list[pos] = op
    res = op_list[1]
    for i in 2:N
        res = kron(res, op_list[i])
    end
    return res
end

###############################################
# 1D spin chain Hamiltonian (sparse)
###############################################
function H_sparse_chain(N::Int)
    dim = 2^N
    Hrow, Hcol, Hval = Int[], Int[], Float64[]
    for i in 0:(dim-1)
        # Diagonal contributions
        diag_term = 0.0
        for k in 1:(N-1)
            spin_k  = ((i >> (N - k)) & 1 == 1) ? 1 : -1
            spin_k1 = ((i >> (N - (k+1))) & 1 == 1) ? 1 : -1
            diag_term += 0.25 * spin_k * spin_k1
        end
        push!(Hrow, i+1); push!(Hcol, i+1); push!(Hval, diag_term)
        # Off-diagonal contributions: flip adjacent spins if they differ.
        for k in 1:(N-1)
            spin_k  = ((i >> (N - k)) & 1 == 1) ? 1 : -1
            spin_k1 = ((i >> (N - (k+1))) & 1 == 1) ? 1 : -1
            if spin_k != spin_k1
                flipped = i ⊻ ((1 << (N - k)) | (1 << (N - (k+1))))
                push!(Hrow, flipped+1); push!(Hcol, i+1); push!(Hval, 0.5)
            end
        end
    end
    return sparse(Hrow, Hcol, Hval, dim, dim)
end

###############################################
# Helper function to approximate equality for floats.
###############################################
approx_equal(x, y; tol=1e-8) = abs(x - y) < tol

###############################################
# Filter an operator (in the energy eigenbasis) to keep only the matrix elements for which E_i - E_j ≈ nu. Build a sparse matrix.
###############################################
function filter_op(A_energy::AbstractMatrix, energies::Vector{Float64}, nu::Float64; tol=1e-8)
    N = length(energies)
    row_idx = Int[]
    col_idx = Int[]
    vals = ComplexF64[]
    for i in 1:N, j in 1:N
        if approx_equal(energies[i] - energies[j], nu; tol=tol) && abs(A_energy[i,j]) > tol
            push!(row_idx, i)
            push!(col_idx, j)
            push!(vals, A_energy[i,j])
        end
    end
    return sparse(row_idx, col_idx, vals, N, N)
end

###############################################
# Transform a jump operator A into the energy eigenbasis, filter it for transitions with energy difference nu, then transform back.
###############################################
function filtered_jump_op(A::AbstractMatrix, V::AbstractMatrix, energies::Vector{Float64}, nu::Float64; tol=1e-8)
    A_energy = V' * A * V
    A_energy_filtered = filter_op(A_energy, energies, nu; tol=tol)
    return V * Array(A_energy_filtered) * V'
end

###############################################
# Computes the action of the Lindbladian on a density matrix ρ:
#
# L(ρ) = ∑_a p(a) ∑_ν γ(ν) [A^a_ν  * ρ * (A^a_ν)^dag - 1/2 - ((A^a_ν)^dag * A^a_ν  * ρ) - 1/2 (ρ * (A^a_ν)^dag * A^a_ν)]
# where:
#   - A^a are the jump operators (embedded on the full Hilbert space)
#   - p(a) is the probability for jump operator a (default uniform)
#   - γ(ν) is the metropolis acceptance rate for the energy difference ν
###############################################
function lindbladian(rho::AbstractMatrix, H::AbstractMatrix, jump_ops::Vector{<:AbstractMatrix}, beta::Float64;
                             p=nothing, tol=1e-8, verbose=false)
    N = size(H, 1)
    eigs_H = eigen(Matrix(H))
    energies = eigs_H.values
    V = eigs_H.vectors

    num_jumps = length(jump_ops)
    if p === nothing
        p = fill(1.0 / num_jumps, num_jumps)
    end

    # Group energy differences by degeneracy
    distinct_nus = Dict{Float64, Bool}()
    for i in 1:N, j in 1:N
        nu_val = energies[i] - energies[j]
        key = round(nu_val, digits=8)
        distinct_nus[key] = true
    end
    if verbose
        println("Number of distinct energy differences: ", length(keys(distinct_nus)))
    end

    Lrho = zeros(ComplexF64, N, N)
    for (a_index, A) in enumerate(jump_ops)
        for nu in keys(distinct_nus)
            A_nu = filtered_jump_op(A, V, energies, nu; tol=tol)
            gamma_nu = min(1.0, exp(-beta * nu))
            if norm(A_nu) > tol
                Lrho += p[a_index] * gamma_nu * (A_nu * rho * A_nu' - 0.5 * (A_nu' * A_nu * rho + rho * A_nu' * A_nu))
            end
        end
    end
    return Lrho
end

###############################################
# Compute the Gibbs state as a matrix
###############################################
function gibbs_state_matrix(H::AbstractMatrix, beta::Float64)
    H_dense = Matrix(H)
    expH = exp(-beta * H_dense)
    return expH / tr(expH)
end

###############################################
# format matrices a little
###############################################
function truncate_round(x; tol=1e-10, digits=2)
    if abs(x) < tol
        return 0
    else
        if x isa Complex
            return complex(round(real(x), digits=digits), round(imag(x), digits=digits))
        else
            return round(x, digits=digits)
        end
    end
end

truncate_round_matrix(mat; tol=1e-10, digits=3) = [truncate_round(x; tol=tol, digits=digits) for x in mat]



###############################################
# Main
###############################################
function main()
    n_spins = 3
    beta = 10.0

    H = Matrix(H_sparse_chain(n_spins))
    gibbs = gibbs_state_matrix(H, beta)

    # This is the jump operator on all the spins. we could probably do just say, the first spin though? 
    jump_ops = [embed_op(pauli, site, n_spins) for site in 1:n_spins for pauli in [pauliX, pauliY, pauliZ]]
    
    t6 = time()
    L_gibbs = real(lindbladian(gibbs, H, jump_ops, beta; verbose=true))
    t7 = time()
    @printf("Davies Lindbladian on Gibbs state time: %.4f seconds\n", t7 - t6)

    println("\nGibbs State (ρ ∝ exp(-βH)):")
    pretty_table(truncate_round_matrix(gibbs; tol=1e-10, digits=3); header=nothing)
    println("\nLindbladian Acting on Gibbs State (L(ρ_Gibbs)):")
    pretty_table(truncate_round_matrix(L_gibbs; tol=1e-10, digits=3); header=nothing)
    
    # Is trace preserved? 
    tr_Lgibbs = tr(L_gibbs)
    @printf("\nTrace of L(ρ_Gibbs): %1.4e\n", abs(tr_Lgibbs))
    
    max_deviation = maximum(abs.(L_gibbs))
    @printf("\nMaximum deviation of L(rho_gibbs) from zero: %1.4e\n", max_deviation)

    # Test on a random density matrix.
    dim = size(H, 1)
    R = randn(ComplexF64, dim, dim) .+ im * randn(ComplexF64, dim, dim)
    rho_random = R * R'
    rho_random /= tr(rho_random)

    
    t10 = time()
    L_rho_random = real(lindbladian(rho_random, H, jump_ops, beta))
    t11 = time()
    @printf("Lindbladian on random density matrix time: %.4f seconds\n", t11 - t10)
    
    println("\nLindbladian Acting on Random Density Matrix (L(ρ_random)):")
    pretty_table(truncate_round_matrix(L_rho_random; tol=1e-10, digits=3); header=nothing)
    
    # trace preservation for random density matrix.
    tr_Lrandom = tr(L_rho_random)
    @printf("\nTrace of L(ρ_random): %1.4e\n", abs(tr_Lrandom))
end
main()

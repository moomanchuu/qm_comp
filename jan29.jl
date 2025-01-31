using LinearAlgebra, SparseArrays, PrettyTables, Arpack, Base.Threads, Plots, Graphs, Printf

# ====== 1. Constants: Pauli Matrices & Identity Matrix ====== #
pauliX = ComplexF64[0 1; 1 0]
pauliY = ComplexF64[0 -im; im 0]
pauliZ = ComplexF64[1 0; 0 -1]
I2 = ComplexF64[1 0; 0 1]

# ====== 2. Constructing the Hamiltonian ====== #
function get_spin(state::Int, pos::Int, N::Int)
    return ((state >> (N - pos)) & 1 == 1) ? 1 : -1
end

function flip_spin(state::Int, pos1::Int, pos2::Int, N::Int)
    return state ⊻ ((1 << (N - pos1)) | (1 << (N - pos2)))
end

function H_sparse_chain(N::Int)
    dim = 2^N
    H = spzeros(dim, dim)
    for i in 0:(dim-1)
        diag_term = 0.0
        for k in 1:(N-1)
            diag_term += 0.25 * get_spin(i, k, N) * get_spin(i, k+1, N)
        end
        H[i+1, i+1] = diag_term

        for k in 1:(N-1)
            if get_spin(i, k, N) != get_spin(i, k+1, N)
                flipped = flip_spin(i, k, k+1, N)
                H[flipped+1, i+1] = 0.5
            end
        end
    end
    return sparse(H)
end

# ====== 3. Gibbs State Computation ====== #
function gibbs_state(H, beta)
    exp_neg_beta_H = exp(-beta * H)
    return exp_neg_beta_H / tr(exp_neg_beta_H)
end

# ====== 4. Transition Matrix Computation ====== #
function transition_matrix(H, beta, jump_operators)
    num_states = size(H, 1)

    # Compute Eigenbasis of H
    eigs_H = eigen(H)
    V = eigs_H.vectors  # Columns are eigenstates
    energies = eigs_H.values
    println(V)
    println(energies)

    transition_amplitudes = spzeros(num_states, num_states)
    log_states = Int(log2(num_states))

    for op in jump_operators
        for qubit in 1:log_states
            full_operator = sparse(kron([qubit == k ? op : I2 for k in 1:log_states]...))
            Ak_eigenbasis = V' * full_operator * V  # Transform to eigenbasis
            transition_amplitudes .+= abs2.(Ak_eigenbasis)/(3*Int(log2(num_states))) # Compute |⟨ψ_j | A_k | ψ_i⟩|²
        end
    end
    println("Raw Transition Amplitudes:")
    pretty_table(Matrix(transition_amplitudes))


    # Compute Energy Differences
    # delta_energies = [energies[j] - energies[i] for i in 1:num_states, j in 1:num_states]
    # exp_factors = exp.(-beta .* delta_energies)

    # Perhaps the acceptance should be this
    acceptance = [i == j ? 0.0 : min(1, exp(-beta * (energies[j] - energies[i]))) for i in 1:num_states, j in 1:num_states]

    # Construct Transition Matrix P
    P = spzeros(num_states, num_states)
    for i in 1:num_states
        for j in 1:num_states
            if i != j
                P[i, j] = acceptance[i,j] * transition_amplitudes[i, j]
            end
        end
    end

    # Normalize each row to ensure stochasticity
    for i in 1:num_states
        row_sum = sum(P[i, :]) - P[i, i]
        if row_sum < 1
            P[i, i] = 1 - row_sum
        else
            println("Warning: Normalization issue at state $i, adjusting row.")
            P[i, :] ./= row_sum  # Ensure proper normalization
            P[i, i] = 0
        end
    end
    # println("Heres what P looks like:")
    # pretty_table(sparse(P))

    return sparse(P)
end


# function evolve_density_matrix(P, rho, iterations)
#     for _ in 1:iterations
#         rho = P * rho * P'
#         rho ./= tr(rho)
#     end
#     return rho
# end

function evolve_density_matrix(P, rho_vec, iterations)
    for _ in 1:iterations
        rho_vec = P * rho_vec
    end
    return rho_vec
end

# ====== 6. Convergence to Gibbs ====== #
function test_convergence_to_gibbs(H, beta, iterations, jump_operators)
    dim = size(H, 1)
    gibbs_matrix = gibbs_state(H, beta)
    P = transition_matrix(H, beta, jump_operators)

    # Detailed balance check
    detailed_balance_check = P .* exp.(-beta * diag(H)) - P' .* exp.(-beta * diag(H))
    # println("Max deviation from detailed balance: ", maximum(abs.(detailed_balance_check)))

    # println("Is P complex? ", any(imag.(P) .!= 0))
    P = real(P)
    pretty_table(P)

    # Compute dominant eigenvalue of P
    vals, vecs = eigs(P'; nev=1, which=:LR)
    # println("Dominant Eigenvalue of P: ", vals)

    # Compute stationary distribution
    stationary_dist = normalize(real(vecs[:,1]), 1)
    # println("Stationary Distribution (should match Gibbs state):")
    # pretty_table(stationary_dist)

    # Initialize and evolve density matrix
    rho = gibbs_state(H, 1.0)  # Start from Gibbs
    evolved_rho = evolve_density_matrix(P, rho, iterations)

    # Compute Frobenius norm difference
    difference = norm(evolved_rho - gibbs_matrix)

    # Compare eigenvalues
    eigs_evolved = eigen(evolved_rho).values
    eigs_gibbs = eigen(gibbs_matrix).values
    eigs_diff = norm(eigs_evolved - eigs_gibbs)

    # println("Evolved Density Matrix:")
    # pretty_table(evolved_rho)
    # println("Gibbs Density Matrix:")
    # pretty_table(gibbs_matrix)
    println("Frobenius Norm Difference: ", difference)
    println("Eigenvalue Norm Difference: ", eigs_diff)
end

# ====== 7. Run the Test ====== #
N = 5
beta = 1.0
iterations = 10000

H = Matrix(H_sparse_chain(N))
jump_operators = [pauliX, pauliY, pauliZ]

test_convergence_to_gibbs(H, beta, iterations, jump_operators)

P = transition_matrix(H, beta, jump_operators)

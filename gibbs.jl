using LinearAlgebra, SparseArrays, PrettyTables, Arpack, Base.Threads, Plots

const pauliX = ComplexF64[0 1; 1 0]
const pauliY = ComplexF64[0 -im; im 0]
const pauliZ = ComplexF64[1 0; 0 -1]
const I2 = ComplexF64[1 0; 0 1]


function get_spin(state::Int, pos::Int, N::Int)
    return ((state >> (N - pos)) & 1 == 1) ? 1 : -1
end

# We want to flip the spins at positions "pos1" and "pos2", using bit manipulatin
# This corresponds to the transverse (h * sig_x) term 
# Intuitively this corresponds to a quantum transition
function flip_spin(state::Int, pos1::Int, pos2::Int, N::Int)
    return state ⊻ ((1 << (N - pos1)) | (1 << (N - pos2)))
end


function H_sparse_chain(N::Int)
    # Dimension of the hamiltonian
    dim = 2^N
    Hrow, Hcol, Hval = Int[], Int[], Float64[]
    for i in 0:(dim-1)
        # Diagonal terms
        diag_term = 0.0
        for k in 1:(N-1)
            spin_k = get_spin(i, k, N)
            spin_k1 = get_spin(i, k + 1, N)
            diag_term += 0.25 * spin_k * spin_k1
        end
        push!(Hrow, i + 1)
        push!(Hcol, i + 1)
        push!(Hval, diag_term)
        # Off-diagonal terms
        for k in 1:(N-1)
            spin_k = get_spin(i, k, N)
            spin_k1 = get_spin(i, k + 1, N)
            if spin_k != spin_k1
                flipped = flip_spin(i, k, k + 1, N)
                push!(Hrow, flipped + 1)
                push!(Hcol, i + 1)
                push!(Hval, 0.5)
            end
        end
    end
    return sparse(Hrow, Hcol, Hval, dim, dim)
end

function gibbs_state(H, beta)
    eigs_H = eigen(H)
    V = eigs_H.vectors  # Columns are eigenstates
    energies = eigs_H.values  # Diagonal energy values
    gibbs_diag = exp.(-beta .* energies)
    gibbs_diag ./= sum(gibbs_diag)  # Normalize
    gibbs_matrix = V * Diagonal(gibbs_diag) * V'

    return gibbs_matrix
end



function transition_matrix(H, beta, jump_operators)
    num_states = size(H, 1)
    log_states = Int(log2(num_states))

    # Compute eigendecomposition of H
    eigs_H = eigen(H)
    V = eigs_H.vectors  # Eigenstates as columns
    energies = eigs_H.values  # Diagonal energy values

    # Initialize transition amplitude matrix
    transition_amplitudes = spzeros(num_states, num_states)

    # Instead of summing over all qubits, choose a random qubit flip per operator
    for op in jump_operators
        qubit = rand(1:log_states)  # Randomly pick a qubit
        full_operator = sparse(kron([qubit == k ? op : I2 for k in 1:log_states]...))
        Ak_eigenbasis = V' * full_operator * V  # Transform to eigenbasis
        transition_amplitudes .+= abs2.(Ak_eigenbasis)  # Compute |⟨ψ_j | A_k | ψ_i⟩|²
    end

    # Compute energy differences and Boltzmann factors
    delta_energies = [energies[j] - energies[i] for i in 1:num_states, j in 1:num_states]
    exp_factors = exp.(-beta .* delta_energies)

    # Construct transition matrix P
    P = spzeros(num_states, num_states)
    for i in 1:num_states
        for j in 1:num_states
            if i != j
                P[i, j] = min(1, exp_factors[i, j]) * transition_amplitudes[i, j] / log_states
            end
        end
    end

    # Normalize rows
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

    return sparse(P)
end



H = Matrix(H_sparse_chain(3))
gibbs=gibbs_state(H, 1.0)
#pretty_table(gibbs)
#println(sum(diag(gibbs)))  # Should output 1.0
#println(eigvals(gibbs))

using LinearAlgebra, SparseArrays


# Function to create a random density matrix
function create_random_density_matrix(dim::Int)
    A = randn(Float64, dim, dim)  # Random real matrix
    rho = A * A'  # Make it symmetric and positive semi-definite
    return rho / tr(rho)  # Normalize so Tr(rho) = 1
end

function evolve_density_matrix(P, rho, iterations)
    for _ in 1:iterations
        rho = P * rho
        rho ./= tr(rho)
        rho = real(rho)
    end
    return rho
end


function test_convergence_to_gibbs(H, beta, iterations, jump_operators)
    dim = size(H, 1)

    # Compute the Gibbs density matrix
    gibbs_matrix = exp(-beta * H)
    gibbs_matrix ./= tr(gibbs_matrix)
    gibbs_matrix = gibbs_state(H, beta)
    # Generate the transition matrix
    P = transition_matrix(H, beta, jump_operators)
    detailed_balance_check = P .* exp.(-beta * diag(H)) - P' .* exp.(-beta * diag(H))
    println("Max deviation from detailed balance: ", maximum(abs.(detailed_balance_check)))

    # Ensure P is real
    println("Is P complex? ", any(imag.(P) .!= 0))
    P = real(P)
    pretty_table(P)

    # Compute eigenvalues of P (for sparse matrices, use eigs instead of eigen)
    vals, vecs = eigs(P'; nev=1, which=:LR)  # Find the largest eigenvalue (nev=1)

    println("Dominant Eigenvalue of P: ", vals)
    stationary_dist = normalize(real(vecs[:,1]), 1)  # Extract stationary state

    println("Stationary Distribution (should match Gibbs state):")
    pretty_table(stationary_dist)

    # Initialize from a high-temperature Gibbs-like state
    rho = create_random_density_matrix(dim)
    pretty_table(rho)
    evolved_rho = evolve_density_matrix(P, rho, iterations)
    
    # Check imaginary parts
    println("Max imaginary part of evolved_rho: ", maximum(abs.(imag.(evolved_rho))))
    evolved_rho = real(evolved_rho)  # Final cleanup

    # Compute Frobenius norm difference between matrices
    difference = norm(evolved_rho - gibbs_matrix)

    # Compute eigenvalues
    eigs_evolved = eigen(evolved_rho).values
    eigs_gibbs = eigen(gibbs_matrix).values

    # Compute norm difference of eigenvalues
    eigs_diff = norm(eigs_evolved - eigs_gibbs)
    
    println("Evolved Density Matrix:")
    pretty_table(evolved_rho)
    println("Gibbs Density Matrix:")
    pretty_table(gibbs_matrix)
    println("Frobenius Norm Difference:", difference)
    println("Eigenvalues of Evolved Density Matrix: ", eigs_evolved)
    println("Eigenvalues of Gibbs Density Matrix: ", eigs_gibbs)
    println("Eigenvalue Norm Difference:", eigs_diff)

    # Check if P^t converges to the Gibbs state
    function check_convergence(P, rho, max_t=1000, tol=1e-6)
        P_t = copy(P)
        for t in 1:max_t
            P_t = P_t * P  # Multiply P by itself (Markov evolution)
            if norm(P_t - P_t * P, Inf) < tol  # Check for stability
                println("P^t converged at t = $t")
                return P_t  # Return stationary matrix
            end
        end
        println("P^t did not fully converge in $max_t steps")
        return P_t
    end
    P_stationary = check_convergence(P, gibbs_matrix)
end


# Example usage
H = Matrix(H_sparse_chain(3))
jump_operators = [pauliX, pauliY, pauliZ]  # Pauli operators as jump operators
test_convergence_to_gibbs(H, 1.0, 100, jump_operators)


H = Matrix(H_sparse_chain(3))
P = transition_matrix(H, 1.0, jump_operators)


using Graphs

function check_irreducibility(P)
    num_states = size(P, 1)
    
    # Convert P into an adjacency matrix
    adjacency_matrix = P .> 0  # Convert nonzero entries to 1 (binary connectivity)
    
    # Convert to Graph structure
    G = Graph(adjacency_matrix)
    
    # Find the number of strongly connected components
    num_scc = length(connected_components(G))
    
    println("Number of Strongly Connected Components (SCCs): ", num_scc)
    
    if num_scc == 1
        println("Transition matrix P is irreducible (fully connected).")
    else
        println("P is reducible! Not all states communicate.")
    end
end

function check_aperiodicity(P, max_t=50, tol=1e-6)
    num_states = size(P, 1)
    P_t = copy(P)
    
    for t in 1:max_t
        P_t = P_t * P  # Multiply P by itself
        
        if all(P_t .> tol)  # If all entries are positive, P is aperiodic
            println(" Transition matrix P is aperiodic (t = $t).")
            return true
        end
    end
    
    println(" P might be periodic (no full positive matrix within t = $max_t).")
    return false
end

check_irreducibility(P)
check_aperiodicity(P)

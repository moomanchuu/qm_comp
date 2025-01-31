using LinearAlgebra, SparseArrays, PrettyTables, Arpack, Base.Threads, Plots, JSON
using Graphs

pauliX = ComplexF64[0 1; 1 0]
pauliY = ComplexF64[0 -im; im 0]
pauliZ = ComplexF64[1 0; 0 -1]
I2 = ComplexF64[1 0; 0 1]

function get_spin(state::Int, pos::Int, N::Int)
    return ((state >> (N - pos)) & 1 == 1) ? 1 : -1
end

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

function random_state_vector(dim::Int)
    x = rand(dim)  # Generate random positive numbers
    return x / sum(x)  # Normalize so sum(x) = 1
end

function evolve_state_vector(P, x, iterations)
    return (x' * (P^iterations))'  # Evolve as a row vector and return as column vector
end

function create_random_density_matrix(dim::Int)
    A = randn(Float64, dim, dim)  # Random real matrix
    rho = A * A'  # Make it symmetric and positive semi-definite
    return rho / tr(rho)  # Normalize so Tr(rho) = 1
end

function evolve_density_matrix(P, rho, iterations)
    P_iter = P^iterations  # Compute P^iterations
    rho_final = rho' * P_iter  # Right multiply by P_iter
    rho_final ./= tr(rho_final)  # Normalize
    return real(rho_final)  # Ensure real values
end

function check_irreducibility(P)
    num_states = size(P, 1)
    
    # Convert P into an adjacency matrix
    adjacency_matrix = P .> 0  # Convert nonzero entries to 1 (binary connectivity)
    
    # Convert to Graph structure
    G = Graph(adjacency_matrix)
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

function gibbs_state(H, beta)
    E = eigen(H).values  # Compute eigenvalues of H
    weights = exp.(-beta .* E)  # Compute Boltzmann weights
    Z = sum(weights)  # Partition function (normalization factor)
    return weights / Z  # Normalize to get probability distribution
end

function transition_matrix(H, beta, jump_operators)
    num_states = size(H, 1)
    log_states = Int(log2(num_states))
    I2 = sparse(I, 2, 2)

    eigs_H = eigen(H)
    energies = eigs_H.values  
    
    transition_amplitudes = spzeros(num_states, num_states)
    for op in jump_operators
        for qubit in 1:log_states
            full_operator = sparse(kron([qubit == k ? op : I2 for k in 1:log_states]...))
            transition_amplitudes .+= abs2.(full_operator) 
        end
    end

    # Compute delta energies using correct eigenvalues
    delta_energies = [energies[j] - energies[i] for i in 1:num_states, j in 1:num_states]
    exp_factors = exp.(-beta .* delta_energies)

    P = spzeros(num_states, num_states)
    @threads for i in 1:num_states
        for j in 1:num_states
            if i != j
                P[i, j] = min(1, exp_factors[i, j]) * transition_amplitudes[i, j]/ (3 * log_states)
            end
        end
    end

    for i in 1:num_states
        row_sum = sum(P[i, :]) - P[i, i]
        if row_sum < 1
            P[i, i] = 1 - row_sum

        else
            println("RING RING RING RING RING")
            P[i, :] ./= row_sum 
            P[i, i] = 0
        end
    end

    return sparse(P)
end

# function test_convergence_to_gibbs(H, beta, iterations, jump_operators)
#     dim = size(H, 1)

#     # Compute the Gibbs density matrix
#     gibbs_matrix = exp(-beta * H)
#     gibbs_matrix ./= tr(gibbs_matrix)

#     # Generate the transition matrix
#     P = transition_matrix(H, beta, jump_operators)
#     P = real(P)
#     println("Left Stochastic Transition Matrix")
#     pretty_table(P)

#     # Compute eigenvalues of P (for sparse matrices, use eigs instead of eigen)
#     vals, vecs = eigs(P'; nev=1, which=:LR)  # Find the largest eigenvalue (nev=1)
#     println("Gibbs State")
#     println(gibbs_state(H, beta))
#     println("Dominant Eigenvalue of P: ", vals)
#     stationary_dist = normalize(real(vecs[:,1]), 1)  # Extract stationary state

#     println("Stationary Distribution (should match Gibbs state):")
#     pretty_table(stationary_dist)
    
#     # Initialize from a high-temperature Gibbs-like state
#     #rho = create_random_density_matrix(dim)
#     rho = random_state_vector(dim)
#     println("Random State Vector")
#     pretty_table(rho)
#     evolved_rho = evolve_state_vector(P, rho, iterations)
#     println("Evolved State Vector")
#     pretty_table(evolved_rho)

#     # Check if P^t converges to the Gibbs state
#     function check_convergence(P, rho, max_t=1000, tol=1e-6)
#         P_t = copy(P)
#         for t in 1:max_t
#             P_t = P_t * P  # Multiply P by itself (Markov evolution)
#             if norm(P_t - P_t * P, Inf) < tol  # Check for stability
#                 println("P^t converged at t = $t")
#                 return P_t  # Return stationary matrix
#             end
#         end
#         println("P^t did not fully converge in $max_t steps")
#         return P_t
#     end
#     P_stationary = check_convergence(P, gibbs_matrix)

#     println("Difference Between Evolved State Vector and Gibbs State")
#     println(norm(evolved_rho - gibbs_state(H, beta)))
# end

function test_convergence_to_gibbs(H, beta, iterations, jump_operators)
    dim = size(H, 1)

    # Compute the Gibbs density matrix
    gibbs_matrix = exp(-beta * H)
    gibbs_matrix ./= tr(gibbs_matrix)

    # Generate the transition matrix
    P = transition_matrix(H, beta, jump_operators)
    P = real(P)
    println("Left Stochastic Transition Matrix")
    pretty_table(P)

    # Compute eigenvalues of P
    vals, vecs = eigs(P'; nev=1, which=:LR)  # Find the largest eigenvalue (nev=1)
    println("Gibbs State")
    gibbs_dist = gibbs_state(H, beta)
    println(gibbs_dist)
    println("Dominant Eigenvalue of P: ", vals)
    stationary_dist = normalize(real(vecs[:,1]), 1)  # Extract stationary state

    println("Stationary Distribution (should match Gibbs state):")
    pretty_table(stationary_dist)
    
    # Initialize from a random state
    rho = random_state_vector(dim)
    println("Random State Vector")
    pretty_table(rho)
    
    # Track convergence and save data
    open("state_data.jsonl", "w") do io
        # Save Gibbs state at the beginning
        gibbs_data = Dict("gibbs_state" => gibbs_dist)
        println(io, JSON.json(gibbs_data))
        
        for t in 1:iterations
            rho = evolve_state_vector(P, rho, 1)  # Apply transition matrix
            diff_sq = sum((rho - gibbs_dist).^2)  # Compute squared difference
            
            data = Dict("iteration" => t, "difference_sq" => diff_sq, "rho" => rho)
            println(io, JSON.json(data))  # Save state data as JSON line
        end
    end

    println("Difference Between Evolved State Vector and Gibbs State")
    println(norm(rho - gibbs_state(H, beta)))
end


# Example usage
H = Matrix(H_sparse_chain(3))
jump_operators = [pauliX, pauliY, pauliZ]  # Pauli operators as jump operators
test_convergence_to_gibbs(H, 1.0, 100, jump_operators)


H = Matrix(H_sparse_chain(3))
P = transition_matrix(H, 1.0, jump_operators)




check_irreducibility(P)
check_aperiodicity(P)

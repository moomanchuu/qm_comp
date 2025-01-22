using LinearAlgebra, SparseArrays, PrettyTables, Arpack, Base.Threads, Plots, BenchmarkTools, JSON, ThreadsX

const pauliX = ComplexF64[0 1; 1 0]
const pauliY = ComplexF64[0 -im; im 0]
const pauliZ = ComplexF64[1 0; 0 -1]
const I2 = ComplexF64[1 0; 0 1]

# functions for 1D chain H
# We want to get the spin of the state
# Binary represenation encodes the spin configuration of the state
# We use bitwise shifting to get the spin at the "pos" position
function get_spin(state::Int, pos::Int, N::Int)
    return ((state >> (N - pos)) & 1 == 1) ? 1 : -1
end

# We want to flip the spins at positions "pos1" and "pos2", using bit manipulatin
# This corresponds to the transverse (h * sig_x) term 
# Intuitively this corresponds to a quantum transition
function flip_spin(state::Int, pos1::Int, pos2::Int, N::Int)
    return state ⊻ ((1 << (N - pos1)) | (1 << (N - pos2)))
end

# We compute Kronecker products of multiple matrices
# This allows us to create multi-qubit operaators from single ones
# (Ie increasing N )
function kronN(mats...)
    result = mats[1]
    for i in eachindex(mats)[2:end]
        result = kron(result, mats[i])
    end
    return result
end

# Function for transverse Ising H for N qubits
# Should probably make it sparse --> made sparse 
# We construct the Transverse Ising hamiltonian. We compute the interaction term and the transverse term seperately
# We follow the convention of the Ising model having nearest-neighbor interactions 
function transverse_ising_H(J, h, N)
    H = spzeros(ComplexF64, 2^N, 2^N)
    
    # Interaction term (-J * Σ σ_i^z σ_(i+1)^z)
    for i in 1:(N-1)
        z_term = kronN([(j == i || j == i+1) ? pauliZ : I2 for j in 1:N]...)
        H -= J * z_term
    end

    # Transverse field term (-h * Σ σ_i^x)
    for i in 1:N
        x_term = kronN([(j == i) ? pauliX : I2 for j in 1:N]...)
        H -= h * x_term
    end

    return H
end

# Function for sparse 1D chain H
function H_sparse_chain(N::Int)
# Dimension of the hamiltonian
    dim = 2^N
    Hrow, Hcol, Hval = Int[], Int[], Float64[]

# Psuedo code:
# iterate over all possible configurations in hilbert space (2^N)
#
# For diagonal terms:
# itereate over N (There are N terms in the diagonal)
# for each one we want to compute whether the spin and the next spin are either aligned or anti-aligned
# This gives us either a +1 or -1, which we add to the total diagonal term
# the diagonal term can be calucated by 0.25*(# of aligned pairs - # of anti aligned pairs)
# Then we push this to the arrays we are keeping (hrow, hcol, hval)
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

# For each off diagonal term:
# we want to again iterate over N and get the spin
# if the spins are not the same, 
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

# Stochastic transition matrix P where entry P[i,j] is given by:
# min(1, e^-beta(Ej-Ei)) * Sigma_k |< Phi_j | Ak | Phi_i>|^2
# where Ak are the jump operators to be summed over
function compute_tm_jump(H, beta, jump_operators)
    num_states = size(H, 1)
    P = spzeros(num_states, num_states)  
    log_states = Int(log2(num_states))
    I2 = sparse(I, 2, 2)

    for i in 1:num_states
        for j in 1:num_states
            if i != j
                transition_amplitude = 0.0
                for op in jump_operators
                    for qubit in 1:log_states
                        # Create the sparse Kronecker product of operators
                        ops = [qubit == k ? op : I2 for k in 1:log_states]
                        full_operator = sparse(kron(ops...))
                        transition_amplitude += abs(full_operator[i, j])^2
                    end
                end
                P[i, j] = min(1, exp(-beta * (real(H[j, j]) - real(H[i, i])))) * transition_amplitude
            end
        end
    end

    # Normalize
    for i in 1:num_states
        row_sum = sum(P[i, :])
        if row_sum > 0
            P[i, :] ./= row_sum
        end
    end

    return sparse(P)
end

# function compute_tm_jump_optimized(H, beta, jump_operators)
#     num_states = size(H, 1)
#     log_states = Int(log2(num_states))
#     I2 = sparse(I, 2, 2)  # Sparse identity matrix

#     # Precompute all Kronecker products for each jump operator
#     precomputed_ops = Dict()
#     for op in jump_operators
#         precomputed_ops[op] = [sparse(kron([qubit == k ? op : I2 for k in 1:log_states]...)) for qubit in 1:log_states]
#     end

#     P = spzeros(num_states, num_states)

#     # Parallelizing outer loops
#     #@threads for i in 1:num_states
#     ThreadsX.foreach(1:num_states) do i
#         for j in 1:num_states
#             if i != j
#                 transition_amplitude = 0.0
#                 for op in jump_operators
#                     for full_operator in precomputed_ops[op]
#                         transition_amplitude += abs(full_operator[i, j])^2
#                     end
#                 end
#                 P[i, j] = min(1, exp(-beta * (real(H[j, j]) - real(H[i, i])))) * transition_amplitude
#             end
#         end
#     end

#     # Normalize each row efficiently
#     row_sums = vec(sum(P, dims=2))
#     nz_rows = findall(row_sums .> 0)  # Avoid division by zero
#     P[nz_rows, :] ./= row_sums[nz_rows]

#     return sparse(P)
# end

function compute_tm_jump_optimized(H, beta, jump_operators)
    num_states = size(H, 1)
    log_states = Int(log2(num_states))
    I2 = sparse(I, 2, 2)

    # Precompute combined operator contributions
    transition_amplitudes = spzeros(num_states, num_states)
    for op in jump_operators
        for qubit in 1:log_states
            full_operator = sparse(kron([qubit == k ? op : I2 for k in 1:log_states]...))
            transition_amplitudes .+= abs2.(full_operator)
        end
    end

    # Precompute exponential factors
    delta_energies = [real(H[j, j] - H[i, i]) for i in 1:num_states, j in 1:num_states]
    exp_factors = exp.(-beta .* delta_energies)

    # Construct P
    P = spzeros(num_states, num_states)
    @threads for i in 1:num_states
        for j in 1:num_states
            if i != j
                prob = min(1, exp_factors[i, j]) * transition_amplitudes[i, j]
                P[i, j] = prob
            end
        end
    end

    # Normalize rows
    row_sums = vec(sum(P, dims=2))
    nz_rows = findall(row_sums .> 0)
    P[nz_rows, :] ./= row_sums[nz_rows]

    return sparse(P)
end



# Spectral Gap
function compute_sg(tm, Nq)
    if Nq > 4
        vals, vecs = eigs(tm, nev=2, which=:LR)
    else
        vals, vecs = eigs(tm, nev=2, which=:LR)
        #eigensystem = eigen(tm)
        #vals = eigensystem.values
    end
    s = sort(real(vals), rev=true)
    sg = 1 - s[2]
    return sg, s
end
# N = 9
# J = 1.0 
# h = 0.5 
# beta = 1.0  

#=tm_transverse_jump = compute_tm_jump(transverse_ising_H(J, h, N), beta, [pauliX, pauliY, pauliZ])
println("Transition Matrix with Jump Operators (Transverse Ising):")
pretty_table(tm_transverse_jump)

println("Transition Matrix with Jump Operators (1D Chain):")
pretty_table(tm_chain_jump)
sg_transverse, asdf = compute_sg(tm_transverse_jump, N)
println("Eigenvalues, Spectral Gap: ", asdf, ", ", sg_transverse)=#

# tm_chain_jump = compute_tm_jump_optimized(Matrix(H_sparse_chain(N)), beta, [pauliX, pauliY, pauliZ])

# sg_chain, asdf = compute_sg(tm_chain_jump, N)
# println("Eigenvalues, Spectral Gap: ", asdf, ", ", sg_chain)

function compute_sg_and_time(N, J, h, beta)
    H = H_sparse_chain(N)
    jump_ops = [pauliX, pauliY, pauliZ]
    start_time = time_ns()  # Start timing
    tm_chain_jump = compute_tm_jump_optimized(Matrix(H), beta, jump_ops)
    spectral_gap, eigenvalues = compute_sg(tm_chain_jump, N)
    elapsed_time = (time_ns() - start_time) * 1e-9  # Convert to seconds
    return spectral_gap, eigenvalues, elapsed_time
end

results_file = "results.json"
# # Load existing results if the file exists
# if isfile(results_file)
#     results = JSON.parsefile(results_file)
#     spectral_gaps = results["spectral_gaps"]
#     computation_times = results["computation_times"]
#     eigenvalues = results["eigenvalues"]
#     last_N = results["last_N"]
# else
spectral_gaps = []
computation_times = []
eigenvalues = []
last_N = 2
# end

J = 1.0 
h = 0.5 
beta = 1.0  
N_range = 2:14
# N_range = (last_N+1):12

# Compute results
for N in N_range
    sg, eigens, time = compute_sg_and_time(N, J, h, beta)
    push!(spectral_gaps, sg)
    push!(computation_times, time)
    push!(eigenvalues, eigens)

    # Save progress
    results = Dict(
        "spectral_gaps" => spectral_gaps,
        "computation_times" => computation_times,
        "eigenvalues" => eigenvalues,
        "last_N" => N
    )
    open(results_file, "w") do io
        JSON.print(io, results)
    end

    println("N = $N, Spectral Gap = $sg, Computation Time = $time seconds")
    println("Results saved to $results_file")
end

print("All results have been saved to results.json. Please run 'plot1.jl' to obtain the relevant plots.")

# plot(N_range, spectral_gaps, marker=:circle, label="Spectral Gap",
#      xlabel="N (Number of Qubits)", ylabel="Spectral Gap",
#      title="Spectral Gap vs N", xticks = N_range)
# savefig("spectral_gap_vs_N.png")

# plot(N_range, computation_times, marker=:circle, label="Computation Time",
#      xlabel="N (Number of Qubits)", ylabel="Computation Time (s)",
#      title="Computation Time vs N", xticks = N_range)
# savefig("computation_time_vs_N.png")



# # Load results
# results = JSON.parsefile(results_file)
# N_range = 2:results["last_N"]
# spectral_gaps = results["spectral_gaps"]
# computation_times = results["computation_times"]

# # Plot Spectral Gap
# plot(N_range, spectral_gaps, marker=:circle, label="Spectral Gap",
#      xlabel="N (Number of Qubits)", ylabel="Spectral Gap",
#      title="Spectral Gap vs N", xticks=N_range)
# savefig("spectral_gap_vs_N.png")

# # Plot Computation Time
# plot(N_range, computation_times, marker=:circle, label="Computation Time",
#      xlabel="N (Number of Qubits)", ylabel="Computation Time (s)",
#      title="Computation Time vs N", xticks=N_range)
# savefig("computation_time_vs_N.png")

# N = 9 gives us: Eigenvalues, Spectral Gap: [0.9999999999999987, 0.8398751763036383], 0.16012482369636172


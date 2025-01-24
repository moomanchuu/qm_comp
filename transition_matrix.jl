using LinearAlgebra, SparseArrays, PrettyTables, Arpack,Base.Threads

const pauliX = ComplexF64[0 1; 1 0]
const pauliY = ComplexF64[0 -im; im 0]
const pauliZ = ComplexF64[1 0; 0 -1]
const I2 = ComplexF64[1 0; 0 1]

# functions for 1D chain H
function get_spin(state::Int, pos::Int, N::Int)
    return ((state >> (N - pos)) & 1 == 1) ? 1 : -1
end

function flip_spin(state::Int, pos1::Int, pos2::Int, N::Int)
    return state ⊻ ((1 << (N - pos1)) | (1 << (N - pos2)))
end

function kronN(mats...)
    result = mats[1]
    for i in eachindex(mats)[2:end]
        result = kron(result, mats[i])
    end
    return result
end

# Function for transverse Ising H for N qubits
# Should probably make it sparse
function transverse_ising_H(J, h, N)
    H = zeros(ComplexF64, 2^N, 2^N)
    
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

function compute_tm_jump_optimized(H, beta, jump_operators)
    num_states = size(H, 1)
    log_states = Int(log2(num_states))
    I2 = sparse(I, 2, 2)  # Sparse identity matrix

    # Precompute all Kronecker products for each jump operator
    precomputed_ops = Dict()
    for op in jump_operators
        precomputed_ops[op] = [sparse(kron([qubit == k ? op : I2 for k in 1:log_states]...)) for qubit in 1:log_states]
    end

    P = spzeros(num_states, num_states)

    # Parallelizing outer loops
    @threads for i in 1:num_states
        for j in 1:num_states
            if i != j
                transition_amplitude = 0.0
                for op in jump_operators
                    for full_operator in precomputed_ops[op]
                        transition_amplitude += abs(full_operator[i, j])^2
                    end
                end
                P[i, j] = min(1, exp(-beta * (real(H[j, j]) - real(H[i, i])))) * transition_amplitude
            end
        end
    end

    # Normalize each row efficiently
    row_sums = vec(sum(P, dims=2))
    nz_rows = findall(row_sums .> 0)  # Avoid division by zero
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
N = 12
J = 1.0 
h = 0.5 
beta = 1.0  

#=tm_transverse_jump = compute_tm_jump(transverse_ising_H(J, h, N), beta, [pauliX, pauliY, pauliZ])
println("Transition Matrix with Jump Operators (Transverse Ising):")
pretty_table(tm_transverse_jump)

println("Transition Matrix with Jump Operators (1D Chain):")
pretty_table(tm_chain_jump)
sg_transverse, asdf = compute_sg(tm_transverse_jump, N)
println("Eigenvalues, Spectral Gap: ", asdf, ", ", sg_transverse)=#

tm_chain_jump = compute_tm_jump_optimized(Matrix(H_sparse_chain(N)), beta, [pauliX, pauliY, pauliZ])

sg_chain, asdf = compute_sg(tm_chain_jump, N)
println("Eigenvalues, Spectral Gap: ", asdf, ", ", sg_chain)

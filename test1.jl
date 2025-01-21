using LinearAlgebra
using PrettyTables
using Yao
using Distributions
using Statistics
using Random

# reg_zero3 = zero_state(3) # zero state |000>

# print_table(reg_zero3)

# num_registers = 4
# r = 8
# 
# regs = [zero_state(r) for _ in 1:num_registers]
# 
# for (i, reg) in enumerate(registers)
# 	println("$i: ", reg)
# end
#
# Pauli matrices
pauliX = [0 1; 1 0]
pauliZ = [1 0; 0 -1]
pauliY = [0 -1im; 1im 0]
I2 = [1.0 0.0; 0.0 1.0]  # 2x2 identity matrix

# Tensor product function
function kronN(mats...)
    result = mats[1]
    for i in eachindex(mats)[2:end]
        result = kron(result, mats[i])
    end
    return result
end

# Transverse Ising Hamiltonian construction for N spins
function transverse_ising_hamiltonian(J, h, N)
    H = zeros(2^N, 2^N)
    
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

# Parameters
J = 1.0  # Coupling constant # Affects numbers on diagonal
h = 0.5  # Transverse magnetic field # Affects numbers on off diagonal
N = 3   # Number of spins # related to dimension -> Dim(H) = 2^N

# Construct the Hamiltonian
H = transverse_ising_hamiltonian(J, h, N)

n = 2^N # dimension

beta = 1 # Inverse Temperature

# Define hamiltonian, diagonalize. Make it hermitian to ensure real eigenvalues
# A = rand(Real,n,n)
# H = 0.5*(A + A')

# Formally write the diagonalization algorithm
function Diagonalize(x)
	E = eigen(x)
	eigenvalues = E.values
	eigenvectors = E.vectors
	D = Diagonal(eigenvalues)
	P = eigenvectors
	P_inv = inv(P)
	return D, P*D*P_inv
end

# function Evolve(H, N; dist=nothing)
#     jump_operators = [pauliX, pauliY, pauliZ]
# 
#     if dist === nothing
#         # Uniform distribution: each Pauli operator on each qubit has probability 1/(3N)
#         operator_dist = Categorical([1/3 for _ in jump_operators])
#         qubit_dist = Categorical([1/N for _ in 1:N])
#     else
#         if length(dist) != 3 * N
#             error("Custom distribution `dist` must have length 3 * N.")
#         end
#         operator_dist = Categorical(dist[1:3])      # Probability distribution for each Pauli operator
#         qubit_dist = Categorical(dist[4:end])       # Probability distribution for each qubit
#     end
# 
# 
#     C = jump_operators[rand(operator_dist)]
#     qubit_index = rand(qubit_dist)
#     if rand(Bool)
#         C_op = C
#     else
#         C_op = C'  # Conjugate transpose of C (C†)
#     end
# 
#     full_operator = kronN([j == qubit_index ? C_op : I2 for j in 1:N]...)
#     # pretty_table(H)
#     #pretty_table(full_operator)
#     H_new = full_operator * H #* full_operator'
#     # pretty_table(H_new)
#     return H_new
#     
# end

function EvolveState(state, num_qubits; dist=nothing)
    jump_operators = [pauliX, pauliY, pauliZ]
    if dist === nothing
        operator_dist = Categorical([1/3 for _ in jump_operators])
        qubit_dist = Categorical([1/num_qubits for _ in 1:num_qubits])
    else
        # Use a custom distribution if provided
        if length(dist) != 3 * num_qubits
            error("Custom distribution `dist` must have length 3 * num_qubits.")
        end
        # Split `dist` into operator and qubit distributions
        operator_dist = Categorical(dist[1:3])
        qubit_dist = Categorical(dist[4:end])
    end

    # Randomly select a Pauli operator and a qubit index to apply it to
    selected_operator = jump_operators[rand(operator_dist)]
    selected_qubit = rand(qubit_dist)

    # Choose the operator or its adjoint with equal probability
    applied_operator = rand(Bool) ? selected_operator : selected_operator'
    full_operator = kronN([j == selected_qubit ? applied_operator : I2 for j in 1:num_qubits]...)
    
    new_state = full_operator * state
    return new_state
end

function f_metro(D, D_new, beta)
	return min(1, exp(beta*(median(diag(D)) - median(diag(D_new)))))
end



dist = Uniform(0,1)
D, _ = Diagonalize(H)
# H_new = Evolve(H, N)
# D_new, _ = Diagonalize(H_new)
# # D_q = Evolve(D, dist) #Test to see if just evolving the diagonal would give the same result ---No
# diag_D = diag(D)
# med_D = median(diag(D))
# med_D_new = median(diag(D_new))


# pretty_table(H)
# pretty_table(D)
# pretty_table(H_new)
# pretty_table(D_new)
# 
# println(med_D)
# println(med_D_new)
# 
# println(f_metro(D, D_new, beta))
#
#

# 1. Basis state |000>
println("Initial basis state |000⟩:")
state_000 = zeros(ComplexF64, 2^num_qubits)
state_000[1] = 1.0
display_state(state_000, "Before Evolution |000⟩")
evolved_state_000 = EvolveState(state_000, num_qubits)
display_state(evolved_state_000, "After Evolution |000⟩")

# 2. Superposition state |+>^3
println("\nInitial superposition state |+>^3:")
function hadamard_transform(num_qubits)
    H = [1 1; 1 -1] / sqrt(2)
    full_H = H
    for _ in 2:num_qubits
        full_H = kron(full_H, H)
    end
    return full_H
end

state_plus = zeros(ComplexF64, 2^num_qubits)
state_plus[1] = 1.0
state_plus = hadamard_transform(num_qubits) * state_plus
display_state(state_plus, "Before Evolution |+⟩^3")
evolved_state_plus = EvolveState(state_plus, num_qubits)
display_state(evolved_state_plus, "After Evolution |+⟩^3")

# 3. Random state
println("\nInitial random state:")
state_random = randn(ComplexF64, 2^num_qubits)
state_random /= norm(state_random)
display_state(state_random, "Before Evolution Random State")
evolved_state_random = EvolveState(state_random, num_qubits)
display_state(evolved_state_random, "After Evolution Random State")

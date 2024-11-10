using LinearAlgebra

# Pauli matrices
pauliX = [0 1; 1 0]
pauliZ = [1 0; 0 -1]
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
J = 1.0  # Coupling constant
h = 0.5  # Transverse magnetic field
N = 2   # Number of spins

# Construct the Hamiltonian
H = transverse_ising_hamiltonian(J, h, N)

# Eigen decomposition of Hamiltonian (gives energy levels and eigenstates)
eigenvalues, eigenvectors = eigen(H)

# Transition matrix (stochastic)
transition_matrix = abs.(eigenvectors)^2  # Probabilities based on the eigenvectors

# Metropolis acceptance rule
function metropolis_accept(energy_current, energy_new, beta)
    # Metropolis rule
    P_accept = exp(-beta * (energy_new - energy_current))
    return rand() < min(1, P_accept)  # Accept or reject
end

# Choose initial state (e.g., the ground state)
phi = eigenvectors[:, 1]  # Ground state

# Example energy of the current state
energy_current = eigenvalues[1]

# Modify the state (you can define your own modification rule)
phi_new = eigenvectors[:, 2]  # Example modification, target new state
energy_new = eigenvalues[2]    # Energy of the new state

# Decide whether to accept the transition
beta = 1.0  # Inverse temperature (β = 1/kT)
if metropolis_accept(energy_current, energy_new, beta)
    phi = phi_new  # Accept new state
    energy_current = energy_new
end

# Initialize local matrices for X and Z terms
local X_total = zeros(2^N, 2^N)
local Z_total = zeros(2^N, 2^N)

# Apply Pauli X and Z to each qubit in the system
for i in 1:N
    # Apply Pauli X to the i-th qubit
    X_term = kronN([i == j ? pauliX : I2 for j in 1:N]...)
    X_total += X_term  # Sum the contributions

    # Apply Pauli Z to the i-th qubit
    Z_term = kronN([i == j ? pauliZ : I2 for j in 1:N]...)
    Z_total += Z_term  # Sum the contributions
end

# Apply (1 / 2N) * sum(X + Z) to the state phi
operator = (X_total + Z_total) / (2 * N)
phi_modified = operator * phi

# Display the modified state
println(phi_modified)


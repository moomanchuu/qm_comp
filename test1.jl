using LinearAlgebra
using PrettyTables
using Yao
using Distributions
using Statistics

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

function Evolve(x, dist)
	S = rand(dist, n,n) # Random matrix drawn from probabilty distribution
	S = S ./ sum(S, dims=1) # normalize matrix so its Left Stochastic ie, Columns sum to 1
	U = S / sqrt(S * S')

	# pretty_table(U)
	H_new = U*H*U' # evolution via stochastic matrix. ' is shorthand for hermitian conjugate
	# pretty_table(S)
	return H_new
end

function f_metro(D, D_new, beta)
	return min(1, exp(beta*(median(diag(D)) - median(diag(D_new)))))
end



dist = Uniform(0,1)
D, _ = Diagonalize(H)
H_new = Evolve(H, dist)
D_new, _ = Diagonalize(H_new)
# # D_q = Evolve(D, dist) #Test to see if just evolving the diagonal would give the same result ---No
# diag_D = diag(D)
med_D = median(diag(D))
med_D_new = median(diag(D_new))


pretty_table(H)
pretty_table(D)
pretty_table(H_new)
pretty_table(D_new)
# 
println(med_D)
println(med_D_new)

println(f_metro(D, D_new, beta))

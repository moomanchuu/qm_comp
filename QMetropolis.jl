# Last updated 11/11/2024
using LinearAlgebra
using Random
using Distributions
using PrettyTables

# Define single-qubit Pauli matrices and identity matrix
const pauliX = [0 1; 1 0]
const pauliY = [0 -im; im 0]
const pauliZ = [1 0; 0 1]
const I2 = [1 0; 0 1]


function gibbs_state(H, beta)
    exp_neg_beta_H = exp(-beta * H)

    Z = tr(exp_neg_beta_H)
    
    rho_beta = exp_neg_beta_H / Z
    return rho_beta
end

function kronN(mats...)
    result = mats[1]
    for i in eachindex(mats)[2:end]
        result = kron(result, mats[i])
    end
    return result
end

function EvolveStateSinglePauli(state; dist=nothing)
    num_qubits = Int(log2(length(state)))
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
    return new_state, full_operator
end

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

function calculate_energies(H, state, proposed_state)
    E_current = real(state' * H * state)
    E_proposed = real(proposed_state' * H * proposed_state)
    return E_current, E_proposed
end

function calc_energy(H, state)
    E_current = real(state' * H * state)
    return E_current
end

function display_state(state, title)
    table_data = [string(state[i]) for i in 1:length(state)]
    pretty_table(table_data, header = [title])
end

function f_metro(energy, new_energy, beta)
	return min(1, exp(beta*(energy-new_energy)))
end

function run(state; H=nothing, iterations::Int=50, num_qubits::Int=2, beta=1.0, tau=0.9, r=2, g=2, check_energies=nothing)
    if H === nothing
        J = 1.0  # Coupling constant
        h = 0.5  # Transverse magnetic field
        H = transverse_ising_hamiltonian(J, h, num_qubits)
    end
    accepts = 0 
    energy = calc_energy(H, state)  # Initial energy
    for i in 1:iterations
        new_state, jump_operator = EvolveStateSinglePauli(state)
        energy = calc_energy(H, state)
        new_energy = calc_energy(H, new_state)
        #println("Energy of current state: ", energy)
        #println("Energy of proposed state: ", new_energy)

        # Decision Time
        f = f_metro(energy, new_energy, beta) * tau * tau
        if rand() < f
            state = new_state  # Accept the proposed state
            accepts += 1
            if new_energy > energy
                #println("exploring higher energy state")
            else
                #println("accepted lower energy")
            end
        end
        if i === iterations - 1 && check_energies !== nothing
            println("Final converged energy: ", energy)
        end

    end
    # Return the density matrix and the total number of acceptances
    density_matrix = state * state'
    return density_matrix, accepts, energy
end

function QMetropolisV1(num_samples::Int, num_qubits::Int; H=nothing, iterations::Int=50, beta=1.0, tau=0.77, r=2, g=2, check_energies=nothing)
    state = randn(ComplexF64, 2^num_qubits)
    state /= norm(state)  # Normalize the initial state
    total_energy = 0
    rho_sum = zeros(ComplexF64, 2^num_qubits, 2^num_qubits)
    total_accepts = 0
    # Run QMA a bunch
    for _ in 1:num_samples
        state = randn(ComplexF64, 2^num_qubits)
        state /= norm(state)  # Normalize the initial state
        rho_sample, accepts, converged_energy = run(state; H=H, iterations=iterations, num_qubits=num_qubits, beta=beta, tau=tau, r=r, g=g, check_energies=check_energies)
        rho_sum += rho_sample  
        total_accepts += accepts
        total_energy += converged_energy
    end

    rho_approx = rho_sum / num_samples
    println("Total acceptances: ", total_accepts, "/", iterations * num_samples)
    println("Average converged energy: ", total_energy/num_samples)
    return real(rho_approx) 
end

function QMetropolisV2(num_samples::Int, num_qubits::Int; H=nothing, iterations::Int=50, beta=1.0, tau=0.77, r=2, g=2, check_energies=nothing)
    total_accepts = 0
    rho_sum = zeros(ComplexF64, 2^num_qubits, 2^num_qubits)

    # Array to store rho_samples and their corresponding energies
    results = []

    # Run QMA a bunch of times, collecting energy and rho_sample
    for _ in 1:num_samples
        state = randn(ComplexF64, 2^num_qubits)
        state /= norm(state)  # Normalize the initial state
        rho_sample, accepts, converged_energy = run(state; H=H, iterations=iterations, num_qubits=num_qubits, beta=beta, tau=tau, r=r, g=g, check_energies=check_energies)

        # Collect the density matrix and energy for later sorting
        push!(results, (rho_sample, converged_energy))
        total_accepts += accepts
    end

    # Sort results by converged_energy and take the 20 lowest-energy samples
    sorted_results = sort(results, by=x -> x[2])
    lowest_20_results = sorted_results[1:min(20, length(sorted_results))]

    # Accumulate rho_sample and energy for the lowest 20 energies
    total_lowest_energy = 0
    for (rho_sample, energy) in lowest_20_results
        rho_sum += rho_sample
        total_lowest_energy += energy
    end

    # Calculate the averaged density matrix and average energy for the lowest 20 states
    rho_approx = rho_sum / min(20, length(lowest_20_results))
    average_lowest_energy = total_lowest_energy / min(20, length(lowest_20_results))

    # Print total acceptances and average energy for the lowest 20 states
    println("Total acceptances: ", total_accepts, "/", iterations * num_samples)
    println("Average converged energy of lowest 20 states: ", average_lowest_energy)
    return real(rho_approx)
end

# Use QMetropolisV2
num_samples = 50
num_qubits = 3
beta = 0.05
QMA = QMetropolisV2(num_samples, num_qubits; iterations=10000, beta=beta, tau=0.77, check_energies=nothing)

println("Approximated Gibbs State:")
pretty_table(QMA)

# directly calculate gibbs state to compare

H = transverse_ising_hamiltonian(1.0, 0.5, num_qubits)
explicit_gibbs = gibbs_state(H, beta)
println("Explicit Gibbs State:")
pretty_table(explicit_gibbs)
gibbs_energy = real(tr(H * explicit_gibbs))
println("Gibbs state energy at beta = $beta: $gibbs_energy")


# Distance metrics to check how far apart our approximation is:
trace_distance = 0.5 * sum(svdvals(explicit_gibbs - QMA))
println(trace_distance)

MAE = sum(abs.(explicit_gibbs.- QMA)) / (size(explicit_gibbs, 1) * size(QMA, 2))
println(MAE)

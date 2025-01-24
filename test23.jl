# Parameters
J = 1.0  # Coupling constant # Affects numbers on diagonal
h = 0.5  # Transverse magnetic field # Affects numbers on off diagonal
num_qubits = 2
H = transverse_ising_hamiltonian(J, h, num_qubits)

state = randn(ComplexF64, 2^num_qubits)
state /= norm(state)
new_state, jump_operator = EvolveStateSinglePauli(state)
display_state(state, "state")
display_state(new_state, "new state")
energy = calc_energy(H, state)
new_energy = calc_energy(H, new_state)
println(energy, ", ", new_energy)

f = f_metro(energy, new_energy, 1.0)
println(f)
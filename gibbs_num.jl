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

function gibbs_state(H, beta)
    # Calculates the Gibbs State of a Hamiltonian given H and Beta
    exp_neg_beta_H = exp(-beta * H)
    Z = tr(exp_neg_beta_H)
    rho_beta = exp_neg_beta_H / Z
    return rho_beta
end

H = Matrix(H_sparse_chain(3))
gibbs=gibbs_state(H, 1.0)
pretty_table(gibbs)
println(sum(diag(gibbs)))  # Should output 1.0
println(eigvals(gibbs))

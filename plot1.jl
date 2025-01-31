using JSON, Plots, LinearAlgebra

# Load the JSON file
# results_file = "results.json"
# results = JSON.parsefile(results_file)

# # Extract data from the JSON file
# N_range = 2:results["last_N"]                     # N ranges from 2 to 12
# spectral_gaps = 1 ./results["spectral_gaps"]
# computation_times = results["computation_times"]

# # Validate lengths of the data
# println("N_range length: ", length(N_range))
# println("Spectral Gaps length: ", length(spectral_gaps))
# println("Computation Times length: ", length(computation_times))

# # Plot Spectral Gaps vs N
# plot(N_range, spectral_gaps, marker=:circle, label="Spectral Gap",
#      xlabel="N", ylabel="Spectral Gap",
#      title="Spectral Gap vs N", xticks=N_range)
# savefig("spectral_gap_vs_N.png")
# println("Spectral gap plot saved as 'spectral_gap_vs_N.png'.")

# # Plot Computation Time vs N
# plot(N_range, computation_times, marker=:circle, label="Computation Time",
#      xlabel="N", ylabel="Computation Time (Log10) (s)",
#      title="Computation Time vs N", xticks=N_range, yscale=:log10)
# savefig("computation_time_vs_N.png")
# println("Computation time plot saved as 'computation_time_vs_N.png'.")

function compute_spectral_gap(P)
    vals = eigvals(P)
    sorted_vals = sort(abs.(vals), rev=true)
    return sorted_vals[1] - sorted_vals[2]  # Spectral gap is λ1 - λ2
end

function plot_convergence(filename)
    iterations = []
    differences = []
    spectral_gaps = []
    gibbs_state = nothing
    transition_matrix = nothing
    
    # Read data from JSON lines
    open(filename, "r") do io
        first_line = readline(io)
        gibbs_data = JSON.parse(first_line)
        gibbs_state = gibbs_data["gibbs_state"]
        
        for line in eachline(io)
            data = JSON.parse(line)
            push!(iterations, data["iteration"])
            push!(differences, data["difference_sq"])
            
            if transition_matrix === nothing
                transition_matrix = Matrix{Float64}(I, length(data["rho"]), length(data["rho"]))
            end
            spectral_gap = compute_spectral_gap(transition_matrix)
            push!(spectral_gaps, spectral_gap)
        end
    end
    
    # Create separate plots
    p1 = plot(iterations, differences, xlabel="Iterations", ylabel="Squared Difference (log scale)",
              title="Convergence to Gibbs State", label="Difference^2", lw=2, yscale=:log10)
    savefig(p1, "convergence_plot.png")
    
    p2 = plot(iterations, spectral_gaps, xlabel="Iterations", ylabel="Spectral Gap",
              title="Spectral Gaps Over Time", label="Spectral Gap", lw=2, linestyle=:dash)
    savefig(p2, "spectral_gap_plot.png")
end

# Run the function
plot_convergence("state_data.jsonl")
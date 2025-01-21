using JSON, Plots

# Load the JSON file
results_file = "results.json"
results = JSON.parsefile(results_file)

# Extract data from the JSON file
N_range = 2:results["last_N"]                     # N ranges from 2 to 12
spectral_gaps = results["spectral_gaps"]
computation_times = results["computation_times"]

# Validate lengths of the data
println("N_range length: ", length(N_range))
println("Spectral Gaps length: ", length(spectral_gaps))
println("Computation Times length: ", length(computation_times))

# Plot Spectral Gaps vs N
plot(N_range, spectral_gaps, marker=:circle, label="Spectral Gap",
     xlabel="N", ylabel="Spectral Gap",
     title="Spectral Gap vs N", xticks=N_range)
savefig("spectral_gap_vs_N.png")
println("Spectral gap plot saved as 'spectral_gap_vs_N.png'.")

# Plot Computation Time vs N
plot(N_range, computation_times, marker=:circle, label="Computation Time",
     xlabel="N", ylabel="Computation Time (Log10) (s)",
     title="Computation Time vs N", xticks=N_range, yscale=:log10)
savefig("computation_time_vs_N.png")
println("Computation time plot saved as 'computation_time_vs_N.png'.")
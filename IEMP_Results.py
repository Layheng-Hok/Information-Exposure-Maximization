import matplotlib.pyplot as plt
import pandas as pd

ratio_data = {
    'Dataset': ['1', '2', '3'],
    'Heuristic': [0.953, 0.981, 0.983],
    'EA (without halt)': [0.918, 0.964, 0.969],
    'EA (with halt)': [0.903, 0.959, 0.945]
}
ratio_df = pd.DataFrame(ratio_data)

# Plotting grouped bar chart for ratio values
plt.figure(figsize=(10, 6))
bar_width = 0.25
x = range(len(ratio_df['Dataset']))

# Plot each algorithm's ratio values as a set of bars
plt.bar([p - bar_width for p in x], ratio_df['Heuristic'], width=bar_width, label='Heuristic')
plt.bar(x, ratio_df['EA (without halt)'], width=bar_width, label='EA (without halt)')
plt.bar([p + bar_width for p in x], ratio_df['EA (with halt)'], width=bar_width, label='EA (with halt)')

# Graph customization
plt.xticks(x, ratio_df['Dataset'])
plt.title("Evaluation Value to Total Number of Nodes Ratio")
plt.xlabel("Datasets")
plt.ylabel("Ratio")
plt.ylim(0.9, 1.0)  # Set y-axis limits for better visual contrast
plt.legend(title="Algorithm")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

# Data setup based on provided values
data = {
    'Dataset': ['1', '2', '3'],
    'Heuristic': [18.11, 192.35, 350.03],
    'EA (without halt)': [25.27, 309.12, 925.81],
    'EA (with halt)': [2.38, 17.72, 68.82]
}
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))
for col in df.columns[1:]:  # Skip the 'Dataset' column
    plt.plot(df['Dataset'], df[col], marker='o', label=col)

# Graph customization
plt.title("Algorithm Performance Across Different Datasets")
plt.xlabel("Datasets")
plt.ylabel("Running Time (Seconds)")
plt.legend(title="Algorithm")
plt.grid(True)
plt.show()

# Data for evaluation values and running times
scatter_data = {
    'Dataset': ['Dataset 1', 'Dataset 2', 'Dataset 3'],
    'Heuristic': [(452.88, 18.11), (13722.19, 192.35), (36125.45, 320.03)],
    'EA (without halt)': [(435.91, 25.27), (13479.56, 309.12), (35591.82, 925.81)],
    'EA (with halt)': [(427.72, 2.38), (13411.19, 17.72), (34721.65, 68.82)]
}

# Convert data into a DataFrame for easier handling
scatter_df = pd.DataFrame(scatter_data)

# Plotting
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^']  # Different marker shapes for each dataset
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot each algorithm's data for each dataset
for i, alg in enumerate(scatter_df.columns[1:]):  # Skip the 'Dataset' column
    eval_values = [val[0] for val in scatter_df[alg]]  # Evaluation values
    runtimes = [val[1] for val in scatter_df[alg]]  # Running times
    plt.scatter(runtimes, eval_values, color=colors[i], marker=markers[i], s=100, label=alg)

# Graph customization
plt.title("Evaluation Values vs. Running Time for Each Algorithm and Dataset")
plt.xlabel("Running Time (seconds)")
plt.ylabel("Evaluation Value")
plt.legend(title="Algorithm")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

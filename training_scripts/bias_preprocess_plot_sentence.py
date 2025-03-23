import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File path for the preprocessed dataset
file_path = "news_bias_dataset/preprocessed_dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Map event names to the corresponding id_event values
event_names = {1: "Johnson", 2: "Facebook", 3: "NFL", 4: "NorthKorea"}
df["event_name"] = df["id_event"].map(event_names)

# Calculate percentages for bias scores across events
bias_distribution = (
    df.groupby(["event_name", "bias_score"])
    .size()
    .unstack(fill_value=0)
    .apply(lambda x: 100 * x / x.sum(), axis=1)
)
bias_distribution["Total"] = bias_distribution.sum(axis=1)

# Overall percentage distribution across all events
overall_distribution = (
    df.groupby("bias_score").size() / len(df) * 100
)

# Plot the distribution of bias scores
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x="event_name", hue="bias_score", palette="muted")

# Add numbers on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

plt.title("Distribution of Bias Scores Across Events")
plt.xlabel("Event Name")
plt.ylabel("Count")
plt.legend(title="Bias Score")
plt.tight_layout()

# Save the plot to a file
plot_path = "./plots/bias_distribution_plot_sentence.png"
plt.savefig(plot_path)

# Print the bias score distributions by event
bias_distribution_formatted = bias_distribution.applymap(lambda x: f"{x:.2f}%")
overall_distribution_formatted = overall_distribution.apply(lambda x: f"{x:.2f}%")

# Output results
print("Bias Distribution by Event:")
print(bias_distribution_formatted)
print("\nOverall Bias Distribution:")
print(overall_distribution_formatted)
print(f"\nPlot saved as {plot_path}")
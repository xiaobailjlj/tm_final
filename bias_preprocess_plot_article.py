import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (adjust file path as needed)
file_path = "news_bias_dataset/Sora_LREC2020_biasedsentences.csv"
data = pd.read_csv(file_path)

# Define the mapping for event ids to event names
event_name_mapping = {
    1: "Johnson",
    2: "Facebook",
    3: "NFL",
    4: "NorthKorea"
}

# We are interested in the `id_event` and `article_bias` columns.
data = data[['id_event', 'article_bias']]

# Filter to keep valid article_bias values (1, 2, 3, 4).
data = data[data['article_bias'].isin([1, 2, 3, 4])]

# Map `id_event` to the event names
data['event_name'] = data['id_event'].map(event_name_mapping)

# Compute percentages per event
percentages_per_event = data.groupby(['event_name', 'article_bias']).size().unstack(fill_value=0)
percentages_per_event = percentages_per_event.div(percentages_per_event.sum(axis=1), axis=0) * 100

# Compute overall percentages
overall_percentages = data.groupby('article_bias').size() / len(data) * 100

# Print percentages
print("\nPercentage Distribution Per Event:")
print(percentages_per_event)

print("\nOverall Percentage Distribution:")
print(overall_percentages)

# Plot the distribution using seaborn
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=data, x='event_name', hue='article_bias', palette='viridis')

# Add labels and title
plt.title("Distribution of Article Bias Scores Across Different Events", fontsize=14)
plt.xlabel("Event Name", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(title='Article Bias', loc='upper right', labels=['1', '2', '3', '4'])

# Annotate the bars with the count value
for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                fontsize=10, color='black',
                xytext=(0, 5), textcoords='offset points')

# Save the plot to a file
plt.tight_layout()
plt.savefig("./plots/bias_distribution_plot.png", dpi=300)

# Show the plot
plt.show()
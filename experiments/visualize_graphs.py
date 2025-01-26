import pandas as pd
import matplotlib.pyplot as plt
import sys

fname = sys.argv[1]

# Load the data from a CSV file
csv_file = fname
data = pd.read_csv(csv_file)

# Create a plot
plt.figure(figsize=(10, 6))

# Plot each column against the epoch
plt.plot(data['Epoch'], data['Training Loss'], marker='.', label='Loss', linestyle='-')
plt.plot(data['Epoch'], data['Validation Loss'], marker='.', label='Validation Loss', linestyle='-')
plt.plot(data['Epoch'], data['Validation Accuracy']/100.0, marker='.', label='Validation Accuracy', linestyle='-')

# Customize the plot
plt.title("Training Progress Over Epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Metrics", fontsize=14)
plt.xticks(range(0, 1000, 10), fontsize=12)  # Epoch steps of 10
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12, loc="best")

# Save the plot as a high-quality image suitable for journals
plt.savefig("training_progress.png", dpi=300, bbox_inches="tight")

# Display the plot
plt.show()

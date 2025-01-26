import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


# Load the data from a CSV file
csv_file = sys.argv[1]
model_name = csv_file.rsplit('.', maxsplit=1)[0]
data = pd.read_csv(csv_file)

# Interpolate data for smooth curves
epochs = data['Epoch']
smooth_epochs = np.linspace(epochs.min(), epochs.max(), 200)  # Generate more points for smoothing
print(epochs)
# Create smooth lines for each metric
loss_spline = make_interp_spline(epochs, data['Training Loss'], k=3)  # Cubic spline
# val_loss_spline = make_interp_spline(epochs, data['Validation Loss'], k=3)
accuracy_spline = make_interp_spline(epochs, data['Validation Accuracy'], k=2)

smooth_loss = loss_spline(smooth_epochs)
# smooth_val_loss = val_loss_spline(smooth_epochs)
smooth_accuracy = accuracy_spline(smooth_epochs)

# Create a plot
plt.figure(figsize=(10, 6))

# Plot smooth lines
plt.plot(smooth_epochs, smooth_loss, label='Training Loss', linestyle='-', color='blue')
# plt.plot(smooth_epochs, smooth_val_loss, label='Validation Loss', linestyle='-', color='orange')
plt.plot(smooth_epochs, smooth_accuracy, label='Validation Accuracy', linestyle='-', color='green')

# Customize the plot
plt.title(model_name, fontsize=16)
plt.xlabel("Epoch", fontsize=14)
# plt.ylabel("Metrics", fontsize=14)
plt.xticks(np.arange(0, 100, 10), fontsize=12)  # Epoch steps of 10
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12, loc="best")

# Save the plot as a high-quality image suitable for journals
plt.savefig(model_name+".png", dpi=300, bbox_inches="tight")

# Display the plot
plt.show()

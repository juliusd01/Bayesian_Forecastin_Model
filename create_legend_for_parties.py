import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

mcolors.TABLEAU_COLORS

# Define colors (manually match them from the plot)
party_colors = {
    "CDU": "tab:blue",
    "SPD": "orange",
    "Greens": "tab:green",
    "FDP": "tab:red",
    "Left": "tab:purple",
    "AfD": "tab:brown",
    "BSW": "tab:pink",
    "Other": "tab:gray"
}

# Create the legend plot
fig, ax = plt.subplots(figsize=(6, 2))

# Generate dummy handles for legend
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in party_colors.values()]
labels = party_colors.keys()

ax.legend(handles, labels, loc="center", frameon=False, ncol=4)
ax.axis("off")

plt.show()
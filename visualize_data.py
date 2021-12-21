import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
sns.set(style="darkgrid")

# Subset the iris dataset by species


sad = data.query("Class == 0")
happy = data.query("Class == 1")

# Set up the figure
f, ax = plt.subplots(figsize=(16, 16))
ax.set_aspect("equal")


# Draw the two density plots
# ax = sns.clustermap(data = [sad['t1a'], sad['t1b'], sad['t1c'], sad['t2a'], sad['t2b'], sad['t2c'], sad['t3a'], sad['t3b']],
#                  )
ax = sns.clustermap(data = [happy['t1a'], happy['t1b'], happy['t1c'], happy['t2a'], happy['t2b'], happy['t2c'], happy['t3a'], happy['t3b']],
                 )                  
# ax = sns.kdeplot(data = [sad['t1a'], sad['t1b'], sad['t1c'], sad['t2a'], sad['t2b'], sad['t2c'], sad['t3a'], sad['t3b']],
#                   shade=True)
# ax = sns.kdeplot(data = [happy['t1a'], happy['t1b'], happy['t1c'], happy['t2a'], happy['t2b'], happy['t2c'], happy['t3a'], happy['t3b']],
#                   shade=True,)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
# ax.text(20, 40, "sad", size=16, color=blue)
# ax.text(20, 35, "happy", size=16, color=red)

plt.show()
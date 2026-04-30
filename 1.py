import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('lines.csv')

# 2. Fix the column names (This handles the '# x1' issue automatically)
df.columns = [col.strip().replace('# ', '') for col in df.columns]

# 3. Choose the row you want to plot (e.g., the first row)
# Change '0' to any row index you want to visualize
row = df.iloc[0]

# 4. Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid') # Makes the plot look professional
plt.figure(figsize=(8, 6))

# 5. Define points and lines
x_points = [row['x1'], row['x2'], row['x3']]
y_points = [row['y1'], row['y2'], row['y3']]

# Plot the lines
# Connecting P1 to P2 (Blue)
plt.plot([row['x1'], row['x2']], [row['y1'], row['y2']], 
         marker='o', linestyle='-', color='blue', label='Line P1-P2', linewidth=2)
# Connecting P2 to P3 (Red)
plt.plot([row['x2'], row['x3']], [row['y2'], row['y3']], 
         marker='o', linestyle='-', color='red', label='Line P2-P3', linewidth=2)

# 6. Add Labels to Points
plt.text(row['x1'], row['y1'], ' P1', fontsize=12, fontweight='bold')
plt.text(row['x2'], row['y2'], ' P2', fontsize=12, fontweight='bold')
plt.text(row['x3'], row['y3'], ' P3', fontsize=12, fontweight='bold')

# 7. Add Titles and Labels
plt.title('Geometric Plot of Lines P1-P2 and P2-P3', fontsize=14)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 8. Show the plot
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def simulate_random_walk(num_steps):
    # Initialize arrays to store x and y coordinates
    x_positions = np.zeros(num_steps)
    y_positions = np.zeros(num_steps)
    
    # Starting point
    x_positions[0] = 0
    y_positions[0] = 0
    
    # Loop through each step
    for i in range(1, num_steps):
        # Pick a random direction: 
        # 0: Up, 1: Down, 2: Left, 3: Right
        direction = np.random.randint(0, 4)
        
        # Start from previous position
        x = x_positions[i-1]
        y = y_positions[i-1]
        
        if direction == 0:   # Up
            y += 1
        elif direction == 1: # Down
            y -= 1
        elif direction == 2: # Left
            x -= 1
        elif direction == 3: # Right
            x += 1
            
        # Update current position
        x_positions[i] = x
        y_positions[i] = y
        
    return x_positions, y_positions

# --- Parameters ---
N = 10000  # Number of steps

# --- Run Simulation ---
x, y = simulate_random_walk(N)

# --- Calculate Distance from Origin ---
# Distance formula: sqrt(x^2 + y^2)
distances = np.sqrt(x**2 + y**2)
steps = np.arange(N)

# --- Plotting ---
plt.figure(figsize=(10, 5))

# Plot 1: The Distance vs Number of Steps
plt.subplot(1, 2, 1)
plt.plot(steps, distances, color='blue', linewidth=1)
# Add a sqrt(N) curve for comparison (scaled for visibility)
plt.plot(steps, np.sqrt(steps), color='red', linestyle='--', label='Theoretical $\sqrt{N}$')
plt.title("Net Displacement vs. Steps")
plt.xlabel("Number of Steps")
plt.ylabel("Distance from Origin")
plt.legend()
plt.grid(True)

# Plot 2: The 2D Path (Visualizing the walk)
plt.subplot(1, 2, 2)
plt.plot(x, y, color='purple', alpha=0.6, linewidth=0.8)
plt.plot(0, 0, 'go', label='Start') # Green dot start
plt.plot(x[-1], y[-1], 'ro', label='End') # Red dot end
plt.title("2D Random Walk Path")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axis('equal') # Make x and y scales equal
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
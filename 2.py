import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Define the shape (Parametric Equation of an Ellipse/Teardrop)
def get_shape(t, width, height):
    # Standard ellipse: x = a*cos(t), y = b*sin(t)
    # You can customize these parametric equations to fit your specific figure
    a = width / 2
    b = height / 2
    x = a * np.cos(t)
    y = b * np.sin(t)
    return x, y

def calculate_geometry(width, height):
    # Area calculation using Green's Theorem: Area = 0.5 * integral(x*dy - y*dx)
    # Since dy = b*cos(t)dt and dx = -a*sin(t)dt
    # A = 0.5 * integral(a*b*cos^2(t) + a*b*sin^2(t)) dt = 0.5 * integral(a*b) dt
    a = width / 2
    b = height / 2
    
    # Area = pi * a * b
    area = np.pi * a * b
    
    # Centroid (for a symmetric shape centered at origin, it is 0,0)
    cx, cy = 0.0, 0.0
    
    return area, cx, cy

# 2. Define Dimensions
width = 10.0
height = 20.0

# 3. Perform Calculations
area, cx, cy = calculate_geometry(width, height)

print(f"--- Geometric Analysis ---")
print(f"Calculated Area: {area:.2f} sq units")
print(f"Centroid (Cx, Cy): ({cx}, {cy})")

# 4. Plotting
t = np.linspace(0, 2 * np.pi, 500)
x, y = get_shape(t, width, height)

plt.figure(figsize=(6, 8))
plt.fill(x, y, color='#D4AF37', alpha=0.6, label='Earring Shape')
plt.scatter([cx], [cy], color='red', label='Centroid')
plt.title(f"Area: {area:.1f}, Centroid: ({cx}, {cy})")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
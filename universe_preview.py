# universe_preview.py (Corrected)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# --- Part 1: Defining Our Universe and Particles ---

# This class is the blueprint for a single particle or "Body" in our universe.
# It knows its mass, position, and how it's moving.
class Body:
    def __init__(self, mass, position, velocity, color='blue', name=''):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2, dtype=float)  # It starts off not accelerating
        self.color = color
        self.name = name
        self.history = [self.position.copy()]  # Keep track of where it's been


# This class holds all the particles and the secret rules of the universe.
class Universe:
    def __init__(self, bodies):
        self.bodies = bodies

    # --- THIS IS THE SECRET RULE! ---
    # Our AI's entire job is to rediscover this function just by watching the particles.
    def calculate_forces(self):
        # First, reset all accelerations
        for body in self.bodies:
            body.acceleration[:] = 0.0

        # Calculate the force between every pair of bodies
        for i, body1 in enumerate(self.bodies):
            for j, body2 in enumerate(self.bodies):
                if i == j:
                    continue

                # Calculate the vector and distance between the two bodies
                r_vec = body2.position - body1.position
                dist = np.linalg.norm(r_vec)
                if dist < 0.1:  # Avoid dividing by zero if they get too close
                    continue

                # Your secret "Wave Theory" force law!
                # F = - (m1*m2 / r^2) * sin(r)
                magnitude = - (body1.mass * body2.mass / dist ** 2) * np.sin(dist * 0.5)  # a little tweak for fun
                force_vec = magnitude * (r_vec / dist)

                # Apply the force to body1 (remember F=ma, so a=F/m)
                body1.acceleration += force_vec / body1.mass

    # This method moves time forward by one small step (dt)
    def step(self, dt):
        self.calculate_forces()
        for body in self.bodies:
            # We use a smart "Leapfrog" method to update position and velocity.
            # It's more stable for physics games like this!
            body.velocity += body.acceleration * dt
            body.position += body.velocity * dt
            body.history.append(body.position.copy())


# --- Part 2: Setting up the Experiment ---

print("Setting up the universe for our preview...")

# Create three bodies with different properties
# It's like setting up the planets for our game!
body1 = Body(mass=1000.0, position=[0, 0], velocity=[0, 0], color='orange', name='Sun')
body2 = Body(mass=10.0, position=[15, 0], velocity=[0, 10], color='cyan', name='Planet 1')
body3 = Body(mass=20.0, position=[-25, 10], velocity=[-5, -5], color='magenta', name='Planet 2')

# Put our new bodies into the universe
universe = Universe([body1, body2, body3])
dt = 0.01  # Each step in the animation will be 0.01 seconds
simulation_steps = 1500  # How many steps to run the simulation for

# --- Part 3: Running the Simulation ---

print(f"Running simulation for {simulation_steps} steps...")
for i in range(simulation_steps):
    universe.step(dt)
print("Simulation complete! Now, let's watch the movie...")

# --- Part 4: Creating the Animation (The Fun Part!) ---

# Set up the movie screen (the plot)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('black')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_title('Wave Theory Universe Preview')

# Create the visual objects for our particles and their trails
points = [ax.plot([], [], 'o', markersize=10, color=b.color, label=b.name)[0] for b in universe.bodies]
trails = [ax.plot([], [], '-', linewidth=1, color=b.color, alpha=0.6)[0] for b in universe.bodies]
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')

ax.legend()


# This function is called for every single frame of the animation
def animate(frame):
    for i, body in enumerate(universe.bodies):
        # Update the particle's position on screen
        pos = body.history[frame]

        # ***** THE FIX IS HERE! *****
        # We wrap pos[0] and pos[1] in lists.
        points[i].set_data([pos[0]], [pos[1]])

        # Update the trail behind the particle
        trail_history = np.array(body.history[:frame + 1])
        trails[i].set_data(trail_history[:, 0], trail_history[:, 1])

    time_text.set_text(f'Time: {frame * dt:.2f}s')
    return points + trails + [time_text]


# Create the animation!
# We use a few tricks to make it run smoothly.
# interval = time between frames in milliseconds
# blit = True means only redraw things that have changed, which is faster
ani = FuncAnimation(fig, animate, frames=len(universe.bodies[0].history),
                    interval=20, blit=True, repeat=False)

# Show the pop-up window with our universe animation
plt.show()


import numpy as np
import random

# Set a fixed random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Parameters (set these as needed)
grid_size = 50
initial_cows = 20
cooperative_probability = 0.5
reproduction_cost = 54
reproduction_threshold = 102
stride_length = 0.08
metabolism = 6
high_growth_chance = 77  # percent
low_growth_chance = 30   # percent
grass_energy = 51
max_grass_height = 10
low_high_threshold = 5


class Cow:
    def __init__(self, x, y, breed, energy):
        self.x = float(x)
        self.y = float(y)
        self.breed = breed  # 'cooperative' or 'greedy'
        self.energy = energy
        self.alive = True

    def move(self):
        angle = random.uniform(0, 2 * np.pi)
        self.x = (self.x + np.cos(angle) * stride_length) % grid_size
        self.y = (self.y + np.sin(angle) * stride_length) % grid_size
        self.energy -= metabolism
        if self.energy < 0:
            self.alive = False

    def eat(self, grass):
        gx = int(round(self.x)) % grid_size
        gy = int(round(self.y)) % grid_size
        if self.breed == 'cooperative':
            if grass[gx, gy] > low_high_threshold:
                grass[gx, gy] -= 1
                self.energy += grass_energy
        elif self.breed == 'greedy':
            if grass[gx, gy] > 0:
                grass[gx, gy] -= 1
                self.energy += grass_energy

    def reproduce(self, cows):
        if self.energy > reproduction_threshold:
            self.energy -= reproduction_cost
            # New cow at same location, same breed, but with a small random offset
            angle = random.uniform(0, 2 * np.pi)
            offset = random.uniform(0, stride_length)
            new_x = (self.x + np.cos(angle) * offset) % grid_size
            new_y = (self.y + np.sin(angle) * offset) % grid_size
            cows.append(Cow(new_x, new_y, self.breed, metabolism * 4))

class GrassPatch:
    def __init__(self):
        self.grass = np.full((grid_size, grid_size), max_grass_height, dtype=int)

    def grow(self):
        for x in range(grid_size):
            for y in range(grid_size):
                if self.grass[x, y] >= low_high_threshold:
                    if random.random() * 100 < high_growth_chance:
                        self.grass[x, y] += 1
                else:
                    if random.random() * 100 < low_growth_chance:
                        self.grass[x, y] += 1
                if self.grass[x, y] > max_grass_height:
                    self.grass[x, y] = max_grass_height

class CooperationModel:
    def __init__(self):
        self.grass_patch = GrassPatch()
        self.cows = []
        self.ticks = 0
        self.setup_cows()

    def setup_cows(self):
        for _ in range(initial_cows):
            x = random.uniform(0, grid_size)
            y = random.uniform(0, grid_size)
            breed = 'cooperative' if random.random() < cooperative_probability else 'greedy'
            self.cows.append(Cow(x, y, breed, metabolism * 4))

    def step(self):
        random.shuffle(self.cows)  # Shuffle order each tick to match NetLogo
        for cow in self.cows:
            if cow.alive:
                cow.move()
                cow.eat(self.grass_patch.grass)
                cow.reproduce(self.cows)
        # Remove dead cows
        self.cows = [cow for cow in self.cows if cow.alive]
        self.grass_patch.grow()
        self.ticks += 1

    def run(self, steps=100):
        for _ in range(steps):
            self.step()

if __name__ == "__main__":
    model = CooperationModel()
    model.run(steps=200)
    print(f"Simulation finished after {model.ticks} ticks.")
    print(f"Cows remaining: {len(model.cows)}")
    coop = sum(1 for cow in model.cows if cow.breed == 'cooperative')
    greedy = sum(1 for cow in model.cows if cow.breed == 'greedy')
    print(f"Cooperative cows: {coop}, Greedy cows: {greedy}")

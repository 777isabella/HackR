"""
foraging_simulation_explained.py
--------------------------------
Agent-based simulation of multi-robot foraging using a server-based
waypoint system and a scalable "low-visited cluster" improvement strategy.

Each robot:
 - Explores randomly until it finds a resource
 - Returns to the nest and reports its recent memory to the server
 - The server stores "virtual pheromone" waypoints that decay over time
 - Robots can query server waypoints ONLY when at the nest

The "improved" version activates once 80% of resources are collected.
The server identifies low-visited regions (clusters) and directs robots there,
allowing the swarm to efficiently locate the final 20% of resources.
"""

import random
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# simm parameters
# ------------------------------------------------------------
GRID = 40                # size of the square grid (GRID x GRID)
N_RESOURCES = 200        # total resources randomly placed on grid
N_ROBOTS = 10            # number of robots in the swarm
MAX_ROBOT_MEMORY = 50    # maximum number of cells each robot remembers
WAYPOINT_DECAY = 0.995   # rate at which waypoint weights decay per tick
WAYPOINT_LIFETIME = 2000 # max ticks a waypoint can live before deletion
TRIALS = 5               # how many independent runs to average results over
DENSITY_R = 3            # radius for local density check around resources

# Directions for robot movement (8-neighbor connectivity)
dirs = [(0,1),(1,0),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

def clamp(x, a, b):
    """Keep value x within the range [a, b]."""
    return max(a, min(b, x))


# ------------------------------------------------------------
# serve: coordination
# ------------------------------------------------------------
class Server:
    """
    The Server stores:
      - Waypoints (like digital pheromones): cells with attraction weight and age
      - Visit counts: how often each grid cell was visited
      - Functions to decay, suggest, and detect low-visited clusters

    Robots only communicate with the Server when physically at the nest.
    """

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.waypoints = {}  # dictionary: (x,y) -> [weight, age]
        self.visit_counts = np.zeros((grid_size, grid_size), dtype=int)

    # ----------------------------
    def decay(self):
        """
        Decay all waypoints slightly each tick to simulate pheromone evaporation.
        Remove weak or old waypoints to save memory.
        """
        remove = []
        for k, v in self.waypoints.items():
            v[0] *= WAYPOINT_DECAY  # reduce weight
            v[1] += 1               # increment age
            if v[0] < 0.1 or v[1] > WAYPOINT_LIFETIME:
                remove.append(k)
        for k in remove:
            del self.waypoints[k]

    # ----------------------------
    def add_waypoint(self, pos, weight):
        """
        Add or reinforce a waypoint at position `pos` with given `weight`.
        """
        if pos in self.waypoints:
            self.waypoints[pos][0] += weight
        else:
            self.waypoints[pos] = [weight, 0]

    # ----------------------------
    def suggest_waypoints(self, k=5):
        """
        Return up to k most attractive waypoints sorted by descending weight.
        """
        return sorted(self.waypoints.items(), key=lambda x: -x[1][0])[:k]

    # ----------------------------
    def mark_visit(self, pos):
        """
        Increment the visit counter for a cell (used for low-visited cluster detection).
        """
        x, y = pos
        self.visit_counts[y, x] += 1

    # ----------------------------
    def low_visited_clusters(self, threshold=5, min_size=5):
        """
        Identify clusters (connected components) of cells that have been visited
        less than or equal to the threshold number of times.

        Returns a list of dictionaries, each describing a low-visited cluster:
        { 'center': (cx, cy), 'bbox': (xmin, ymin, xmax, ymax), 'size': n_cells }
        """
        g = (self.visit_counts <= threshold)
        visited = np.zeros_like(g, dtype=bool)
        clusters = []

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                # Skip cells that are not low-visited or already explored
                if visited[y, x] or not g[y, x]:
                    continue

                # --- BFS flood-fill to find connected low-visited region ---
                queue = [(x, y)]
                cells = []
                visited[y, x] = True
                while queue:
                    cx, cy = queue.pop()
                    cells.append((cx, cy))
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if not visited[ny, nx] and g[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((nx, ny))
                # --- End BFS ---

                # Only keep clusters above size threshold
                if len(cells) >= min_size:
                    xs, ys = zip(*cells)
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    cx, cy = np.mean(xs), np.mean(ys)
                    clusters.append({
                        'center': (int(cx), int(cy)),
                        'bbox': bbox,
                        'size': len(cells)
                    })

        return clusters


# ------------------------------------------------------------
# robot
# ------------------------------------------------------------
class Robot:
    """
    Represents one robot agent.
    Each robot has:
      - position (pos)
      - memory (recently visited locations)
      - carrying (True if currently holding a resource)
      - target (goal location from server or cluster center)
    """

    def __init__(self, nest):
        self.pos = nest              # start at nest
        self.carrying = False        # not carrying at start
        self.memory = deque(maxlen=MAX_ROBOT_MEMORY)
        self.target = None

    # ----------------------------
    def step_random(self, grid_size, server):
        """
        Move one random step in any of 8 directions and record the visit.
        """
        dx, dy = random.choice(dirs)
        x, y = self.pos
        x, y = clamp(x+dx, 0, grid_size-1), clamp(y+dy, 0, grid_size-1)
        self.pos = (x, y)
        server.mark_visit(self.pos)

    # ----------------------------
    def go_towards(self, target, server):
        """
        Move one step toward the target location.
        This is a greedy move: it just steps closer, not necessarily optimal path.
        """
        x, y = self.pos
        tx, ty = target
        dx = np.sign(tx - x)
        dy = np.sign(ty - y)
        x, y = clamp(x+dx, 0, server.grid_size-1), clamp(y+dy, 0, server.grid_size-1)
        self.pos = (x, y)
        server.mark_visit(self.pos)

    # ----------------------------
    def at(self, p):
        """Return True if robot is at position p."""
        return self.pos == p


# ------------------------------------------------------------
#helper func to compute resource density
# ------------------------------------------------------------
def local_density(resources, pos, radius):
    """Count how many resources are within a given radius of pos."""
    x, y = pos
    return sum(abs(rx - x) <= radius and abs(ry - y) <= radius for (rx, ry) in resources)


# ------------------------------------------------------------
# main simm function
# ------------------------------------------------------------
def run_trial(improved=False, visualize=False):
    """
    Run one full trial of the foraging simulation.

    Parameters:
      improved : bool
          Whether to enable the low-visited cluster improvement strategy.
      visualize : bool
          Whether to display a plot of collection progress over time.

    Returns:
      (last20_time, timeline)
          last20_time = time to collect last 20% of resources
          timeline = list of resources collected over time (for plotting)
    """

    # --- 1. Randomly place resources ---
    resources = set()
    while len(resources) < N_RESOURCES:
        resources.add((random.randrange(GRID), random.randrange(GRID)))

    nest = (GRID//2, GRID//2)
    server = Server(GRID)
    robots = [Robot(nest) for _ in range(N_ROBOTS)]

    collected = 0          # total collected so far
    timeline = []          # record of progress for plotting
    times = {}             # record when we reach 80% and 100%
    tick = 0

    # --- 2. Main time loop ---
    while collected < N_RESOURCES and tick < 20000:
        tick += 1
        server.decay()  # simulate waypoint decay each tick

        # --- Each robot acts independently ---
        for r in robots:
            # --- Case 1: Robot carrying a resource, must return to nest ---
            if r.carrying:
                r.go_towards(nest, server)
                if r.at(nest):
                    # Drop off the resource
                    collected += 1
                    r.carrying = False

                    # Report memory (recently visited cells)
                    for loc in r.memory:
                        # Weight by 1 + local resource density to emphasize good zones
                        server.add_waypoint(loc, 1 + local_density(resources, loc, DENSITY_R))
                    r.memory.clear()

            # --- Case 2: Robot NOT carrying a resource (exploring) ---
            else:
                # (a) If robot already has a target (waypoint or cluster center)
                if r.target:
                    r.go_towards(r.target, server)
                    if r.at(r.target):
                        # Once arrived, forget the target and do a small local exploration
                        r.target = None
                        r.step_random(GRID, server)

                # (b) If robot has no current target
                else:
                    if r.at(nest):
                        # Robot can query the server ONLY at the nest
                        if improved and collected >= 0.8 * N_RESOURCES:
                            # --- Improved mode: include cluster suggestions ---
                            clusters = server.low_visited_clusters()
                            wp = server.suggest_waypoints()
                            # With 80% chance: choose cluster center (if any)
                            if clusters and random.random() < 0.8:
                                r.target = random.choice(clusters)['center']
                            elif wp and random.random() < 0.8:
                                r.target = random.choice(wp)[0]
                            else:
                                # Random explore if no suggestion used
                                r.step_random(GRID, server)
                        else:
                            # --- Baseline mode: only use server waypoints ---
                            wp = server.suggest_waypoints()
                            if wp and random.random() < 0.85:
                                r.target = random.choice(wp)[0]
                            else:
                                r.step_random(GRID, server)
                    else:
                        # Robot is out in the field, just wandering randomly
                        r.step_random(GRID, server)
                        # Occasionally remember current cell (20% chance)
                        if random.random() < 0.2:
                            r.memory.append(r.pos)

                # Check if robot found a resource by landing on it
                if not r.carrying and r.pos in resources:
                    r.carrying = True
                    resources.remove(r.pos)
                    r.memory.append(r.pos)  # remember where it found one

        # --- 3. Record progress ---
        timeline.append(collected)
        if collected >= 0.8 * N_RESOURCES and 't80' not in times:
            times['t80'] = tick
        if collected == N_RESOURCES and 't100' not in times:
            times['t100'] = tick

    # --- 4. Compute time for last 20% ---
    last20 = (times.get('t100', tick) - times.get('t80', 0))

    # --- 5. Optional visualization ---
    if visualize:
        plt.figure()
        plt.plot(timeline, label='Collected')
        plt.xlabel("Time steps")
        plt.ylabel("Resources collected")
        plt.title(f"Foraging {'Improved' if improved else 'Baseline'}")
        plt.legend()
        plt.show()

    return last20, timeline


# ------------------------------------------------------------
# run multipler trials
# ------------------------------------------------------------
if __name__ == "__main__":
    baseline_results = []
    improved_results = []

    # Run several trials for statistical comparison
    for t in range(TRIALS):
        print(f"Running trial {t+1}/{TRIALS}...")
        b, _ = run_trial(improved=False)
        i, _ = run_trial(improved=True)
        baseline_results.append(b)
        improved_results.append(i)
        print(f"Trial {t+1}: baseline last20={b}, improved last20={i}")

    # Print average results
    print("\nSummary over trials:")
    print(f"Baseline avg time to collect last 20%: {np.mean(baseline_results):.2f}")
    print(f"Improved avg time to collect last 20%: {np.mean(improved_results):.2f}")

    # Show one example timeline plot
    _, _ = run_trial(improved=True, visualize=True)

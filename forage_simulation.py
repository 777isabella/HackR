import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ------------------- PARAMETERS -------------------
GRID = 40  # arena size (40x40)
N_RESOURCES = 200
N_ROBOTS = 10
MAX_ROBOT_MEMORY = 50
WAYPOINT_DECAY = 0.995
WAYPOINT_LIFETIME = 2000
TRIALS = 5
random.seed(1)
np.random.seed(1)

# ------------------- HELPER FUNCTIONS -------------------
dirs = [(0,1),(1,0),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
def clamp(x, a, b): return max(a, min(b, x))

# ------------------- SERVER -------------------
class Server:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.waypoints = {}  # (x,y): [weight, age]
        self.visit_counts = np.zeros((grid_size, grid_size), dtype=int)
    
    def decay(self):
        remove = []
        for k,(w,age) in list(self.waypoints.items()):
            w *= WAYPOINT_DECAY
            age += 1
            if w < 1e-3 or age > WAYPOINT_LIFETIME:
                remove.append(k)
            else:
                self.waypoints[k] = [w, age]
        for k in remove:
            del self.waypoints[k]
    
    def add_waypoint(self, pos, weight):
        if pos in self.waypoints:
            self.waypoints[pos][0] += weight
            self.waypoints[pos][1] = 0
        else:
            self.waypoints[pos] = [weight, 0]
    
    def suggest_waypoints(self, k=5):
        items = sorted(self.waypoints.items(), key=lambda it: it[1][0], reverse=True)
        return [pos for pos,_ in items[:k]]
    
    def mark_visit(self, pos):
        x,y = pos
        self.visit_counts[x,y] += 1

    def low_visited_clusters(self, threshold=1, min_size=3):
        g = (self.visit_counts <= threshold).astype(int)
        visited = np.zeros_like(g)
        clusters = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if g[i,j] and not visited[i,j]:
                    q = [(i,j)]
                    visited[i,j] = 1
                    comp = [(i,j)]
                    qi = 0
                    while qi < len(q):
                        cx,cy = q[qi]; qi+=1
                        for dx,dy in [(0,1),(1,0),(-1,0),(0,-1)]:
                            nx,ny = cx+dx, cy+dy
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and g[nx,ny] and not visited[nx,ny]:
                                visited[nx,ny] = 1
                                q.append((nx,ny))
                                comp.append((nx,ny))
                    if len(comp) >= min_size:
                        xs = [c[0] for c in comp]; ys=[c[1] for c in comp]
                        xmin,xmax = min(xs), max(xs)
                        ymin,ymax = min(ys), max(ys)
                        cx = (xmin+xmax)//2; cy = (ymin+ymax)//2
                        clusters.append(((cx,cy),(xmin,xmax,ymin,ymax), len(comp)))
        clusters.sort(key=lambda t: t[2], reverse=True)
        return clusters

# ------------------- ROBOT -------------------
class Robot:
    def __init__(self, start_pos, server):
        self.pos = start_pos
        self.carrying = False
        self.memory = deque(maxlen=MAX_ROBOT_MEMORY)
        self.server = server
        self.target = None
    
    def step_random(self):
        dx,dy = random.choice(dirs)
        nx = clamp(self.pos[0]+dx, 0, GRID-1)
        ny = clamp(self.pos[1]+dy, 0, GRID-1)
        self.pos = (nx,ny)
        self.server.mark_visit(self.pos)
    
    def go_towards(self, target):
        tx,ty = target
        x,y = self.pos
        dx = np.sign(tx-x); dy = np.sign(ty-y)
        nx = clamp(x+dx, 0, GRID-1)
        ny = clamp(y+dy, 0, GRID-1)
        self.pos = (nx,ny)
        self.server.mark_visit(self.pos)
    
    def at(self, p): return self.pos == p

# ------------------- SIMULATION -------------------
def run_trial(improved=False, visualize=False):
    resources = set()
    while len(resources) < N_RESOURCES:
        resources.add((random.randrange(GRID), random.randrange(GRID)))
    
    nest = (GRID//2, GRID//2)
    server = Server(GRID)
    robots = [Robot(nest, server) for _ in range(N_ROBOTS)]
    
    tick = 0
    collected = 0
    timeline = []
    times = {}
    DENSITY_R = 2

    def local_density(pos):
        x,y = pos; c = 0
        for dx in range(-DENSITY_R, DENSITY_R+1):
            for dy in range(-DENSITY_R, DENSITY_R+1):
                nx,ny = x+dx, y+dy
                if 0<=nx<GRID and 0<=ny<GRID and (nx,ny) in resources:
                    c += 1
        return c

    while collected < N_RESOURCES and tick < 20000:
        tick += 1
        for r in robots:
            if r.carrying:
                if r.pos == nest:
                    r.carrying = False
                    for loc in list(r.memory):
                        w = 1 + local_density(loc)
                        server.add_waypoint(loc, weight=w)
                    r.memory.clear()
                    r.target = None
                    continue
                else:
                    r.go_towards(nest)
                    continue
            
            if r.target is not None:
                if r.at(r.target):
                    r.target = None
                    r.step_random()
                else:
                    r.go_towards(r.target)
            else:
                if r.at(nest):
                    waypoints = server.suggest_waypoints(k=10)
                    if improved and (collected >= 0.8 * N_RESOURCES):
                        clusters = server.low_visited_clusters(threshold=1, min_size=4)
                        if clusters and (not waypoints or random.random() < 0.7):
                            center = clusters[0][0]
                            r.target = center
                        elif waypoints and random.random() < 0.9:
                            r.target = random.choice(waypoints)
                        else:
                            r.step_random()
                    else:
                        if waypoints and random.random() < 0.85:
                            r.target = random.choice(waypoints)
                        else:
                            r.step_random()
                else:
                    if random.random() < 0.2:
                        r.memory.append(r.pos)
                    r.step_random()

            if (not r.carrying) and r.pos in resources:
                r.carrying = True
                resources.remove(r.pos)
                collected += 1
                r.memory.append(r.pos)
                if collected >= 0.8 * N_RESOURCES and 't80' not in times:
                    times['t80'] = tick
                if collected == N_RESOURCES:
                    times['t100'] = tick

        server.decay()
        timeline.append(collected)

    last20 = None
    if 't80' in times and 't100' in times:
        last20 = times['t100'] - times['t80']
    elif 't100' in times:
        last20 = times['t100']

    if visualize:
        plt.plot(timeline)
        plt.xlabel('Ticks')
        plt.ylabel('Resources collected')
        plt.title(f'Collected over time (improved={improved})')
        plt.show()
    return last20, timeline

# ------------------- MAIN EXPERIMENT -------------------
res_baseline = []
res_improved = []

print("Running simulation...\n")

for t in range(TRIALS):
    last20_b, _ = run_trial(improved=False)
    last20_i, _ = run_trial(improved=True)
    res_baseline.append(last20_b or 20000)
    res_improved.append(last20_i or 20000)
    print(f"Trial {t+1}: baseline={res_baseline[-1]}, improved={res_improved[-1]}")

print("\nAverage time to collect last 20%:")
print("Baseline:", np.mean(res_baseline))
print("Improved:", np.mean(res_improved))

# Plot one example run
_, timeline_example = run_trial(improved=True, visualize=True)

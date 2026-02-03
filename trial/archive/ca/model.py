import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import random

@dataclass
class CAParams:
    """Parameters for Cell-Only CA model."""
    p_move: float = 0.5
    p_div: float = 0.05
    alpha: float = 1.0  # Density inhibition for migration
    beta: float = 1.0   # Contact inhibition for division
    edge_bonus: float = 2.0  # Extra migration probability for edge cells
    k_time: float = 1.0  # Time scaling factor (CA steps per real time unit)
    gamma: float = 0.0  # Directional bias toward wound (0=random, 1=strongly biased)

@dataclass
class CAMolParams:
    """Parameters for CA with molecular field."""
    base: CAParams = field(default_factory=CAParams)
    diffusion_rate: float = 0.1
    secretion_rate: float = 0.05
    decay_rate: float = 0.01
    chemoattraction: float = 1.0
    growth_response: float = 1.0

class CellOnlyCA:
    """
    Cell-only stochastic Cellular Automaton for wound healing.
    
    States: 0=empty, 1=cell
    """
    
    def __init__(self, height: int, width: int, params: CAParams):
        self.height = height
        self.width = width
        self.params = params
        self.grid = np.zeros((height, width), dtype=np.int32)
        self.concentration = None  # No molecular field in Phase I
        self.distance_field: Optional[np.ndarray] = None  # Distance transform from wound edge
        
    def initialize_from_mask(self, mask: np.ndarray, k: int = 1):
        """Initialize grid from downsampled binary mask."""
        from preprocess.extract_observations import downsample_binary
        self.grid = downsample_binary(mask, k).astype(np.int32)
        self.height, self.width = self.grid.shape
    
    def compute_distance_field(self):
        """
        Compute distance transform from wound edge.
        
        Creates a field where:
        - d(x) = 0 for wound pixels (empty space)
        - d(x) = distance to nearest wound pixel for cell pixels
        
        This is used for directional bias: cells prefer moving to lower d values.
        """
        from scipy.ndimage import distance_transform_edt
        
        wound_mask = (self.grid == 0).astype(np.uint8)
        
        # Distance from wound (0 in wound, increasing outward)
        self.distance_field = distance_transform_edt(wound_mask)
    
    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get Moore neighborhood (8 neighbors)."""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    neighbors.append((ni, nj))
        return neighbors
    
    def get_local_density(self, i: int, j: int) -> float:
        """Calculate local cell density."""
        neighbors = self.get_neighbors(i, j)
        occupied = sum(1 for ni, nj in neighbors if self.grid[ni, nj] > 0)
        return occupied / len(neighbors) if neighbors else 0
    
    def is_edge_cell(self, i: int, j: int) -> bool:
        """Check if cell is at the wound edge."""
        neighbors = self.get_neighbors(i, j)
        return any(self.grid[ni, nj] == 0 for ni, nj in neighbors)
    
    def migration_probability(self, i: int, j: int) -> float:
        """Calculate migration probability."""
        density = self.get_local_density(i, j)
        p = self.params.p_move * np.exp(-self.params.alpha * density)
        
        if self.is_edge_cell(i, j):
            p *= self.params.edge_bonus
        
        return min(p, 1.0)
    
    def weighted_choose_neighbor(self, i: int, j: int, empty_neighbors: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Choose a neighbor to migrate to, with optional directional bias toward wound.
        
        If gamma > 0, prefers neighbors that reduce distance to wound.
        
        Args:
            i, j: Current cell position
            empty_neighbors: List of empty neighbor positions
            
        Returns:
            Chosen neighbor position
        """
        if self.params.gamma <= 0 or len(empty_neighbors) == 1:
            return random.choice(empty_neighbors)
        
        # Compute distance field if not already computed
        if self.distance_field is None:
            self.compute_distance_field()
        
        # Calculate weights based on distance change
        weights = []
        
        for ni, nj in empty_neighbors:
            new_dist = self.distance_field[ni, nj] if self.distance_field is not None else 0
            # Prefer moving to lower distance (toward wound)
            # Weight = exp(-gamma * distance_to_wound)
            weight = np.exp(-self.params.gamma * new_dist)
            weights.append(weight)
        
        # Normalize weights (with safety checks)
        weights = np.array(weights, dtype=np.float64)
        weight_sum = weights.sum()
        
        if weight_sum == 0 or not np.isfinite(weight_sum):
            # Fallback to uniform random if weights are invalid
            return random.choice(empty_neighbors)
        
        weights = weights / weight_sum
        
        # Additional safety: check for NaN or Inf after normalization
        if not np.all(np.isfinite(weights)):
            return random.choice(empty_neighbors)
        
        # Weighted random choice
        try:
            idx = np.random.choice(len(empty_neighbors), p=weights)
            return empty_neighbors[idx]
        except (ValueError, RuntimeError):
            # Fallback to uniform random if sampling fails
            return random.choice(empty_neighbors)
    
    def division_probability(self, i: int, j: int) -> float:
        """Calculate division probability."""
        density = self.get_local_density(i, j)
        p = self.params.p_div * np.exp(-self.params.beta * density)
        return min(p, 1.0)
    
    def step(self) -> Dict[str, float]:
        """Perform one CA step (async update)."""
        # Get all cell positions
        cell_positions = [(i, j) for i in range(self.height) 
                         for j in range(self.width) if self.grid[i, j] > 0]
        
        # Random order for asynchronous update
        random.shuffle(cell_positions)
        
        migration_count = 0
        division_count = 0
        
        for i, j in cell_positions:
            action = random.random()
            
            # Try migration first
            if action < self.migration_probability(i, j):
                neighbors = [(ni, nj) for ni, nj in self.get_neighbors(i, j) 
                            if self.grid[ni, nj] == 0]
                
                if neighbors:
                    # Use weighted choice if directional bias is enabled
                    ni, nj = self.weighted_choose_neighbor(i, j, neighbors)
                    self.grid[ni, nj] = 1
                    self.grid[i, j] = 0
                    migration_count += 1
            
            # Then try division
            elif action < self.migration_probability(i, j) + self.division_probability(i, j):
                neighbors = [(ni, nj) for ni, nj in self.get_neighbors(i, j) 
                            if self.grid[ni, nj] == 0]
                
                if neighbors:
                    ni, nj = random.choice(neighbors)
                    self.grid[ni, nj] = 1
                    division_count += 1
        
        return {
            "migrations": migration_count,
            "divisions": division_count,
        }
    
    def run(self, num_steps: int) -> List[Dict[str, float]]:
        """Run simulation for multiple steps."""
        history = []
        
        for _ in range(num_steps):
            stats = self.step()
            
            # Calculate observables
            wound_mask = 1 - self.grid
            wound_area = np.sum(wound_mask)
            
            stats["wound_area"] = float(wound_area)
            stats["step"] = len(history)
            
            history.append(stats)
        
        return history
    
    def get_observations(self) -> np.ndarray:
        """Get wound area time series."""
        return np.array([1 - self.grid])

class CAMolField:
    """
    CA with molecular field (Phase II).
    
    States: 0=empty, 1=cell
    Field: continuous concentration C
    """
    
    def __init__(self, height: int, width: int, params: CAMolParams):
        self.height = height
        self.width = width
        self.params = params
        self.grid = np.zeros((height, width), dtype=np.int32)
        self.concentration = np.zeros((height, width), dtype=np.float32)
    
    def initialize_from_mask(self, mask: np.ndarray, k: int = 1):
        """Initialize grid from downsampled binary mask."""
        from preprocess.extract_observations import downsample_binary
        self.grid = downsample_binary(mask, k).astype(np.int32)
        self.height, self.width = self.grid.shape
        self.concentration = np.zeros_like(self.grid, dtype=np.float32)
    
    def diffuse_field(self):
        """Diffuse molecular field."""
        laplacian = (
            np.roll(self.concentration, 1, axis=0) +
            np.roll(self.concentration, -1, axis=0) +
            np.roll(self.concentration, 1, axis=1) +
            np.roll(self.concentration, -1, axis=1) -
            4 * self.concentration
        )
        self.concentration += self.params.diffusion_rate * laplacian
    
    def decay_field(self):
        """Decay molecular field."""
        self.concentration *= (1 - self.params.decay_rate)
    
    def secrete_field(self):
        """Edge cells secrete molecules."""
        edge_cells = []
        
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:
                    # Check if at edge
                    is_edge = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.height and 0 <= nj < self.width:
                                if self.grid[ni, nj] == 0:
                                    is_edge = True
                                    break
                        if is_edge:
                            break
                    
                    if is_edge:
                        edge_cells.append((i, j))
        
        for i, j in edge_cells:
            self.concentration[i, j] += self.params.secretion_rate
    
    def update_field(self):
        """Update molecular field: diffusion + secretion + decay."""
        self.diffuse_field()
        self.secrete_field()
        self.decay_field()
    
    def get_chemoattraction(self, i: int, j: int) -> float:
        """Calculate chemoattraction gradient magnitude."""
        # Simple gradient using finite differences
        grad_x = (self.concentration[i, min(j+1, self.width-1)] - 
                  self.concentration[i, max(j-1, 0)])
        grad_y = (self.concentration[min(i+1, self.height-1), j] - 
                  self.concentration[max(i-1, 0), j])
        
        return np.sqrt(grad_x**2 + grad_y**2)
    
    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get Moore neighborhood."""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    neighbors.append((ni, nj))
        return neighbors
    
    def get_local_density(self, i: int, j: int) -> float:
        """Calculate local density."""
        neighbors = self.get_neighbors(i, j)
        occupied = sum(1 for ni, nj in neighbors if self.grid[ni, nj] > 0)
        return occupied / len(neighbors) if neighbors else 0
    
    def migration_probability(self, i: int, j: int) -> float:
        """Migration with chemoattraction."""
        density = self.get_local_density(i, j)
        chemo = self.get_chemoattraction(i, j)
        conc = self.concentration[i, j]
        
        # Sigmoid-based probability
        logits = (np.log(self.params.base.p_move) - 
                 self.params.base.alpha * density +
                 self.params.chemoattraction * chemo -
                 self.params.base.edge_bonus * density)
        
        return 1.0 / (1.0 + np.exp(-logits))
    
    def division_probability(self, i: int, j: int) -> float:
        """Division with growth factor."""
        density = self.get_local_density(i, j)
        conc = self.concentration[i, j]
        
        logits = (np.log(self.params.base.p_div) - 
                 self.params.base.beta * density +
                 self.params.growth_response * conc)
        
        return 1.0 / (1.0 + np.exp(-logits))
    
    def step(self) -> Dict[str, float]:
        """One step with molecular field."""
        # Update field first
        self.update_field()
        
        # Get all cell positions
        cell_positions = [(i, j) for i in range(self.height) 
                         for j in range(self.width) if self.grid[i, j] > 0]
        
        # Async update
        import random
        random.shuffle(cell_positions)
        
        migration_count = 0
        division_count = 0
        
        for i, j in cell_positions:
            action = random.random()
            
            if action < self.migration_probability(i, j):
                neighbors = [(ni, nj) for ni, nj in self.get_neighbors(i, j) 
                            if self.grid[ni, nj] == 0]
                
                if neighbors:
                    # Prefer higher concentration
                    probs = np.array([self.concentration[ni, nj] for ni, nj in neighbors])
                    probs = probs / (probs.sum() + 1e-10)
                    idx = np.random.choice(len(neighbors), p=probs)
                    ni, nj = neighbors[idx]
                    
                    self.grid[ni, nj] = 1
                    self.grid[i, j] = 0
                    migration_count += 1
            
            elif action < self.migration_probability(i, j) + self.division_probability(i, j):
                neighbors = [(ni, nj) for ni, nj in self.get_neighbors(i, j) 
                            if self.grid[ni, nj] == 0]
                
                if neighbors:
                    ni, nj = random.choice(neighbors)
                    self.grid[ni, nj] = 1
                    division_count += 1
        
        return {
            "migrations": migration_count,
            "divisions": division_count,
            "wound_area": float(np.sum(1 - self.grid)),
        }
    
    def run(self, num_steps: int) -> List[Dict[str, float]]:
        """Run simulation."""
        history = []
        
        for step in range(num_steps):
            stats = self.step()
            stats["step"] = step
            history.append(stats)
        
        return history

if __name__ == "__main__":
    # Test basic CA
    params = CAParams(p_move=0.5, p_div=0.05, alpha=1.0, beta=1.0, edge_bonus=2.0)
    ca = CellOnlyCA(50, 50, params)
    
    # Initialize with a wound in the center
    ca.grid[:] = 1
    ca.grid[15:35, 15:35] = 0
    
    history = ca.run(num_steps=100)
    
    print(f"Simulation complete. Final wound area: {history[-1]['wound_area']}")
    print(f"Initial wound area: {history[0]['wound_area']}")

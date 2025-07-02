# SRAM Optimization Algorithms

This directory implements various optimization algorithms for SRAM circuit parameter tuning, optimizing SNM (Static Noise Margin), power consumption, and area subject to timing constraints.

## Algorithms

### 1. Constrained Bayesian Optimization (CBO)
**File**: `sram_cbo.py`

Primary algorithm using Gaussian Process surrogate models with constrained multi-objective optimization.
- Constrained Expected Improvement acquisition
- Pareto front tracking
- 20 random + 380 BO iterations

**Dependencies**:
```bash
pip install torch botorch gpytorch
```

### 2. Particle Swarm Optimization (PSO)
**File**: `pso.py`

Swarm intelligence approach with velocity-based particle updates.
- Population size: 20
- Merit-based fitness evaluation

**Dependencies**: Standard libraries (numpy, torch, matplotlib)

### 3. Simulated Annealing (SA)
**File**: `sa.py`

Temperature-based metaheuristic with adaptive cooling.
- Metropolis acceptance criterion
- Restart mechanism for exploration

**Dependencies**: Standard libraries (numpy, torch, matplotlib)

### 4. RoSE-Opt (BO + RL)
**File**: `rose_opt.py`

Hybrid approach combining Bayesian Optimization with Reinforcement Learning (PPO).
- GP model for global exploration
- RL agent for local refinement

**Dependencies**:
```bash
pip install gymnasium scikit-learn scipy tqdm
```

### 5. SMAC
**File**: `sram_smac.py`

Model-based optimization using Sequential Model-based Algorithm Configuration.
- Random Forest surrogate models
- Expected Improvement acquisition

**Dependencies**:
```bash
pip install smac ConfigSpace
```

## Configuration

All algorithms support configuration files for circuit parameter definition:
- **Configuration file**: `config_sram.yaml`
- Define parameter spaces, constraints, and simulation settings
- Easy adaptation to different circuits without code modification

## Usage

### Individual Algorithms
```bash
python sram_cbo.py      
python pso.py  
python sa.py   
python rose_opt.py 
python sram_smac.py 
```

### Automated Comparison
```bash
python run_experiments.py
```
Runs multiple algorithms and generates comparison reports.

## Common Dependencies

All algorithms require:
```bash
pip install numpy pandas matplotlib scipy torch pyyaml
```

## Output

All algorithms generate CSV results, Merit tracking, and Pareto front visualizations in `sim/opt/results/` and `sim/opt/plots/`.

## Extended Algorithm Repository

For additional advanced optimization algorithms and cutting-edge techniques in circuit optimization:

**[AIxAnalog](https://github.com/IceLab-X/AIxAnalog)** (Coming Soon)

This repository will be available shortly.

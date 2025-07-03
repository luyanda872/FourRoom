# 🤖 Reinforcement Learning in FourRooms Domain

<div align="center">
  
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

**🎯 Advanced Q-Learning Implementation | 🌟 Multi-Scenario Training | 🔬 Stochastic & Deterministic Modes**

</div>

---

## 🚀 Project Overview

This project implements **sophisticated Q-learning agents** in the classic FourRooms grid environment, exploring different package collection scenarios with increasing complexity. Features both **deterministic** and **stochastic** environments with advanced epsilon-greedy exploration strategies.

> **⚡ Windows Compatible** - Fully tested on Windows with GIT Bash support!

---

## 🎮 Scenario Breakdown

<div align="center">

| 🎯 Scenario | 📦 Task | 🧠 Strategy | 🎲 Difficulty |
|-------------|---------|-------------|----------------|
| **Scenario 1** | Collect **1 package** | Fixed & Decaying ε | ⭐ Beginner |
| **Scenario 2** | Collect **3 packages** (any order) | Decaying ε | ⭐⭐ Intermediate |
| **Scenario 3** | Collect **3 color-coded packages** (R→G→B) | Decaying ε | ⭐⭐⭐ Advanced |

</div>

---

## 📁 Project Architecture

```
FourRooms-RL/
├── 🏠 FourRooms.py         # Core environment (predefined)
├── 🎯 Scenario1.py         # Single package collection
├── 🎯 Scenario2.py         # Multi-package collection
├── 🎯 Scenario3.py         # Sequential color-coded collection
├── 📄 requirements.txt     # Python dependencies
├── 📄 Makefile            # Automated commands
└── 📄 README.md           # Project documentation
```

### 🔧 File Descriptions

#### `FourRooms.py`
- **Core Environment** - Predefined grid world (do not modify)
- **State Management** - Handles agent positioning and package tracking
- **Action Processing** - Manages movement and collection mechanics

#### `Scenario1.py` 
- **Single Package Collection** - Learn to collect 1 package efficiently
- **Dual Strategy Support** - Both fixed and decaying epsilon
- **Baseline Performance** - Foundation for complex scenarios

#### `Scenario2.py`
- **Multi-Package Collection** - Collect 3 packages in any order
- **Strategic Planning** - Agent learns optimal collection routes
- **Decaying Exploration** - Sophisticated epsilon scheduling

#### `Scenario3.py`
- **Sequential Collection** - Collect R→G→B packages in strict order
- **Advanced Planning** - Complex state-action mapping
- **Constraint Satisfaction** - Order-dependent reward structure

---

## 🧠 Exploration Strategies

### 🎯 Fixed Epsilon (ε = 0.1)
- **Constant Exploration**: Maintains steady 10% random action rate
- **Stable Learning**: Consistent exploration throughout training
- **Best For**: Simple environments with clear optimal paths

### 📉 Decaying Epsilon (ε = 1.0 → 0.05)
- **Adaptive Exploration**: Starts with 100% exploration, decays to 5%
- **Efficient Learning**: High initial exploration, then exploitation
- **Best For**: Complex environments requiring strategic planning

---

## 🎮 Quick Start Guide

### 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use Makefile
make setup
```

### 🚀 Running Scenarios

#### Deterministic Mode (Predictable Actions)
```bash
python Scenario1.py
python Scenario2.py
python Scenario3.py
```

#### Stochastic Mode (20% Action Randomness)
```bash
python Scenario1.py --stochastic
python Scenario2.py --stochastic
python Scenario3.py --stochastic
```

---

## 🛠️ Makefile Commands

> **💻 Windows Users**: Use GIT Bash for full compatibility!

### Setup & Dependencies
```bash
make setup          # Install all requirements
```

### Scenario Execution
```bash
make run1           # Scenario 1 (Deterministic)
make run1-stoch     # Scenario 1 (Stochastic)
make run2           # Scenario 2 (Deterministic)
make run2-stoch     # Scenario 2 (Stochastic)
make run3           # Scenario 3 (Deterministic)
make run3-stoch     # Scenario 3 (Stochastic)
```

### Environment Management
```bash
make clean          # Clean generated files
```

---

## 📊 Generated Outputs

Each scenario produces comprehensive analysis files:

### 📈 Learning Curves
- `learning_curves_both_<mode>.png` - Training progress visualization
- **Tracks**: Episode rewards, convergence rate, exploration decay

### 🗺️ Path Visualization
- `final_path_<strategy>_<mode>.png` - Agent's optimal path
- **Shows**: Start/end positions, package locations, learned route

### 📝 Execution Logs
```plaintext
🎯 Training Progress:
Episode 1000: Average Reward = 15.2, Epsilon = 0.1
Episode 2000: Average Reward = 18.7, Epsilon = 0.05

🚀 Final Run:
[Step 4] Action: RIGHT -> (5, 6) | Grid: RED | Packages left: 2
📦 Package collected!
🏁 Reached terminal state
```

---

## 🔬 Technical Features

### 🎯 Q-Learning Implementation
- **Temporal Difference Learning**: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- **Dynamic Learning Rate**: Adaptive α based on state visitation
- **Discount Factor**: γ = 0.95 for future reward consideration

### 🌟 Advanced Exploration
- **Epsilon Scheduling**: Linear decay with minimum threshold
- **Action Selection**: Balanced exploration vs exploitation
- **State Space Coverage**: Efficient environment mapping

### 📊 Performance Metrics
- **Convergence Analysis**: Training stability assessment
- **Path Optimization**: Route efficiency measurement
- **Success Rate**: Task completion percentage

---

## 🎯 Environment Modes

### 🔒 Deterministic Mode
- **Predictable Actions**: Agent moves exactly as intended
- **Consistent Results**: Reproducible learning outcomes
- **Ideal For**: Algorithm testing and baseline performance

### 🎲 Stochastic Mode
- **20% Action Randomness**: Adds realistic uncertainty
- **Robust Learning**: Handles environmental noise
- **Real-World Simulation**: More challenging and realistic

---

## 📋 Requirements

- **Python 3.7+**
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization and plotting
- **Windows**: GIT Bash recommended for Makefile support

---

## 🌟 Key Achievements

- ✅ **Multi-Scenario Learning**: Progressive difficulty scaling
- ✅ **Dual Environment Support**: Deterministic & stochastic modes
- ✅ **Advanced Exploration**: Fixed & decaying epsilon strategies
- ✅ **Comprehensive Visualization**: Learning curves & path analysis
- ✅ **Cross-Platform Support**: Windows-compatible automation
- ✅ **Professional Documentation**: Clear setup and usage guide

---

## 🚀 Future Enhancements

- 🔄 **Deep Q-Learning**: Neural network-based Q-function
- 🎯 **Multi-Agent Systems**: Collaborative package collection
- 🌐 **Custom Environments**: User-defined grid configurations
- 📊 **Advanced Metrics**: Detailed performance analytics

---

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

---

## 📧 Contact

**Luyanda**  
📧 MBLLUY007@myuct.ac.za

---

<div align="center">
  
**⭐ Star this repo if you found it helpful!**

*Building intelligent agents, one Q-value at a time* 🤖

</div>
<div align=center>
  
# Information Exposure Maximization Problem (IEMP)

</div>

This repository explores the **Information Exposure Maximization Problem (IEMP)** and presents heuristic and evolutionary algorithmic solutions for balancing exposure to diverse information in social networks. The study is motivated by the need to counteract echo chambers and filter bubbles by optimizing the influence spread of competing campaigns.

The README below is just an overview of the project. For details, please refer to the [report](https://github.com/Layheng-Hok/Information-Exposure-Maximization/blob/main/ref/IEMP_Report.pdf).

## Abstract
The IEMP is framed as an **optimization problem** within a social network, represented as a **graph** where nodes represent users and edges represent their social connections. Two distinct campaigns attempt to spread their messages within this network. The objective is to select seed sets of users that maximize exposure to both campaigns, thereby reducing polarization. 

The study evaluates two algorithmic approaches:
1. **Heuristic Algorithm:** A Monte Carlo-based greedy selection strategy.
2. **Evolutionary Algorithm:** A genetic algorithm designed to optimize influence spread.

### Key Findings:
- The heuristic method **consistently outperforms** the evolutionary algorithm in both efficiency and evaluation value.
- The evolutionary approach with a **halt condition** shows promise in reducing computational time.

## Methodology
### Problem Formulation
Given a social network **G = (V, E)** with two campaigns and an influence budget, the objective is to identify seed sets **S1** and **S2** such that:

$$
\max \Phi(S_1, S_2) = \mathbb{E}[|V \setminus (r_1 (I_1 \cup S_1) \triangle r_2 (I_2 \cup S_2))|]
$$

where:

$$
|S_1| + |S_2| \leq k, \quad S_1, S_2 \in V
$$

### Algorithms Implemented
#### 1. Heuristic Search (Monte Carlo Greedy Algorithm)
- Selects nodes based on incremental influence value.
- Uses Monte Carlo simulations to approximate influence impact.
- Limits unnecessary recomputation by only considering immediate neighbors.

#### 2. Evolutionary Algorithm (Genetic Algorithm)
- Represents solutions as binary vectors.
- Utilizes fitness functions to evaluate influence spread.
 
$$
fitness(S_1, S_2) =
\begin{cases} 
\Phi(S_1, S_2) & \text{if } |S_1| + |S_2| \leq k \\
-\Phi(S_1, S_2) & \text{otherwise}
\end{cases}
$$
  
- Implements selection, two-point crossover, and flip-bit mutation operations.
- Introduces a halt condition to reduce computational overhead.

## Experiments
### Setup
- **Datasets:** Three datasets with varying scales and structures. Dataset 1 provided a baseline with a smaller network size, while Dataset 2 introduced a moderate level of complexity. Dataset 3, the largest, presented additional challenges due to its scale and influence probabilities skewed towards lower values, which could limit spread and increase unpredictability.

| Dataset | Number of Nodes | Number of Edges |
|---------|---------------|---------------|
| 1       | 475           | 13289         |
| 2       | 13984         | 17319         |
| 3       | 36742         | 49248         |

- **Platform:** macOS Sequoia 15.0.1, Apple M3 Pro processor with 18GB of RAM and 1TB storage capacity, Python 3.12.

### Results
To obtain evaluation values, we ran independent cascade to simulate the influence spread on each outputted balanced seed set for 300 times. We also recorded the running time in the process.

In the halt condition of the evolutionary algorithm, we only ran independent cascade 10 times for each instance, which resulted in a high but less accurate evaluation value. This allowed the halt condition to satisfy and terminate the algorithm early, even though the actual ratio upon running 300 simulations is not that large.

| Algorithm | Dataset 1 | Dataset 2 | Dataset 3 |
|-----------|----------|----------|----------|
| Heuristic | 0.953 | 0.981 | 0.983 |
| EA (No Halt) | 0.918 | 0.964 | 0.969 |
| EA (With Halt) | 0.903 | 0.959 | 0.945 |

<div>
  <img src="https://github.com/Layheng-Hok/Information-Exposure-Maximization/blob/main/ref/img/Fig1.png" width="50%">
  <img src="https://github.com/Layheng-Hok/Information-Exposure-Maximization/blob/main/ref/img/Fig2.png" width="50%">
  <img src="https://github.com/Layheng-Hok/Information-Exposure-Maximization/blob/main/ref/img/Fig3.png" width="50%">
</div>

## Conclusion
- **Heuristic method is the most effective and efficient approach**.
- **Evolutionary algorithm with a halt condition provides a balance** between accuracy and computational cost.
- Future work may explore **hybrid approaches** combining heuristic and genetic algorithms.

## References
- D. Bahdanau, K. Cho, and Y. Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate," in *3rd International Conference on Learning Representations, ICLR 2015*, San Diego, CA, USA, May 7-9, 2015. [Online]. Available: [http://arxiv.org/abs/1409.0473](http://arxiv.org/abs/1409.0473)
- K. Garimella, A. Gionis, N. Parotsidis, N. Tatti. "Balancing Information Exposure in Social Networks." *NeurIPS 2017*, pp. 4663-4671.
- Q. Guo et al., "A Survey on Knowledge Graph-Based Recommender Systems," *IEEE Transactions on Knowledge and Data Engineering*, pp. 1–1, 2020, doi: 10/ghxwqg.

# Comparison-based Active Preference Learning for Multi-dimensional Personalization
Official implementation of [Comparison-based Active Preference Learning for Multi-dimensional Personalization](https://arxiv.org/abs/2411.00524) accepted by ACL 2025 Main (Long). You can visit [our project page](https://minhyeonoh.github.io/research/AMPLe).

# Experiments

The main script (`main.py`) runs individual iterations of preference learning experiments. It simulates interactions with a user using various reward model components, preference priors, and acquisition strategies.

## Usage

The script is executed via the command line with several required arguments defining the task, hyperparameters, and experiment settings.

```bash
python main.py [OPTIONS]
```

### Key Arguments

#### 1. Task & Model Configuration

  * `--task`: The specific domain or dataset to run.
      * *Options:* `summary`, `assistant`, `geval`, `summeval+geval`
  * `--rm_names`: Space-separated list of reward model components to use.
      * *Example (for assistant):* `harmless helpful humor`
      * *Example (for geval):* `coherence consistency fluency relevance`
  * `--inputs`: The input selection strategy or specific input ID.
      * *Options:* `0`, `28`, `all`, `dynamic` (Dependent on task).

#### 2\. Preference & Simulation Parameters

  * `--true_preference`: The ground truth weights for the reward components (a space-separated string of floats summing to 1).
      * *Example:* `"0.2 0.7 0.1"`
  * `--true_beta`: The inverse temperature for the ground truth generation (simulating user noise).
      * *Values:* `10`, `inf` (Infinity implies deterministic preferences).
  * `--beta`: The inverse temperature used for the model's inference.
      * *Values:* `1`, `2`, `5`, `inf`.
  * `--seed`: Random seed for reproducibility (e.g., `1`, `2`, `3`...).

#### 3\. Acquisition & Learning

  * `--acquisition`: The active learning strategy for acquiring new data.
      * *Options:* `volume` (uncertainty/volume sampling), `random`.
  * `--mode`: The learning mode.
      * *Default:* `likelihood`.
  * `--n_rounds`: Number of active learning rounds to perform (Default: `100`).

## Example Commands

Below are examples of how to run the script manually, based on common configurations found in the launcher.

**1. "Assistant" Task (Standard Run):**

```bash
python main.py \
  --seed=1 \
  --task=assistant \
  --rm_names harmless helpful humor \
  --n_rounds=100 \
  --true_preference "0.2 0.7 0.1" \
  --true_beta=10 \
  --beta=2.0 \
  --jitter=0.01 \
  --alpha=10 \
  --init=previous \
  --acquisition=volume \
  --burnin=50000 \
  --gamma=0.0 \
  --mode=likelihood \
  --margin=none \
  --inputs=0 \
  --dims 0 1
```

**2. "GEval" Task (Infinite Beta):**

```bash
python main.py \
  --seed=3 \
  --task=geval \
  --rm_names coherence consistency fluency relevance \
  --n_rounds=100 \
  --true_preference "0.1 0.2 0.3 0.4" \
  --true_beta=inf \
  --beta=inf \
  --jitter=0.01 \
  --alpha=10 \
  --init=previous \
  --acquisition=random \
  --burnin=50000 \
  --gamma=0.3 \
  --mode=likelihood \
  --margin=none \
  --inputs=dynamic \
  --dims 0 1
```

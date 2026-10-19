
# Biologically Plausible Secure MPC for Privacy-Preserving AI Training

Graduation project investigating whether Forward-Forward (FF) learning can improve the efficiency of privacy-preserving neural network training using Multi-Party Computation (MPC).

## Project Goals

- Establish plaintext Backpropagation (BP) and Forward-Forward baselines on MNIST.
- Compare secure BP and secure FF under a documented MPC protocol and security model.
- Profile computation and communication costs.
- Evaluate one MPC-aware modification of FF based on measured bottlenecks.

## Project Structure

- `src/` — experimental Python implementations.
- `configs/` — experiment configurations.
- `results/` — small experimental result files.
- `docs/` — research and experiment notes.

## Environment

- Ubuntu 26.04 under WSL2
- Python 3.14.4
- PyTorch 2.14.1 (CPU)
- Torchvision 0.29.1 (CPU)
- MP-SPDZ is maintained separately at `~/MP-SPDZ`.

## Current Progress

- MP-SPDZ MASCOT tutorial runs successfully.
- MNIST has been downloaded and verified.
- Initial plaintext BP and FF smoke tests have been executed.
- Baseline implementations and experimental settings are still under development.

## Run the Initial Experiments

Activate the Python environment:

```bash
source ~/venvs/mp-spdz-ml/bin/activate
```

Run Backpropagation:

```bash
python src/bp_mnist.py
```

Run Forward-Forward:

```bash
python src/ff_mnist.py
```

These are preliminary smoke tests, not final benchmark results.

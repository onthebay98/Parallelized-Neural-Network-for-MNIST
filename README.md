# Parallelized Neural Network for MNIST

Ensemble training of neural networks on the MNIST handwritten digit dataset using work-stealing and work-balancing parallelism. All matrix operations, gradient descent, forward propagation, and backpropagation are implemented from scratch in Go — no external ML libraries.

## How It Works

The network (784 → 10 → 10, ReLU + Softmax) is trained via **data-parallel ensemble learning**: the 60,000 training images are split into 60 chunks of 1,000, each chunk trains an independent model, and the final weights and biases are averaged.

Two parallel schedulers distribute these training tasks across goroutines:

- **Work-stealing** — each worker has a local deque; idle workers steal tasks from a random peer
- **Work-balancing** — workers probabilistically rebalance queues when load asymmetry exceeds a threshold

Both are compared against a sequential baseline.

## Architecture

```
editor/editor.go            # CLI entry point — parses mode, threads, epochs
scheduler/
├── scheduler.go            # Orchestration: sequential vs parallel execution
├── neuralnetwork.go        # Forward/back prop, gradient descent, ensemble averaging
└── helpers.go              # MNIST loading, normalization, data transposition
concurrent/
├── concurrent.go           # Interfaces: Runnable, Future, ExecutorService
├── stealing.go             # Work-stealing executor with per-worker deques
├── balancing.go            # Work-balancing executor with probabilistic rebalancing
└── unbounded.go            # Lock-protected unbounded deque (doubly-linked list)
mnist/mnist.go              # MNIST binary format parser
benchmark/
├── benchmark-proj3.sh      # SLURM cluster job script
└── speedup.py              # Speedup analysis across thread counts and epochs
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Ensemble averaging over data parallelism within a single model | Eliminates gradient synchronization overhead — each worker trains independently |
| Unbounded deque via linked list | Simplifies growth without resize logic; bidirectional access supports both consumer and thief |
| Probabilistic balancing trigger (1/(n+1)) | Reduces balancing overhead when queues are already well-loaded |
| All matrix ops from scratch | Course requirement — demonstrates understanding of the underlying linear algebra |

## Usage

```bash
cd neuralnetwork
go build -o nn ./editor

# Sequential: 25 epochs
./nn 25 s

# Work-stealing: 25 epochs, 8 threads
./nn 25 ws 8

# Work-balancing: 25 epochs, 8 threads
./nn 25 wb 8
```

MNIST data files (`train-images-idx3-ubyte.gz`, etc.) should be in the working directory.

## Tech Stack

Go (no external dependencies)

# Agent for RoomsEnv

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.10 or higher.
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly
   recommended so that you don't mess up with the system python.
1. Install the requirements by running `pip install -r requirements.txt`

## Jupyter Notebooks

- [Creating rooms environment](create-rooms.ipynb)
- [Training our agent](train-dqn.ipynb)
- [Training baseline agent](train-dqn-baselines.ipynb)

## Training Results

| Capacity | Agent Type   | Phase 1   | Phase 2       |
| -------- | ------------ | --------- | ------------- |
| 12       | **Ours (E)** | 191 (±42) | **194** (±29) |
|          | Ours         | 105 (±37) | 160 (±30)     |
|          | Baseline     | N/A       | 144 (±14)     |
|          | Ours (S)     | 111 (±43) | 124 (±65)     |
| 24       | **Ours**     | 127 (±26) | **214** (±64) |
|          | Ours (E)     | 227 (±21) | 209 (±30)     |
|          | Baseline     | N/A       | 138 (±52)     |
|          | Ours (S)     | 98 (±45)  | 112 (±79)     |
| 48       | **Ours**     | 118 (±18) | **235** (±37) |
|          | Ours (S)     | 192 (±13) | 226 (±97)     |
|          | Ours (E)     | 201 (±42) | 225 (±25)     |
|          | Baseline     | N/A       | 200 (±15)     |
| 96       | Baseline     | N/A       | 155 (±77)     |
| 192      | Baseline     | N/A       | 144 (±68)     |

Also check out [`./trained-results/`](./trained-results) to see the saved training results.

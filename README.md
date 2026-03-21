# Federated Compliance LoRA

Three financial institutions collaboratively fine-tune a shared LoRA adapter on
a small base LLM for compliance question-answering — without any institution
ever sharing its raw data. Each round of training, clients update a lightweight
LoRA adapter (a few million parameters at most) on their local data and send
only those delta weights to a central aggregation server. The server applies
FedAvg to merge the updates and broadcasts the improved adapter back. Raw
transaction records, internal memos, and sensitive client communications never
leave each institution's environment. After several rounds, the federated
adapter outperforms any single institution's locally-trained adapter on a
shared held-out evaluation set, demonstrating the value of privacy-preserving
collaboration.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and partition the dataset
python data/partition.py

# 3. Run the federated simulation
python run_simulation.py
```

---

## How it works

**Only LoRA delta weights travel the network — and why that matters.**

LoRA (Low-Rank Adaptation) fine-tuning works by freezing all base model weights
and injecting small trainable rank-decomposition matrices (`A` and `B`) into
select transformer layers. The full fine-tuned weight matrix is never computed
explicitly; instead, `W_pretrained + α·BA` is evaluated at inference time.

In a federated setting this architecture provides two complementary benefits:

1. **Privacy**: clients share only the low-rank matrices `A` and `B` — a tiny
   fraction of the full model size. The base model weights stay frozen and
   local, and raw training data never leaves the institution.

2. **Efficiency**: for a 500M-parameter model with `r=8`, the LoRA matrices
   account for roughly 0.5–1% of all parameters. This dramatically reduces
   per-round communication overhead compared to sharing full fine-tuned weights.

The server aggregates these delta matrices with FedAvg (a weighted average
proportional to each client's dataset size), producing a globally improved
adapter without centralizing any sensitive data.

---

## Project structure

```
federated-compliance-lora/
├── data/
│   ├── raw/                        # gitignored; populated by partition.py
│   ├── partitions/                 # gitignored; created by partition.py
│   │   ├── client_0/train.jsonl
│   │   ├── client_1/train.jsonl
│   │   ├── client_2/train.jsonl
│   │   └── test.jsonl
│   └── partition.py                # downloads + splits the dataset
├── client.py                       # Flower NumPyClient definition
├── server.py                       # FedAvg strategy + metrics aggregation
├── run_simulation.py               # single entrypoint for the simulation
├── config.py                       # all hyperparameters and paths
└── requirements.txt
```

---

## Phase roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **1** | **Current** | Scaffold — realistic structure, stub training, runnable end-to-end |
| **2** | Planned | Real LoRA fine-tuning with PEFT + Transformers inside `client.fit()` |
| **3** | Planned | Differential privacy (DP-SGD noise), secure aggregation, per-round eval curves |
| **4** | Planned | Full benchmark — federated adapter vs. per-institution local adapter on held-out set |

---

## Configuration

All tunable values live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_CLIENTS` | 3 | Number of federated institutions |
| `NUM_ROUNDS` | 5 | Federated training rounds |
| `LOCAL_EPOCHS` | 2 | Local gradient steps per round |
| `LORA_R` | 8 | LoRA rank |
| `LORA_ALPHA` | 16 | LoRA scaling factor |
| `LORA_DROPOUT` | 0.05 | Dropout applied to LoRA layers |
| `BASE_MODEL_NAME` | `Qwen/Qwen2.5-0.5B` | HuggingFace model ID |
| `MAX_SEQ_LEN` | 256 | Maximum tokenized sequence length |
| `SEED` | 42 | Global random seed |

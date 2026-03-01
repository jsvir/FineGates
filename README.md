# FineGates  
## Train Less, Infer Faster: Efficient Model Finetuning and Compression via Structured Sparsity

[![arXiv](https://img.shields.io/badge/arXiv-2602.09169-b31b1b.svg)](https://arxiv.org/abs/2602.09169)

FineGates is a structured sparsification framework for efficient adaptation and compression of Large Language Models (LLMs).

Instead of adding new parameters (e.g., LoRA), FineGates learns **stochastic binary row and column gates** that directly sparsify the base model weights.

This enables:

- ✔️ Minimal trainable parameters  
- ✔️ Real inference-time speedup  
- ✔️ Up to **40% structured pruning**  
- ✔️ Theoretical convergence guarantees  

---

# 🚀 Key Idea

FineGates replaces low-rank adaptation with trainable structured gates:


$$W \rightarrow \mathrm{Diag}(\omega_r) \, W \, \mathrm{Diag}(\omega_c)$$


- Gates are trained using stochastic relaxation
- They converge to binary values (0/1)
- Entire rows and columns are removed
- Inference cost is reduced directly (no post-pruning step required)

---

# 📊 Main Results

### GLUE Benchmark

- Matches or outperforms **Full Finetuning**
- Competitive or superior to **LoRA**
- Requires up to **10× fewer trainable parameters**
- Compresses the base model by **10–40%**

### Inference Speedup

- Up to **25% CPU inference time reduction**
- No quantization required
- No post-training pruning required

### Pretraining

- Structured pruning during training
- Up to **44% parameter removal**
- Faster convergence with only ~0.09% additional parameters

---

# 🧠 Theoretical Contributions

- FineGates satisfies the **Polyak–Łojasiewicz (PL) condition**
- Simpler and better-conditioned optimization landscape than LoRA
- Convergence guarantees under stochastic gating
- Avoids bilinear degeneracy present in low-rank parameterizations

---

# 📂 Repository Structure

| Script | Description |
|--------|------------|
| `sparse_posttrain_llama.py` | Structured sparsification without a target task |
| `finetune_glue.py` | Finetune on downstream tasks with sparsification |
| `sparse_pretrain_llama.py` | Pretraining with structured sparsity |

---

# 🔧 Usage

## 1️⃣ Sparsify a Pretrained Model (No Target Task)

```bash
python sparse_posttrain_llama.py
```


## 2️⃣ Finetune + Sparsify on Downstream Task (GLUE)

```bash
python finetune_glue.py
```

## 3️⃣ Pretrain with Structured Sparsity

```bash
python sparse_pretrain_llama.py
```


---
# ⚙️ Comparison with LoRA and MaskLLM


| Method      | Adds Trainable Parameters | Reduces Inference Cost | Structured Pruning | Pruning During Training | Post-hoc Search |
|-------------|---------------------------|------------------------|--------------------|--------------------------|-----------------|
| LoRA        | Yes (Low-rank matrices)   | ❌ No                  | ❌ No              | ❌ No                    | ❌ No           |
| MaskLLM     | Yes (Mask per weight)     | ✔️ Yes (Semi-structured) | ❌ Semi-structured | ✔️ Yes                  | ❌ No           |
| FineGates   | Minimal (Gate vectors)    | ✔️ Yes                 | ✔️ Yes             | ✔️ Yes                   | ❌ No           |



- **LoRA** → Parameter-efficient adaptation (no compression)  
- **MaskLLM** → Hardware-aware semi-structured pruning  
- **FineGates** → Structured sparsification *as an adaptation mechanism* during pre-training, post-training or finetuning.

FineGates unifies adaptation and compression in a single training process,
without adding heavy parameter matrices or requiring post-training pruning.


### Compared to LoRA

- **LoRA** adds low-rank matrices but keeps the base model fully dense.
- Inference cost remains unchanged.
- Optimization involves bilinear parameterization with potential degeneracy.
- No structured pruning is performed.

**FineGates**, instead:

- Learns binary row/column gates.
- Directly removes entire dimensions from weight matrices.
- Reduces both trainable and inference-time parameters.
- Provides theoretical convergence guarantees.


### Compared to MaskLLM

- **MaskLLM** learns semi-structured sparsity masks (e.g., 2:4 sparsity).
- Requires training a mask tensor aligned with weight dimensions.
- Designed primarily for GPU-efficient sparsity patterns.
- Retains dense matrix dimensions (semi-structured rather than full row/column removal).

**FineGates**, instead:

- Learns lightweight gate vectors (not full masks).
- Removes entire rows and columns (true structured pruning).
- Naturally reduces matrix dimensions.
- Beneficial for both CPU and GPU inference.
- Requires significantly fewer trainable parameters.
- Does not rely on specialized hardware sparsity patterns.


---

# 📈 Sparsity–Accuracy Tradeoff

FineGates enables structured compression with minimal performance degradation:

- Up to **20% parameter removal** with negligible accuracy drop  
- Up to **40% structured sparsity** with <4% accuracy loss  
- Up to **470M parameters removed** in LLaMA-1B with ~6% drop  

This demonstrates that task-specific subnetworks exist within pretrained models and can be efficiently identified through gating.


---

# 🧩 Implementation Details

The implementation is compatible with:

- RoBERTa (Base / Large)
- LLaMA (1B / 7B)
- Other Transformer-based architectures

---

# 📜 Citation

If you use FineGates in your research, please cite:

```bibtex
@article{svirsky2026train,
  title={Train Less, Infer Faster: Efficient Model Finetuning and Compression via Structured Sparsity},
  author={Svirsky, Jonathan and Refael, Yehonathan and Lindenbaum, Ofir},
  journal={arXiv preprint arXiv:2602.09169},
  year={2026}
}
```

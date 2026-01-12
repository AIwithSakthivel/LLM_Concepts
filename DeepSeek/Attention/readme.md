DeepSeek-V3.2: Scaling Attention to the Horizon
🚀 Theoretical Complexity & VRAM Simulation: Classic Attention vs. DSA
    This repository contains a scientific simulation comparing the Standard Transformer Attention mechanism with DeepSeek Sparse Attention (DSA) and Multi-head Latent Attention (MLA) as implemented in DeepSeek-V3.2.
📖 The Problem: 
    The Quadratic WallTraditional Transformers use Dense Attention, where every token attends to every other token. This creates two massive bottlenecks:Computation ($O(N^2)$): Doubling your context (e.g., from 32k to 64k) quadruples the math required.Memory (VRAM): The Key-Value (KV) cache and the attention score matrix grow so large that they eventually exceed the physical memory of even the most powerful H100/B200 GPUs.
💡 The Solution: 
    DeepSeek’s Two-Stage EfficiencyDeepSeek-V3.2 solves this using a two-pronged approach:MLA (Multi-head Latent Attention): Compresses the KV cache into a low-rank latent vector, reducing the memory footprint by up to 10x.DSA (DeepSeek Sparse Attention): Uses a Lightning Indexer to score token relevance. Instead of attending to all $N$ tokens, the model only performs precise attention on the Top-$k$ (2048) most relevant ones.
🛠️ Simulation Details
    This notebook models the hardware-level performance of both architectures using the following real-world specs from the DeepSeek-V3.2 technical report:Hidden Dimension ($d$): 5120Latent Dimension ($d_c$): 512Sparse Window ($k$): 2048Precision: Mixed FP8/FP16
📊 Key Insights from the Simulation
    Memory Efficiency: At a context of 128,000 tokens, a Classic Transformer requires over 30GB of VRAM just for the attention matrix and KV cache. DeepSeek-V3.2 stays under 2GB.
    Computational Ceiling: While Classic Attention's complexity explodes quadratically, DSA maintains a nearly linear scaling law, making 128k context reasoning as "cheap" as short-context generation was in previous generations.
🚀 How to Run
    Clone the repo.
    Open DeepSeek_Attention_Comparison.ipynb in Jupyter or Google Colab.
    Run all cells to generate the complexity and VRAM comparison plots.
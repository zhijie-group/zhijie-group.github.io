+++
title = "AdaMoE: Token-Adaptive Routing with Null Experts for MoE"
date = 2025-09-08
authors = ["Zihao Zeng*", "Yibo Miao*", "Hongcheng Gao", "Hao Zhang", "Zhijie Deng†"]
author = "Zihao Zeng*, Yibo Miao*, Hongcheng Gao, Hao Zhang, Zhijie Deng†"
ShowReadingTime = true
draft = false
[cover]
      image = "blogs/adamoe/image/vs.png"
      alt = "AdaMoE"
      caption = "**Left**: vanilla top-2 MoE. **Right**: AdaMoE with null experts."
+++

{{< socialBadges paper="https://arxiv.org/pdf/2406.13233" code="https://github.com/zhijie-group/AdaMoE" >}}

<!-- Code: https://github.com/zhijie-group/SIFT

Paper: https://github.com/zhijie-group/SIFT/blob/main/paper.pdf -->

## TL; DR

{{< justify >}}
AdaMoE adds a set of **null experts** (with zero compute) to the expert pool to enable token-adaptive expert choice for MoE.

* Under the top-k routing paradigm, tokens that route to null experts effectively use fewer true experts, making the number of **true experts per token adaptive** under the same average budget.

* With a minor tweak to the load‑balancing loss (treat all nulls as one averaged bucket) and a simple annealing schedule, AdaMoE reduces FLOPs while maintaining or improving accuracy (e.g., on Mixtral‑8×7B/ARC‑C: −14.55% FLOPs with +1.69% accuracy).
{{< /justify >}}

## Why Expert Selection Should Be Adaptive at the Token Level

{{< justify >}}
The rationale for shifting from a fixed **top-k** routing mechanism to a token-adaptive one is based on a critical observation: **not all tokens are computationally equal**. The information content and processing complexity vary dramatically across different tokens within a text. A traditional MoE model, which compels every token to activate a fixed number of experts, allocates computation uniformly, irrespective of this variance. This static approach can lead to inefficient resource distribution.
{{< /justify >}}

{{< image src="image/pre.png" alt="Pre-Experiment" width="55%" title="Proportions of the number of top experts with cumulative routing probabilities exceeding 50% for tokens in the SocialIQA dataset. Each bar represents the proportion of different counts of tokens at the corresponding MoE layer in Mixtral-8x7B.">}}

<!-- {{< image src="image/fig5.png" alt="Sticker" width="48%" title="An example of a query and its Sticker.">}} -->

{{< justify >}}
We provide empirical evidence for this variance. We analyzed the routing probability distributions in Mixtral-8x7B, a model with a fixed top-2 router. Our analysis revealed two key patterns:

1. A large fraction of tokens had routing probabilities that were highly concentrated on a single expert, indicating that the activation of a second expert was often superfluous;

2. A significant portion of other tokens had their probabilities distributed more evenly across multiple experts, suggesting that they required the computational capacity of two or even more experts for effective processing.

This finding demonstrates that a static top-k strategy is suboptimal, leading to computationally excessive allocations for simple tokens and potentially insufficient allocations for complex ones.
{{< /justify >}}

## “Null Experts” for Adaptive Routing

{{< justify >}}
AdaMoE achieves token-adaptive expert selection by incorporating **null experts**, which are defined as an empty operation requiring **zero FLOPs** to process the token feature. In the context of LLMs, common operations satisfying this requirement include a constant zero mapping and an identity mapping (we take the zero mappings null expert as the default choice in the following just for simplicity).

Our mechanism operates as follows:

1. **Extends the expert set** with m null experts (besides n true experts).

2. **Slightly increases** the router’s **top‑k** (e.g., from 2 → 3/4). Now each token’s top‑k may include some nulls.

3. **Makes compute adaptive:** if top‑k includes r nulls, the token uses only k-r true experts.

4. **Balances sensibly** by **aggregating all nulls into a single bucket** in the load‑balance loss (don’t force balance between identical nulls).

5. **Normalizes over true experts only** after top‑k so the output scale matches vanilla MoE.
  {{< /justify >}}

{{< image src="image/vs.png" alt="DeepSeek" width="85%" title="Fixed top-2 vs. AdaMoE. Left: vanilla top‑2 where every token activates exactly 2 true experts. Right: AdaMoE where top‑4 is chosen from 4 true experts + 5 null experts; some tokens hit 3 true experts, others only 1.">}}

## Main Results

{{< justify >}}
On Mixtral‑8×7B fine‑tuning, AdaMoE reduces FFN FLOPs while keeping or improving accuracy on multiple benchmarks. For example, on ARC‑Challenge, FLOPs drop by ~14.55% with +1.69% accuracy, and the layer‑wise average experts/token falls from 2.0 → ~1.67.
{{< /justify >}}

{{< image src="image/res.png" alt="venn" width="80%" title="">}}

## From Paper to Production

{{< justify >}}
We are also encouraged to see that the concept of null experts is not merely theoretical, but has been implemented in state-of-the-art LLMs. The technical report for **LongCat-Flash** [1], a 560-billion-parameter model, identifies **zero-computation experts** as a key architectural innovation and cites our paper.

The report explains that this mechanism enables the model to “allocate a dynamic computation budget to important tokens based on their significance,” activating a variable range of parameters (18.6B to 31.3B) for each token depending on context. This direct industrial application underscores the practicality and scalability of the adaptive routing strategy we proposed.

In addition, LongCat-Flash introduces several optimization techniques to address communication and load-balancing challenges associated with adaptive expert selection—further demonstrating the viability of our approach in large-scale systems.
{{< /justify >}}

## Cite Our Work

If you find our work useful, please cite our paper:

```
@inproceedings{zeng-etal-2024-adamoe,
    title = "{A}da{M}o{E}: Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models",
    author = "Zeng, Zihao  and
      Miao, Yibo  and
      Gao, Hongcheng  and
      Zhang, Hao  and
      Deng, Zhijie",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.361/",
    doi = "10.18653/v1/2024.findings-emnlp.361",
    pages = "6223--6235"
}
```

## Reference

[1] Team, Meituan LongCat, et al. "LongCat-Flash Technical Report." arXiv preprint arXiv:2509.01322 (2025).
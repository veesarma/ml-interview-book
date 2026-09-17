# Unverified citations: Parts XIV (systems) and XV (interpretability and safety)

Both parts were already in good shape when the 2026-09-17 citations pass reached
them. Every non-arXiv source the brief named was already linked in the chapters:
the OPT-175B chronicles in `facebookresearch/metaseq`, the Transformer Circuits
posts (A Mathematical Framework, Toy Models of Superposition, Towards
Monosemanticity, Scaling Monosemanticity, In-context Learning and Induction Heads),
nostalgebraist's logit-lens post on LessWrong, Simon Willison's prompt-injection
series and lethal-trifecta post, Google SAIF, Meta's Purple Llama, NVIDIA's
TensorRT-LLM documentation, the vLLM and Orca posts, Daly's checkpoint-interval
paper, and ISO 26262 and ISO 21448.

What that pass changed:

* Replaced the H100 and A100 datasheet URLs in all four Part XIV chapters with the
  two URLs confirmed in search results
  (`resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306` and
  `.../a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf`). The URLs that
  were there before did not appear in any search result.
* Added the Purple Llama repository and Llama Guard model cards to the Llama Guard
  reference in `02-safety-failure-modes.md`, which previously carried only the arXiv
  link in the References section.
* Gave the A100 bandwidth to datasheet precision in the roofline TL;DR
  (2.04 TB/s, 2,039 GB/s on the datasheet), keeping 2.0 as the figure the worked
  arithmetic uses.

## Hardware numbers checked against the vendor datasheets

| Claim in the book | Datasheet | Verdict |
|---|---|---|
| H100 SXM 989 TFLOP/s dense BF16 | 1,979 TFLOPS BF16 tensor core with 2:4 sparsity, so 989.5 dense | correct |
| H100 SXM 3.35 TB/s HBM3 | 3.35 TB/s | correct |
| H100 ridge 295 FLOP/byte | 989 / 3.35 = 295.2 | correct |
| A100 80 GB 312 TFLOP/s BF16 | 312 TFLOPS dense (624 with sparsity) | correct |
| A100 80 GB about 2.0 TB/s | 2,039 GB/s (SXM) | correct to the stated rounding; the exact ratio is 153, and the book's 156 comes from taking bandwidth as 2.0 |
| NVLink about 450 GB/s per direction | NVLink 900 GB/s aggregate | correct |
| PCIe 5.0 x16 about 64 GB/s per direction | PCIe Gen5 128 GB/s aggregate | correct |
| InfiniBand NDR 50 GB/s per port | 400 Gb/s per port | correct |

Nothing in the roofline worked examples had to be recomputed.

## Remaining

- [ ] docs/part14-systems/01-distributed-training.md | Bandwidth optimal all-reduce algorithms for clusters of workstations | Patarasuk & Yuan, JPDC 2009 | cited inside a parenthetical; publisher page not checked
- [ ] docs/part15-interpretability-safety/01-interpretability.md | Several classic attribution papers (Simonyan et al. 2013, Sundararajan et al. ICML 2017, Selvaraju et al. Grad-CAM, Zhou et al. CVPR 2016, Hewitt & Liang EMNLP 2019, Meng et al. NeurIPS 2022) | all carry arXiv links already; conference-proceedings URLs (ACL Anthology, CVF) were not added
- [ ] docs/part15-interpretability-safety/02-safety-failure-modes.md | The attack and privacy papers (BadNets, Spectral Signatures, Eykholt et al., Yeom et al., Shokri et al., Carlini et al., Lee et al., Tramer et al., Greshake et al., Zou et al.) | all carry arXiv links already; USENIX and IEEE S&P proceedings URLs were not added

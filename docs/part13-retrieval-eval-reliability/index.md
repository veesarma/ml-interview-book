# Part XIII: Retrieval, evaluation and reliability

> **Why this part matters at staff level.** Most senior candidates can train a model.
> What separates a staff signal in an ML depth or system-design round is knowing how
> the model will be *judged* (evaluation), how it *finds* what it needs at scale
> (retrieval), and what happens when the world drifts away from the training set
> (reliability). Interviewers at autonomy companies, ranking companies and frontier labs
> all weight these three topics far more heavily than most curricula do, because they are
> where shipped systems actually fail.

## What is in this part

| Chapter | You will be able to | Code you will write |
|---|---|---|
| [Retrieval & RAG](01-retrieval-and-rag.md) | Derive BM25 and the MIPS-to-NN reduction, explain IVF / HNSW / PQ with their cost models, design a hybrid RAG pipeline with reranking and citations, and evaluate it. | `mlbook.retrieval`: similarity, BM25, IVF, HNSW, PQ, RRF, a toy end-to-end RAG pipeline |
| [Evaluation](02-evaluation.md) | Pick the right metric under imbalance (PR-AUC vs ROC-AUC), derive NDCG and detection AP, implement CER/WER and FID, compute pass@k without bias, put error bars on benchmark deltas, and validate an LLM judge. | `mlbook.evaluation`: classification, ranking, detection mAP, text, generative and LLM metrics, a bootstrap evaluation harness |
| [Uncertainty & reliability](03-uncertainty-reliability.md) | Separate aleatoric from epistemic uncertainty, calibrate with temperature scaling, build conformal prediction sets with a coverage guarantee, detect OOD inputs and drift, and craft and defend against adversarial examples. | `mlbook.reliability`: calibration, entropy/MI, split conformal, OOD scores, FGSM/PGD, drift detectors |

## Prerequisites

* [Probability](../part01-math/03-probability.md) and [statistics](../part01-math/04-statistics.md):
  expectations, quantiles, hypothesis tests, the bootstrap.
* [Information theory](../part01-math/05-information-theory.md): entropy, KL, mutual information.
* [KNN & K-means](../part02-classical/04-knn-kmeans.md): the coarse quantiser in IVF *is* k-means.
* [Logistic & softmax regression](../part02-classical/02-logistic-softmax-regression.md): logits,
  softmax and cross-entropy, which calibration and OOD scores all operate on.
* [Object detection](../part04-vision/04-detection.md) for the AP/mAP chapter section;
  [CLIP & contrastive learning](../part08-multimodal/03-clip-contrastive.md) for training
  embedding models.

## If you have one day

Read in this order; the times are for a first careful pass including running the tests.

1. **Evaluation §2.1–2.4** (PR vs ROC, AUC as a rank statistic, NDCG, detection AP), 90 min.
   These are the derivations interviewers ask for verbatim.
2. **Evaluation §2.6–2.8** (pass@k, LLM-as-judge, error bars on evals), 45 min.
3. **Retrieval §2** (similarities, MIPS reduction, BM25, IVF/HNSW/PQ cost models), 90 min.
4. **Retrieval §4** (the recall / latency / memory table, filtering, long-context vs RAG), 30 min.
5. **Reliability §2.1–2.4** (heteroscedastic loss, entropy/MI, ECE, temperature scaling,
   split conformal), 75 min.
6. **Reliability §2.5–2.7** (OOD scores, shift taxonomy, FGSM/PGD), 45 min.
7. Run all three test suites and re-derive every boxed equation with the book closed, 60 min.

```bash
pytest tests/test_retrieval_* tests/test_evaluation_* tests/test_reliability_* -q
```

## How the three chapters connect

```mermaid
flowchart LR
  R[Retrieval & RAG] -->|recall@k, faithfulness| E[Evaluation]
  E -->|calibration, ECE, error bars| U[Uncertainty & reliability]
  U -->|uncertainty gating, drift alarms| P[XVII. ML system design]
  R --> P
  E --> P
```

Retrieval systems are only as good as their recall@k measurements; evaluation is only
trustworthy with confidence intervals and calibrated scores; reliability is what turns a
calibrated score into a decision (abstain, fall back, page an engineer). The
[system-design framework](../part17-ml-system-design/00-framework.md) assumes all three.

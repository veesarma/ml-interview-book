# Part XVII: ML system design

> **Why this part exists.** The ML system design round is where staff-level offers are
> won or lost. The interviewer is not checking whether you know what a two-tower model
> is; they are checking whether you can turn a vague business problem into a scoped
> system, justify every decision against its alternative, and predict how it fails.
> This part gives you the method (chapter 0) and twelve fully worked designs, each
> grounded in how a real company solved the same problem and each framed as a mock
> interview you can rehearse aloud.

## How to use this part

1. **Read [the framework](00-framework.md) first**, twice. It defines the order of
   play (clarify → metrics → data → model → serve → evaluate → iterate), the time
   budget, the "decision journal" habit, and the back-of-envelope numbers you must be
   able to produce without notes.
2. **Then read the two chapters closest to your target company** (table below) in
   full, including the *staff-level follow-ups*. Those follow-ups are the actual
   questions; the chapter body is what you need in order to answer them.
3. **Rehearse the "How to say it in the interview" scripts.** Every major decision in
   every chapter has a boxed first-person script: the decision, the rejected
   alternative, the trade-off, and the company evidence. Say them out loud until the
   evidence comes without effort ("Airbnb reported in their KDD 2019 paper that…").
4. **Skim the remaining chapters for their TL;DR cards and trade-off tables.** Many
   interviewers pivot mid-round ("now make it real-time", "now the labels are delayed
   a week"); the cross-chapter patterns are what you'll draw on.

!!! note "On citations in this part"
    Every case study and every "how to say it" script cites a specific public paper,
    engineering-blog post, product document or recorded talk by company, exact title,
    venue and year. Where the source has been verified, the citation links straight to
    it: the arXiv abstract page, the company's own engineering blog, the official
    documentation, or the recording. A small number of older academic references still
    carry title and venue only; `citations-todo/part17-ml-system-design.md` lists which
    ones and why. Nothing here attributes a design choice to a company without such a
    source, and where the text infers a reason the source does not state, it says so.

## From "the question the interviewer asks" to the chapter

| The prompt, as the interviewer phrases it | Chapter | The hidden test |
|---|---|---|
| "Design the home feed / For You page / Reels ranking." | [01 Feed & recommendation ranking](01-recommendation-feed-ranking.md) | Multi-stage funnel, two-tower retrieval with the logQ correction, multi-task ranking and how you combine heads into one score, position bias, real-time features. |
| "Design search for Airbnb listings / Amazon products / LinkedIn jobs." | [02 Search ranking](02-search-ranking.md) | Query understanding, lexical + embedding retrieval, learning to rank (LambdaMART/LambdaRank), NDCG, relevance vs engagement, cross-encoders under a latency budget. |
| "Predict click-through / conversion for ads." | [03 Ads CTR / conversion](03-ads-ctr-prediction.md) | The auction makes calibration a hard requirement; delayed conversions; feature crosses; sparse embeddings at scale; online learning; negative down-sampling and its correction. |
| "Detect fraudulent payments / fake accounts / promo abuse." | [04 Fraud & anomaly detection](04-fraud-anomaly-detection.md) | Extreme imbalance, adversarial drift, label delay and censoring (declined transactions never get labels), graph features, cost-matrix thresholds, rules + ML, human review. |
| "Design the perception stack for a self-driving car / delivery robot." | [05 AV perception](05-perception-system-av.md) | Sensors and fusion, multi-camera BEV/occupancy, tracking, a hard onboard latency budget, the data engine (triggers, auto-labelling, active learning), evaluation by range and by scenario, OTA safety. |
| "Design a detector for crosswalks (or any novel class) given petabytes of unlabeled fleet data." | [13 Novel-object 3D detection](13-novel-object-3d-detection.md) | Reframing the estimand for a static planar class, the offboard auto-labelling teacher, tri-modal weak supervision with zero labels, and why you cannot evaluate against your own generated labels. |
| "Design visual search / 'shop the look' / duplicate detection." | [06 Visual search & image retrieval](06-visual-search-image-retrieval.md) | Metric learning vs CLIP-style embeddings, billion-scale ANN and its memory, multi-modal queries, near-duplicate hashing, index refresh. |
| "Design content moderation for uploads / comments / livestreams." | [07 Content moderation](07-content-moderation.md) | Policy taxonomies, multimodal classifiers, hash matching, review-queue economics, prevalence as the true metric, adversarial evasion, appeals. |
| "Design an enterprise assistant / support bot / coding assistant." | [08 LLM assistant with RAG](08-llm-product-rag-assistant.md) | Retrieval design, context budget, guardrails and prompt injection, evaluation with validated LLM judges, routing/caching/streaming for cost and latency, RAG vs fine-tuning vs long context. |
| "Extract fields from receipts / IDs / invoices; read text in photos." | [09 OCR & document understanding](09-ocr-document-understanding.md) | Detection + recognition vs end-to-end, layout and key-value extraction, document VLMs vs pipelines, synthetic data, CER/WER/field-F1, mobile vs server, human-in-the-loop verification. |
| "Predict delivery ETA / demand next week / rider supply." | [10 Forecasting & ETA](10-forecasting-eta.md) | Time-series features, hierarchical forecasting, route/graph features, quantile losses and calibrated uncertainty, leakage-free backtesting, marketplace feedback. |
| "Decide whom to notify and prove it worked." | [11 Notifications, uplift & experimentation](11-notifications-uplift-experimentation.md) | Uplift (treatment effect) not propensity, budgeted policies, interference and switchbacks, CUPED, sequential testing, long-term holdouts. |
| "Design the ML platform / feature store / monitoring." | [12 ML platform, feature store & monitoring](12-ml-platform-feature-store-monitoring.md) | Point-in-time correctness, online/offline consistency, registry and CI/CD for models, shadow/canary, drift, data validation, GPU scheduling, LLMOps additions. |

## Which chapters matter for which company

The company deep dives in [Part XVIII](../part18-company-deep-dives/index.md) give the
business context; this table says which designs each company is known to ask for.
"Primary" means you should be able to run the whole chapter from memory; "secondary"
means the TL;DR and follow-ups.

| Company (Part XVIII page) | Primary chapters | Secondary chapters |
|---|---|---|
| [Meta](../part18-company-deep-dives/meta.md) | 01 Feed, 03 Ads, 07 Moderation | 11 Experimentation, 12 Platform, 08 LLM assistant |
| [Google & YouTube](../part18-company-deep-dives/google-youtube.md) | 02 Search, 01 Feed, 03 Ads | 06 Visual search, 09 OCR, 08 LLM assistant |
| [TikTok / ByteDance](../part18-company-deep-dives/tiktok-bytedance.md) | 01 Feed, 07 Moderation | 03 Ads, 12 Platform |
| [Netflix & Spotify](../part18-company-deep-dives/netflix-spotify.md) | 01 Feed, 11 Experimentation | 02 Search, 12 Platform |
| [Pinterest](../part18-company-deep-dives/pinterest.md) | 01 Feed, 06 Visual search, 03 Ads | 11 Notifications, 02 Search |
| [Uber & DoorDash](../part18-company-deep-dives/uber-doordash.md) | 10 Forecasting/ETA, 04 Fraud, 12 Platform | 11 Experimentation, 09 OCR (document verification), 08 LLM assistant |
| [Airbnb](../part18-company-deep-dives/airbnb.md) | 02 Search, 04 Fraud, 12 Platform | 11 Experimentation, 01 Feed |
| [Amazon](../part18-company-deep-dives/amazon.md) | 02 Search, 10 Forecasting, 09 OCR (Textract) | 03 Ads, 06 Visual search, 04 Fraud |
| [Apple](../part18-company-deep-dives/apple.md) | 09 OCR (Live Text), 06 Visual search | 05 Perception, 08 LLM assistant |
| [Tesla](../part18-company-deep-dives/tesla.md), [Waymo](../part18-company-deep-dives/waymo.md), [Zoox/Nuro/Aurora](../part18-company-deep-dives/zoox-nuro-aurora.md), [NVIDIA](../part18-company-deep-dives/nvidia.md) | 05 AV perception, 13 Novel-object 3D detection, 12 Platform (data engine) | 09 OCR (signs/text), 10 Forecasting (prediction as forecasting) |
| [Stripe & fintech](../part18-company-deep-dives/stripe-fintech.md) | 04 Fraud, 12 Platform | 09 OCR (KYC documents), 11 Experimentation |
| [OpenAI, Anthropic & DeepMind](../part18-company-deep-dives/frontier-labs.md) | 08 LLM assistant, 07 Moderation (safety classifiers) | 12 Platform (LLMOps), 02 Search |
| [Scale AI & data engines](../part18-company-deep-dives/scale-ai-data-engines.md) | 05 AV perception (data engine), 13 Novel-object 3D detection, 12 Platform | 09 OCR, 07 Moderation |

## A mock-interview practice plan (three weeks)

The goal is to make the *method* automatic and to have evidence ready for every
decision, rather than to memorise twelve designs. Each session is 60 minutes: 45 minutes
speaking to a whiteboard (or a blank document), 15 minutes reviewing against the
chapter.

| Week | Sessions | What to practise |
|---|---|---|
| 1 | Framework ×2, then Feed, Search, Ads | Run the seven phases with a timer. Say every decision as "I choose X over Y because Z; the cost is W." Do the back-of-envelope numbers out loud. |
| 2 | Fraud, Perception, Novel-object 3D detection, Visual search or OCR, LLM assistant | Practise the pivots: "make it real-time", "labels arrive a week late", "traffic spikes 5×", "the model is better offline but flat online". Use the follow-up sections as the interviewer. |
| 3 | Moderation, Forecasting, Notifications, Platform, then two full mocks with a partner | Bring your own production stories as evidence (framework §7). Have your partner interrupt with follow-ups from a *different* chapter. |

Grade yourself with the rubric in [the framework](00-framework.md#8-the-rubric): a
staff-level answer scores at least "strong" on scoping, on the decision journal, and
on evaluation, and never leaves a stage without a latency and a cost.

## Chapter list

- [00 The framework](00-framework.md)
- [01 Feed & recommendation ranking](01-recommendation-feed-ranking.md)
- [02 Search ranking](02-search-ranking.md)
- [03 Ads CTR / conversion prediction](03-ads-ctr-prediction.md)
- [04 Fraud & anomaly detection](04-fraud-anomaly-detection.md)
- [05 Autonomous-vehicle perception](05-perception-system-av.md)
- [06 Visual search & image retrieval](06-visual-search-image-retrieval.md)
- [07 Content moderation & trust](07-content-moderation.md)
- [08 LLM assistant with RAG](08-llm-product-rag-assistant.md)
- [09 OCR & document understanding](09-ocr-document-understanding.md)
- [10 Forecasting & ETA](10-forecasting-eta.md)
- [11 Notifications, uplift & experimentation](11-notifications-uplift-experimentation.md)
- [12 ML platform, feature store & monitoring](12-ml-platform-feature-store-monitoring.md)
- [13 Novel-object 3D detection from unlabeled fleet data](13-novel-object-3d-detection.md)

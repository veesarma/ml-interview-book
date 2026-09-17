# Unverified citations: Part XVII (ML system design)

Part XVII was written without a live search budget, so its author cited everything by
company, exact title, venue and year and added no URLs. A later mechanical pass linked
the arXiv identifiers. The 2026-09-17 citations pass verified and linked the half that
could not be mechanised: company engineering blog posts, research posts, product
documentation and recorded talks, plus the publisher pages for the industry papers that
have no arXiv preprint.

Every company source named in the brief is now linked in the chapter that uses it, at
the "How real companies did it" case study, inside the boxed "How to say it in the
interview" script, and in the References list. The note in `index.md` that said this
part carries no hyperlinks has been rewritten to match.

## What that pass linked

| Chapter | Sources linked |
|---|---|
| 00 framework | Rules of ML, Hidden Technical Debt, ML Test Score, Covington et al., Hello Interview's ML system design guides |
| 01 feed ranking | YouTube RecSys 2016 and RecSys 2019 (Zhao et al.), Yi et al. RecSys 2019, MMoE, PLE, Meta's News Feed ranking post, Instagram Explore post, TikTok Newsroom, Netflix artwork personalisation, Joachims WSDM 2017, Sculley et al. |
| 02 search | Google's BERT-in-Search post, Burges MSR-TR-2010-82, RankNet ICML 2005, Wang et al. WSDM 2018 |
| 03 ads | Google KDD 2013, Facebook ADKDD 2014, Chapelle KDD 2014, Ktena et al. (arXiv), LinkedIn budget pacing KDD 2014, Meta's "About Ad Auctions" help-centre page, Rendle ICDM 2010 |
| 04 fraud | Stripe Radar docs and "How we built it: Stripe Radar", PayPal's graph-database post, Uber Michelangelo and Palette, three Airbnb risk posts, Amazon Fraud Detector docs |
| 05 AV perception | Tesla AI Day 2021 and 2022 recordings, Waymo Open Dataset and Open Motion Dataset, Waymo's safety pages and safety-framework post, NVIDIA DRIVE documentation and DRIVE perception pages |
| 06 visual search | Alibaba KDD 2018, Google's multisearch announcement and the Search blog post on Lens |
| 07 moderation | Meta's Community Standards Enforcement Report, the WPIE post, the Few-Shot Learner post, the PDQ/TMK open-sourcing announcement, Microsoft PhotoDNA docs, OpenAI's GPT-4 moderation post, YouTube's transparency report |
| 08 LLM assistant | Anthropic's contextual retrieval post, two GitHub Copilot engineering posts, LinkedIn's "Musings on Building a Generative AI Product", DoorDash's Dasher support automation post |
| 09 OCR | Amazon Textract docs, Google Document AI docs and processor list, Azure Document Intelligence accuracy-and-confidence page, Apple Live Text guide and `VNRecognizeTextRequest`, ICAO Doc 9303 |
| 10 forecasting | Uber DeepETA, DoorDash long-tail ETA post plus the multi-task/probabilistic follow-up, DoorDash switchbacks, DeepMind and Google Maps GNN traffic post, Hyndman and Athanasopoulos FPP3 |
| 11 notifications | Pinterest KDD 2018 and the user-state notification post plus NEP, CUPED WSDM 2013, DoorDash switchback post, Lyft's ridesharing-marketplace experimentation post |
| 12 platform | Uber Michelangelo, Palette (Uber post plus the InfoQ talk) and the predictive-to-generative post, Airbnb Chronon, Netflix Metaflow, Meta FBLearner Flow, Google TFX, Data Validation MLSys 2019, ML Test Score, Hidden Technical Debt, Rules of ML, Feast and Tecton docs |
| 13 novel-object 3D | MUTCD, Waymo Open Dataset site (the arXiv sources were already verified by the chapter's author) |

## Claims corrected while verifying

* `11-notifications-uplift-experimentation.md` attributed "Notification Volume Control
  and Optimization System at Pinterest" (KDD 2018) to **Gupta, B. et al.** The paper is
  by **Zhao, B., Narita, K., Orten, B. and Egan, J.** Fixed in the case study and in
  the References list.
* `07-content-moderation.md` cited the WPIE work as Meta AI's "Here's how we're using AI
  to help detect misinformation". No search result confirmed that title. The post that
  does describe Whole Post Integrity Embeddings is **"The shift to generalized AI to
  better identify violating content"** (Meta AI, November 2021), and the reference now
  names and links that one.
* `07-content-moderation.md` credited the PDQ and TMK+PDQF announcement to "Meta
  Engineering". It is a Meta Newsroom post, **"Open-Sourcing Photo- and Video-Matching
  Technology to Make the Internet Safer"** (August 2019), and the title capitalisation
  has been corrected.
* `04-fraud-anomaly-detection.md` cited PayPal's graph work as "public engineering talks
  and papers … (see, for example, their work on graph neural networks for transaction
  risk in industry venues)", which named no source the reader could open. Replaced with
  the PayPal Technology Blog post "How PayPal Uses Real-time Graph Database and Graph
  Analysis to Fight Fraud", which covers the same ground, and added it to References.
* `12-ml-platform-feature-store-monitoring.md` cited "Michelangelo Palette: A Feature
  Engineering Platform at Uber" as an Uber Engineering post. It is a **recorded talk**
  hosted by InfoQ. The reference now cites Uber's own "Palette Meta Store Journey" post
  as the primary written source and the InfoQ talk alongside it.
* `00-framework.md` cited "Hello Interview, 'ML System Design' interview guides". The
  guide's actual name is **"ML System Design in a Hurry"**.

Everything else checked out: the Airbnb KDD 2019 / KDD 2020 / CIKM 2023 papers, the
Facebook embedding-based-retrieval KDD 2020 paper, Amazon's Semantic Product Search,
LinkedIn DeText and the KDD 2019 fairness paper, Etsy unified embeddings, DIN, DIEN,
SIM, TWIN, Monolith, PinSage, PinnerFormer, HSTU, DLRM, ESMM, Llama Guard and MT-Bench
all carry the titles, venues and years the chapters give them.

## Judgement calls preserved

* **Stripe's marketing figures stay out.** The chapter cites Stripe's Radar
  documentation and the "How we built it: Stripe Radar" engineering post for the 0-to-99
  risk score, the network-wide training signal and the rules-and-review layer. The
  product-page claims ("70 trillion data points", "reduces fraud 32%") and the
  engineering post's own headline numbers were not pulled into the text.
* **Company posts over aggregators.** Where a Medium mirror and a company domain both
  existed, the company domain won. Airbnb's Chronon, "Architecting a Machine Learning
  System for Risk", the two other Airbnb risk posts, Pinterest's notification posts and
  PayPal's graph post are on Medium because that is where those companies publish their
  engineering blogs.
* **`engineering.fb.com` cannot be fetched from this environment** (the egress proxy
  blocks it), so the Meta posts were confirmed from search results carrying the exact
  title, date and authors, which is what STYLE.md §4 prescribes.

## Still unverified

All remaining items are older academic references with no company claim attached. Each
chapter keeps its title-and-venue citation, which is what STYLE.md §4 requires.

- [ ] docs/part17-ml-system-design/01-recommendation-feed-ranking.md | Fairness of Exposure in Rankings | Singh & Joachims, KDD 2018 | believed arXiv:1802.07281, not confirmed
- [ ] docs/part17-ml-system-design/02-search-ranking.md | Learning to Rank with Nonsmooth Cost Functions (LambdaRank) | Burges, Ragno & Le, NeurIPS 2006 | NeurIPS proceedings page not confirmed; the LambdaRank derivation the chapter uses is in the linked MSR-TR-2010-82 overview
- [ ] docs/part17-ml-system-design/02-search-ranking.md | Cumulated Gain-Based Evaluation of IR Techniques | Järvelin & Kekäläinen, ACM TOIS 2002 | ACM DL page not confirmed
- [ ] docs/part17-ml-system-design/03-ads-ctr-prediction.md | Feature Hashing for Large Scale Multitask Learning | Weinberger et al., ICML 2009 | believed arXiv:0902.2206, not confirmed
- [ ] docs/part17-ml-system-design/03-ads-ctr-prediction.md | Smart Pacing for Effective Online Ad Campaign Optimization | Xu et al., KDD 2015 | ACM DL page not confirmed
- [ ] docs/part17-ml-system-design/03-ads-ctr-prediction.md | Internet Advertising and the Generalized Second-Price Auction | Edelman, Ostrovsky & Schwarz, American Economic Review 2007 | AEA page not confirmed
- [ ] docs/part17-ml-system-design/04-fraud-anomaly-detection.md | Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy | Dal Pozzolo et al., IEEE TNNLS 2018 | IEEE page not confirmed
- [ ] docs/part17-ml-system-design/04-fraud-anomaly-detection.md | The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets | Saito & Rehmsmeier, PLoS ONE 2015 | PLoS DOI not confirmed
- [ ] docs/part17-ml-system-design/04-fraud-anomaly-detection.md | Isolation Forest | Liu, Ting & Zhou, ICDM 2008 | IEEE/ACM page not confirmed
- [ ] docs/part17-ml-system-design/06-visual-search-image-retrieval.md | Product Quantization for Nearest Neighbor Search | Jégou, Douze & Schmid, IEEE TPAMI 2011 | Inria HAL or IEEE page not confirmed
- [ ] docs/part17-ml-system-design/08-llm-product-rag-assistant.md | Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods | Cormack, Clarke & Buettcher, SIGIR 2009 | ACM DL page not confirmed
- [ ] docs/part17-ml-system-design/09-ocr-document-understanding.md | Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks | Graves et al., ICML 2006 | ACM DL or Toronto PDF not confirmed
- [ ] docs/part17-ml-system-design/10-forecasting-eta.md | Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization | Wickramasuriya, Athanasopoulos & Hyndman, JASA 2019 | Taylor & Francis page not confirmed; the free FPP3 textbook chapter on reconciliation is linked in the same reference list
- [ ] docs/part17-ml-system-design/10-forecasting-eta.md | Regression Quantiles | Koenker & Bassett, Econometrica 1978 | JSTOR page not confirmed
- [ ] docs/part17-ml-system-design/11-notifications-uplift-experimentation.md | Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing | Kohavi, Tang & Xu, Cambridge University Press 2020 | publisher page not confirmed
- [ ] docs/part17-ml-system-design/11-notifications-uplift-experimentation.md | Real-World Uplift Modelling with Significance-Based Uplift Trees | Radcliffe & Surry, Stochastic Solutions white paper 2011 | no stable publisher URL found
- [ ] docs/part17-ml-system-design/11-notifications-uplift-experimentation.md | Discrete Sequential Boundaries for Clinical Trials | Lan & DeMets, Biometrika 1983 | Oxford Academic page not confirmed
- [ ] docs/part17-ml-system-design/11-notifications-uplift-experimentation.md | Graph Cluster Randomization: Network Exposure to Multiple Universes | Ugander et al., KDD 2013 | ACM DL page not confirmed
- [ ] docs/part17-ml-system-design/13-novel-object-3d-detection.md | AIDE: An Automatic Data Engine for Object Detection in Autonomous Driving | Liang et al., CVPR 2024 | believed arXiv:2403.17373, not confirmed

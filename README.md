# Legal-R1: Agentic Retrieval for Evidence-Based Legal Reasoning

## Introduction
**Legal-R1** is an agentic retrieval framework developed for evidence-based legal question answering. Traditional Retrieval-Augmented Generation (RAG) pipelines often rely on single-turn semantic matching, which falls short because legal systems demand verification based on authority, temporal validity, and outcome diversity. 

Legal-R1 reformulates legal question answering as an explicit problem of **evidence construction** rather than a simple database lookup. It decomposes the process into a multi-agent workflow where a post-trained **Lawyer Agent** serves as the central manager to autonomously plan multi-turn evidence seeking. It collaborates with a **Legal Assistant Agent** that conducts authority-aware and sentencing diversity-aware retrieval across heterogeneous normative legal and precedent corpora, and a **Judge Agent** that synthesizes the entire retrieval trajectory to formulate the final grounded judgment.

---

## Key Features
* **Explicit Multi-Turn Planning**: Instead of one-shot lookups, the Lawyer Agent determines whether to retrieve, which source to query, what evidence to seek, and when the collected evidence is fully sufficient to answer.
* **Authority & Diversity-Aware Reranking**: For judicial precedents, the system adapts Maximal Marginal Relevance (MMR) using an Authority Hierarchy Score and a uniquely developed Punishment Severity Index (PSI) to balance legal hierarchy and eliminate outcome redundancy.
* **Evidence-Masked SFT & Trajectory-Aware RL**: Post-trained using masked supervised fine-tuning to focus on tool-use protocols without memorizing texts, followed by Group Relative Policy Optimization (GRPO) to optimize strategic reasoning policies.
* **Automated Data Purification**: Built upon a high-quality dataset of 18,940 structured QA instances processed through rigorous task-driven applicability gating and fingerprint-based contamination filtering to ensure empirical data purity.

---

## Installation

### Prerequisites
```bash
# Insert your software dependencies or environment setup steps here
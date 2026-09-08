# RAG in Cybersecurity Operations: A Q&A System for MITTRE ATT&CK Knowledge Base

**Course:** Text Mining & Applications

**Class:** ABC

**Group**: A

**Student Participants:**
- Member A

**Lecturers:**
- Teacher A

Ho Chi Minh City – September 2026

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-background)
   - 2.1. [Knowledge Base & Benchmark Dataset](#21-knowledge-base--benchmark-dataset)
   - 2.2. [Information Retrieval & RAG](#22-information-retrieval--rag)
   - 2.3. [Question Answering Systems](#23-question-answering-systems)
3. [System Architecture](#3-system-architecture)
   - 3.1. [System Pipeline](#31-system-pipeline)
   - 3.2. [Data Preparation](#32-data-preparation)
   - 3.3. [Indexing & Retrieval](#33-indexing--retrieval)
   - 3.4. [Generation](#34-generation)
4. [Experimental Analysis](#4-experimental-analysis)
   - [4.1. Experimental Setup](#41-experimental-setup)
   - [4.2. Retrieval](#42-retrieval)
   - [4.3. Generation](#43-generation)
5. [Conclusion](#5-conclusion)
6. [References](#references)

## 1. Introduction

Security operations centers (SOCs) face escalating challenges from alert fatigue and a critical skills gap, leading to delayed incident response times [1]. Large language models (LLMs) present a transformative opportunity to automate complex workflows and augment the capabilities of human analysts. However, Large Language Models (LLMs) encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented Generation (RAG) has emerged as a promising solution by incorporating knowledge from external databases.

In this project, we build a RAG-based QA system in cybersecurity domain. helping SOC analysts in security operations centers, helping SOC analysts get quick answers to time-sensitive questions about cyberattacks.

## 2. Background

### 2.1. Knowledge Base & Benchmark Dataset

**MITRE ATT&CK** [2] is a globally-accessible knowledge base of adversary tactics and techniques, grounded in real-world observations and updated biannually. It is stored in an esoteric database format called Structured Threat Information Expression (STIX), making it ill-suited for direct use in Q&A systems. Hence, AttackQA was created in a way that makes it easier for training and inferencing with LLMs.

**AttackQA** [3]  provides 25K QA pairs with supporting rationales, focused on cybersecurity and physical attacks. About 80% of the dataset was synthetically generated using Llama3-8B, grounded in the MITRE ATT&CK knowledge base, with quality control performed via a larger LLM.

### 2.2. Information Retrieval & RAG

**Information retrieval (IR)** is the task of finding, from a large collection of documents, the subset most relevant to a query. Two paradigms for IR are relevant to this project: **sparse retrieval**, which scores documents by term overlap with the query (e.g. BM25), and **dense retrieval**, which encodes both query and document into a shared embedding space and scores relevance by vector similarity (e.g. cosine or inner-product search).

Information retrieval can be integrated into language models via a method called **retrieval-augmented generation** or **RAG**. In the basic RAG scenario, we use IR techniques to **retrieve** documents from some specified store of documents that are likely to have useful information. Then we use a large language model to **generate** an answer conditioned on these documents in addition to the original query [4].

The RAG research paradigm is continuously evolving, and it is categorized into three stages: Naive RAG, Advanced RAG, and Modular RAG [5]. The **Naive RAG** follows a traditional process that includes indexing, retrieval, and generation. **Advanced RAG** introduces specific improvements to overcome the limitations of Naive RAG. Focusing on enhancing retrieval quality, it employs pre-retrieval and post-retrieval strategies. The **modular RAG** architecture advances beyond the former two RAG paradigms, illustrating a progression and refinement within the RAG family.

### 2.3. Question Answering Systems

**Question answering (QA)**, are a type of systems in which a user can ask a question using natural language, and the system provides a concise and correct answer. A QA system is different from a search engine in that the user asks a question and the output is an accurate answer instead of a list of relevant documents [6, 7].

QA can be applied to closed or open domains. In a **closed domain**, questions are focused on a par ticular domain, and the answer is extracted from datasets built for this domain only. In contrast, in an **open domain**, the question can be on any subject, and the QA system uses a large corpus with a variety of topics [6].

There are many types of questions, but they are generally classified into two types: factoid and non-factoid, also known as complex questions. In **factoid** questions, the question has a specific answer. In contrast, **non-factoid** questions are open ended and may have a variety of possible answers [6].

In this project, we focus on two RAG paradigms, Naive RAG and Advanced RAG, and techniques, collected and compiled from various sources, from LangChain [8]. Finally, we formalize our QA system as follows:
- Domain: Closed-domain (only answer question inside the knowledge base)
- Question: Factoid (one way to answer), single-hop (answer located in one place of the knowledge base)
- Answer: Short answer

Ask a question, get a short answer with citations or an explicit "I don't know" if the answer isn't in the knowledge base.

## 3. System Architecture

### 3.1. System Pipeline
- Offline: preparing the index.
- Online: answering a question.
	- Retrieval
	- Generation

### 3.2. Data Preparation

Parses AttackQA into a corpus for indexing, and QA pairs with train/dev/test splits (stratified by `source`, not a random split) for evaluation. Outputs are three artifacts:
- `corpus.jsonl` — one row per **unique** document (deduplicated), matching `Document`
- `qa_train.jsonl` / `qa_dev.jsonl` / `qa_test.jsonl` — one row per QA pair, each pointing at the doc_id(s) it should retrieve, matching `QAExample`

### 3.3. Indexing & Retrieval

First, prepare the index for both BM25 and Dense models.

**PRE-RETRIEVAL**. This is just a minimal design that take raw user question for querying. Future plan may up to query rewriting, expansion, or decomposition.

**RETRIEVAL**. Configuring the hybrid retrieval.
- Reciprocal Rank Fusion (RRF, default)
- Weighted-sum fusion

**POST-RETRIEVAL**. Configuring the reranking.
- the retriever is first asked for a wider candidate pool (`rerank_candidate_k`, default 20–50 depending on configuration)
- and the reranker re-scores and re-sorts only that pool before the top `top_k` are passed on to generation.

### 3.4. Generation

Taken the **reranked list** of document, force the generation model to respond in strict JSON, then build a structured answer to display to user.
- Building context that suits the context window of each generation model.
- Model respond in strict JSON help tracking debug information (the full prompt, latency, raw retrieval scores).
- A QA system must respond in a way that help user/analyst get quick answers to time-sensitive questions. Therefore, generated answer must be reviewed and structured before delivering to user.

## 4. Experimental Analysis

### 4.1. Experimental Setup

AttackQA dataset
- 25,335 Q&A pairs and 17760 unique documents
- Each Q&A pairs is a `(question_id, doc_id)` which is binary retrieval, and a truth answer for generation evaluation.

Stratified 80/10/10  give
- Train (20,268 pairs)
- Dev (2,533 pairs)
- and Test (2,534 pairs).

Evaluation was made on local (CPU).

We finetuned the best hyperparameter on development split. Here give the results on the test split.

### 4.2. Retrieval

Test with metrics
- MRR
- Recall @1, @5, @10

BM25
- k1 = 1.0 (controls the impact of repeated terms, higher means term frequency matters more)
- b = 0.25  (controls document-length normalization)

Dense
- Choosing model `BAAI/bge-small-en-v1.5` for inference, because small improvement not worth the embedding time on CPU.

Hybrid
- The implementation of Hybrid retriever comes with two type of score fusion, RRF and weighted-score. Experiment with weighted-score yield better result.
- We conclude with weighted score, alpha = 0.5 (weight given to dense; 1 - alpha to bm25), rrf_k = 10.

Re-ranking
- First we retrieve 20 candidates (20 is a good number based on experiments) documents with Hybrid retriever.
- Finally, re-rank the list of candidates and return the top 5 documents (5 is a good number based on experiments).

| retriever | n | mrr@10 | recall@1 | recall@5 | recall@10 | latency_s | device | rerank_candidate_k |
|:--|-:|-:|-:|-:|-:|-:|:--|-:|
|bm25 | 2533| 0.785| 0.694| 0.901| 0.942| 0.006|CPU | 0|
|bm25+cross_encoder | 2533| 0.870| 0.818| 0.938| 0.942| 1.266|CPU | 20|
|dense | 2533| 0.863| 0.809| 0.932| 0.958| 0.228|CPU | 0|
|dense+cross_encoder | 2533| 0.882| 0.830| 0.946| 0.956| 2.470|CPU | 20|
|hybrid | 2533| 0.877| 0.825| 0.946| 0.968| 0.108|CPU | 0|
|hybrid+cross_encoder | 2533| 0.887| 0.826| 0.967| 0.978| 3.080|CPU | 20|

Discussion
- From the recall of both BM25 and Dense, recall@20 is a reasonable trade-off between recall (lowest is 0.714) and other costs (time and compute on CPU).
- Also, both BM25 and Dense had their strength in different types of questions (group by `source`), which showed that Hybrid retrieval is a good model. We may also build seperate models for different types of questions if needed, or analyze why some question types can be dealed with Dense, while other can be dealed with BM25.

### 4.3. Generation

Generation...

## 5. Conclusion

This project build a RAG-based QA system over the MITRE ATT&CK knowledge base, using AttackQA as both the source corpus and the evaluation benchmark.

We evaluated three retrieval strategies (BM25, dense, and hybrid fusion of the two), with raw question for pre-retrieval, and cross-encoder reranking for post-retrieval.

Generation...

Future work...

## References

1. \[[paper](https://onlinelibrary.wiley.com/doi/full/10.1155/jece/3383674)\] Large Language Models for Security Operations Centers: A Comprehensive Survey.
2. \[[web](https://attack.mitre.org/)\] The MITRE Corporation. MITRE ATT&CK.
3. \[[paper](https://arxiv.org/abs/2411.01073)\] AttackQA: Development and Adoption of a Dataset for Assisting Cybersecurity Operations using Fine-tuned and Open-Source LLMs.
4. \[[book](https://web.stanford.edu/~jurafsky/slp3/)\] Speech and Language Processing (3rd ed. draft).
5. \[[paper](https://arxiv.org/abs/2312.10997v5)\] Retrieval-Augmented Generation for Large Language Models: A Survey.
6. \[[paper](https://aclanthology.org/R19-2011/)\] Question Answering Systems Approaches and Challenges.
7. \[[paper](https://www.researchgate.net/publication/311425566_The_Question_Answering_Systems_A_Survey)\] The Question Answering Systems: A Survey.
8. \[[source](https://github.com/langchain-ai/rag-from-scratch)\] RAG from scratch, LangChain.
9. 
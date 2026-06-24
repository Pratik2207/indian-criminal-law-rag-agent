# Indian Criminal Law RAG Agent — Project Report

> A fully local, open-source Retrieval-Augmented Generation (RAG) system for Indian Criminal Law.

**Authors:** Pratik Ramdas Sonawane · Sarthak Sunil Khairnar

---

## Abstract

Indian criminal law spans multiple sources such as the Indian Penal Code (IPC), Code of Criminal Procedure (CrPC), Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS) and a vast body of case laws, which makes comprehensive legal research extremely challenging and time consuming. Lawyers, students and researchers spend a significant share of their time searching across scattered statutes, judgements and journals. The proposed system, the **Indian Criminal Law RAG Agent**, is a fully local, open-source Retrieval-Augmented Generation (RAG) agent that combines a vector database with a locally hosted Large Language Model to answer legal questions, perform section lookup, generate case reasoning and provide contextual explanations. The system uses CrewAI for multi-agent orchestration with a Thought–Action–Observation (TAO) loop, Qdrant as the vector database, FastEmbed (`BAAI/bge-small-en-v1.5`) for text embeddings, semantic chunking using Chonkie, and Ollama running LLaMA 3.2 for inference. A Streamlit-based web interface allows users to ask natural language legal questions, view retrieved sources and observe agent reasoning steps. Experimental observations indicate a research-time reduction of approximately 60–70%, average response time of 3–6 seconds and retrieval accuracy of 85–90% on representative legal queries. The system therefore demonstrates that an entirely local, privacy-preserving legal assistant for Indian criminal law is feasible using open-source components.

**Keywords:** Retrieval-Augmented Generation, Indian Criminal Law, Vector Database, Qdrant, CrewAI, LLaMA 3.2, Legal Research, Multi-Agent Systems.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Survey](#2-literature-survey)
3. [Software Requirements Specification](#3-software-requirements-specification)
4. [System Design](#4-system-design)
5. [Technical Specifications](#5-technical-specifications)
6. [Project Estimate, Schedule and Team Structure](#6-project-estimate-schedule-and-team-structure)
7. [Software Implementation](#7-software-implementation)
8. [Software Testing](#8-software-testing)
9. [Results and Discussion](#9-results-and-discussion)
10. [Deployment and Maintenance](#10-deployment-and-maintenance)
11. [Conclusion and Future Scope](#11-conclusion-and-future-scope)
12. [References](#references)
13. [Glossary](#glossary)

**List of Figures**

| Figure | Title |
|--------|-------|
| 3.1 | Data Flow Diagram |
| 3.2 | Class Diagram |
| 3.3 | Entity Relationship Diagram |
| 4.1 | Layered System Architecture |
| 4.2 | Use Case Diagram |
| 4.3 | Sequence Diagram |
| 9.1 | Streamlit web interface — query input |
| 9.2 | Generated grounded answer with citations |

---

## 1. Introduction

### 1.1 Overview

The legal profession is heavily dependent on accurate and timely access to statutes, codes and case laws. In India, criminal law is governed primarily by the Indian Penal Code (IPC) of 1860, the Code of Criminal Procedure (CrPC) of 1973, the Indian Evidence Act, and the recently enacted Bharatiya Nyaya Sanhita (BNS) and Bharatiya Nagarik Suraksha Sanhita (BNSS). The volume of legal text combined with continuously evolving precedents makes manual research increasingly difficult.

This project presents the Indian Criminal Law RAG Agent, a local, open-source intelligent assistant that uses Retrieval-Augmented Generation to answer legal queries grounded in actual statutes. Unlike traditional keyword search, the system understands the semantic meaning of a query, retrieves the most relevant sections from a vector database and uses a Large Language Model to generate a coherent, context-aware explanation.

### 1.2 Brief Description

Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a generative language model. Documents are first converted into high-dimensional vector embeddings and indexed in a vector database. When a user submits a query, the query is also converted into a vector and the database returns the most semantically similar chunks. These retrieved chunks are then supplied to the language model as context, which significantly reduces hallucination and grounds the answer in real source material.

In this system, the source material consists of digital copies of the IPC, CrPC, BNS and BNSS in PDF form. These documents are processed by an ingestion pipeline that performs text extraction, semantic chunking, embedding generation and finally storage in a local Qdrant collection. The CrewAI framework then orchestrates three specialized agents — Retrieval, Research and Legal Reasoning — following a Thought–Action–Observation loop to answer user queries.

### 1.3 Problem Definition

To design and develop a fully local, privacy-preserving Retrieval-Augmented Generation system for Indian Criminal Law that:

- Allows natural language querying over IPC, CrPC, BNS, BNSS and related case laws.
- Returns answers grounded in actual statutory text along with source attribution.
- Reduces the manual effort of legal research while maintaining factual accuracy.
- Operates entirely on a local machine without sending sensitive data to external services.

The fundamental challenge addressed is the gap between natural human queries and the formal, complex language of legal statutes, combined with the requirement of producing grounded, citation-aware answers without relying on cloud-based LLM APIs.

### 1.4 Objectives

1. To accelerate legal research by reducing query resolution time by approximately 60–70% compared to manual lookup.
2. To create a unified semantic access point covering IPC, CrPC, BNS, BNSS and related judgements.
3. To provide AI-assisted drafting of legal arguments based on retrieved precedents.
4. To enhance the precision of legal research through context-aware retrieval and multi-agent reasoning.
5. To build a 100% local-first solution using open-source components, ensuring privacy of user queries.
6. To provide a clean, easy-to-use web interface for end users including students, lawyers and researchers.

### 1.5 Organization of Report

This report is organized as follows. Chapter 2 presents a literature survey of existing work in legal text retrieval and RAG systems. Chapter 3 describes the Software Requirements Specification including functional and non-functional requirements. Chapter 4 details the System Design and architecture. Chapter 5 discusses the technical specifications and tools used. Chapter 6 presents the project estimation and team structure. Chapter 7 describes the software implementation. Chapter 8 covers software testing. Chapter 9 reports the results obtained. Chapter 10 details deployment and maintenance, and Chapter 11 concludes the report and outlines the future scope. References and a glossary follow at the end.

---

## 2. Literature Survey

The intersection of Artificial Intelligence and law is an active area of research. The work most relevant to this project falls into three groups: classical Information Retrieval for legal text, neural embedding-based semantic search, and Retrieval-Augmented Generation with Large Language Models.

**Classical Legal Information Retrieval.** Early systems such as LexisNexis and Westlaw rely heavily on Boolean keyword search and manually curated taxonomies. While effective for experienced legal researchers, these systems demand precise terminology and do not capture semantic intent.

**Embedding-based Semantic Retrieval.** With the advent of transformer-based encoders such as BERT, Sentence-BERT and the BGE family of models, representing legal text as dense vectors has become standard. Reimers and Gurevych (2019) introduced Sentence-BERT, enabling efficient semantic similarity search. Domain-specific variants such as Legal-BERT have demonstrated improved performance on legal benchmarks.

**Retrieval-Augmented Generation.** Lewis et al. (2020) introduced the original RAG architecture, combining a retriever with a sequence-to-sequence generator. Subsequent work on FAISS (Johnson et al., 2017), Qdrant and Chroma has provided practical, scalable vector stores. The advantage of RAG over fine-tuned LLMs is that the knowledge can be updated simply by re-indexing documents, and answers can be traced back to source chunks, which is critical for legal applications.

**Multi-Agent Frameworks.** Frameworks such as CrewAI and AutoGen have shown that decomposing a task into multiple specialized agents — each with its own role, goal and tools — improves both reasoning quality and explainability. The Thought–Action–Observation (TAO) loop enables iterative refinement of answers.

**Comparative summary.** Table 2.1 summarizes representative work and the limitations this system attempts to address.

**Table 2.1 — Summary of representative literature**

| Approach | Advantages | Disadvantages |
|----------|------------|---------------|
| Boolean keyword search (LexisNexis, Westlaw) | Mature, fast, deterministic | No semantic understanding; terminology-sensitive |
| Sentence-BERT / BGE embeddings | Captures semantic similarity | Requires good chunking; cannot generate explanations |
| Cloud LLMs (GPT-4, Gemini) | High fluency and reasoning | Privacy concerns; cost; hallucinations on legal facts |
| Original RAG (Lewis et al.) | Grounded answers with citation | Complex training; needs aligned data |
| **Proposed System** | Local, free, agentic, grounded in IPC/CrPC/BNS/BNSS | Local LLM quality bounded by hardware |

---

## 3. Software Requirements Specification

### 3.1 Introduction

**3.1.1 Purpose.** The purpose of this Software Requirements Specification (SRS) is to provide a detailed description of the functionalities, constraints, and design considerations for the Indian Criminal Law RAG Agent. It is intended to be used by developers, evaluators and end users.

**3.1.2 Project Scope.** The system is scoped to the four core Indian criminal statutes (IPC, CrPC, BNS, BNSS) along with optional web search capability through the Serper API. It is intended to run on a single local machine with adequate RAM and CPU/GPU to host LLaMA 3.2 via Ollama.

**3.1.3 Design and Implementation Constraints.**

- The system must operate locally without sending user queries to any cloud LLM provider.
- The vector database must persist on the local file system.
- All third-party libraries must be open source.
- Python 3.11 or later is required.

**3.1.4 Assumptions and Dependencies.**

- Ollama is installed and the LLaMA 3.2 model has been pulled locally.
- Source PDFs are correctly placed in the `knowledge/` directory.
- The host machine has at least 8 GB RAM (16 GB recommended).

### 3.2 System Features (Functional Requirements)

- **Document Ingestion.** The system shall extract text from PDF statutes, perform semantic chunking, generate embeddings using the BGE small model and persist the vectors in Qdrant with the source filename and chunk text in the payload.
- **Semantic Query.** The user shall be able to enter a natural language legal query through the web interface and receive the top-k most relevant chunks from the vector database.
- **Multi-Agent Reasoning.** The system shall route the query through a sequential CrewAI pipeline of three agents: a Retrieval Agent, a Research Agent and a Legal Reasoning Agent.
- **Optional Web Search.** If a Serper API key is supplied, the Research Agent shall be able to perform external web searches to supplement the local knowledge base.
- **Source Display.** The retrieved sources shall be visible to the user in a debug panel, including the source PDF and the matched text chunk.

### 3.3 External Interface Requirements

**3.3.1 User Interfaces.** A Streamlit-based web UI provides a text input box for the legal question, a button to trigger the agent pipeline, a panel to display the answer, and a sidebar for configuration (debug mode, optional Serper API key).

**3.3.2 Hardware Interfaces.** Standard PC hardware with ≥ 8 GB RAM, multi-core CPU, optional GPU for accelerated inference.

**3.3.3 Software Interfaces.**

- Ollama HTTP API at `http://localhost:11434` for LLM inference.
- Qdrant client library for vector database operations.
- Streamlit for UI rendering.
- Optional Serper REST API for web search.

**3.3.4 Communication Interfaces.** The application communicates with Ollama and Qdrant locally over loopback. External web search uses HTTPS to the Serper endpoint when enabled.

### 3.4 Nonfunctional Requirements

- **Performance.** Average end-to-end response time of 3–6 seconds for typical queries; ingestion of all four statutes within approximately 5–10 minutes on a standard laptop.
- **Safety.** The system shall not provide definitive legal advice and shall always display the disclaimer that responses must be verified by a qualified legal professional.
- **Security.** All sensitive computation and data remain on the local machine. The Serper API key, if provided, is stored only in environment variables.
- **Quality attributes.** Reliability (gracefully handles missing models or an unreachable Ollama instance); maintainability (modular code with separate files for ingestion, tools, agents and UI); usability (single-page UI with minimal cognitive load).

### 3.5 Other Requirements

- **Database.** The Qdrant collection `indian_law` uses 384-dimensional vectors with cosine distance metric.
- **Internationalization.** The first version supports English. Hindi and Marathi support is planned for future releases.
- **Legal.** The included statute PDFs are public-domain Government of India publications.
- **Reuse.** Components such as the ingestion pipeline and the document search tool are designed to be reused for other domain-specific RAG projects.

### 3.6 Analysis Model

**3.6.1 Data Flow Diagram.** The high-level data flow is: User → Streamlit UI → CrewAI Orchestrator → {Retrieval Agent → Qdrant DB, Research Agent → Serper, Reasoning Agent → Ollama LLaMA 3.2} → Final Answer → User.

![Data Flow Diagram](docs/figures/data-flow-diagram.png)

*Figure 3.1 — Data Flow Diagram*

**3.6.2 Class Diagram.** The principal classes are `LegalDocumentIngestor` (PDF → Qdrant), `document_search_tool` (CrewAI tool wrapping Qdrant search), `IndianLawRagCrew` (agents, tasks and crew composition) and the Streamlit App (UI controller).

![Class Diagram](docs/figures/class-diagram.png)

*Figure 3.2 — Class Diagram*

**3.6.3 Entity Relationship Diagram.** The Qdrant collection stores points where each point has: `id` (UUID), `vector` (384-d float), and `payload` containing `source` (PDF filename) and `text` (chunk content).

![Entity Relationship Diagram](docs/figures/entity-relationship-diagram.png)

*Figure 3.3 — Entity Relationship Diagram*

### 3.7 System Implementation Plan

The implementation follows a sequential plan: (1) Document collection, (2) Ingestion pipeline, (3) Tool creation, (4) Agent and task configuration, (5) Streamlit UI, (6) Testing, (7) Deployment.

---

## 4. System Design

### 4.1 System Architecture

The architecture is organized in four logical tiers: a Presentation Tier (Streamlit), an Orchestration Tier (CrewAI), a Reasoning Tier (Ollama LLaMA 3.2) and a Data Tier (Qdrant + the original PDF corpus). User queries originate at the Presentation Tier, travel through the Orchestration Tier where multiple agents collaborate, consult the Data Tier for grounding, and finally use the Reasoning Tier to compose a final answer.

![Layered System Architecture](docs/figures/system-architecture.png)

*Figure 4.1 — Layered System Architecture*

| Tier | Components |
|------|-----------|
| Presentation | Streamlit Web UI (`app.py`) |
| Orchestration | CrewAI: Retrieval Agent, Research Agent, Legal Reasoning Agent |
| Data | Qdrant Vector Database, Knowledge PDFs (IPC, CrPC, BNS, BNSS) |
| Reasoning | Ollama running LLaMA 3.2 (local LLM inference) |

### 4.2 UML Diagrams

**4.2.1 Use Case Diagram.** The primary actors are the End User and the System Administrator. The End User interacts with use cases such as Ask a Legal Question, View Sources and Toggle Debug Mode. The Administrator runs Ingest Documents and Update Knowledge Base.

![Use Case Diagram](docs/figures/use-case-diagram.png)

*Figure 4.2 — Use Case Diagram*

**4.2.2 Sequence Diagram.** A user query flows as follows: User → UI (query) → Crew (delegates) → Qdrant (search) → Crew (chunks) → Ollama (prompt + context) → Crew (answer) → UI (render) → User.

![Sequence Diagram](docs/figures/sequence-diagram.png)

*Figure 4.3 — Sequence Diagram*

---

## 5. Technical Specifications

### 5.1 Technology Details Used in the Project

- **Programming Language:** Python 3.11.
- **Agent Orchestration:** CrewAI and crewai-tools.
- **LLM Inference:** Ollama running LLaMA 3.2.
- **Vector Database:** Qdrant (local, persistent).
- **Embedding Model:** `BAAI/bge-small-en-v1.5` via FastEmbed (384-d vectors, cosine distance).
- **PDF Extraction:** markitdown.
- **Semantic Chunking:** chonkie with `minishlab/potion-base-8M`, threshold 0.5, chunk size 512.
- **Web Interface:** Streamlit.
- **Optional Web Search:** Serper API via SerperDevTool.
- **Configuration:** python-dotenv and YAML files.

### 5.2 References to Technology

- Ollama: <https://ollama.com>
- CrewAI: <https://docs.crewai.com>
- Qdrant: <https://qdrant.tech>
- FastEmbed: <https://github.com/qdrant/fastembed>
- Streamlit: <https://streamlit.io>
- Chonkie: <https://github.com/bhavnicksm/chonkie>

---

## 6. Project Estimate, Schedule and Team Structure

### 6.1 Project Schedule

**Table 6.1 — Project Schedule**

| Phase | Activity | Expected | Actual |
|-------|----------|----------|--------|
| 1 | Problem identification & literature survey | 2 weeks | 2 weeks |
| 2 | Requirement analysis & tool selection | 2 weeks | 3 weeks |
| 3 | Architecture design | 2 weeks | 2 weeks |
| 4 | Document collection & preprocessing | 1 week | 1 week |
| 5 | Ingestion pipeline implementation | 2 weeks | 2 weeks |
| 6 | Agent and tool development | 3 weeks | 3 weeks |
| 7 | Streamlit UI | 1 week | 1 week |
| 8 | Testing and tuning | 2 weeks | 2 weeks |
| 9 | Final report and demo | 1 week | 1 week |

### 6.2 Team Structure

**Table 6.2 — Team Structure**

| Role | Member | Primary Responsibilities |
|------|--------|--------------------------|
| Backend / RAG | Pratik Ramdas Sonawane | Backend, RAG pipeline, ingestion |
| Frontend / Agents | Sarthak Sunil Khairnar | Frontend, agents, testing |

---

## 7. Software Implementation

### 7.1 Introduction

The implementation is organized into four Python files: `ingestion.py` for one-time ingestion, `tools.py` for the CrewAI search tool, `crew.py` for agent and task definitions, and `app.py` for the Streamlit web UI. Configurations are externalized in `config/agents.yaml` and `config/tasks.yaml`.

### 7.2 Database (Data Dictionary)

**Table 7.1 — Qdrant Point Schema**

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID string | Unique identifier of the chunk |
| `vector` | float[384] | BGE small embedding of the chunk |
| `payload.source` | string | PDF filename (e.g., `IPC_1860.pdf`) |
| `payload.text` | string | Original text content of the chunk |

### 7.3 Important Modules, Mathematical Model and Algorithm

**7.3.1 Embedding Model.** For a text chunk *t*, the embedding *e_t* ∈ ℝ³⁸⁴ is produced by the BGE-small encoder. For a query *q*, *e_q* = BGE(*q*). Retrieval is performed using cosine similarity:

```
sim(e_q, e_t) = (e_q · e_t) / (||e_q|| * ||e_t||)
```

The top *k* = 5 chunks are returned.

**7.3.2 Ingestion Algorithm.**

**Table 7.2 — Pseudocode of the Legal Document Ingestion Algorithm**

```
Input:  directory of legal PDFs, Qdrant client, embed model, chunker.
Output: populated Qdrant collection `indian_law`.

1. create collection indian_law (size=384, distance=COSINE) if not exists.
2. for each pdf in the knowledge directory:
3.     raw    <- MarkItDown.convert(pdf)
4.     chunks <- SemanticChunker.chunk(raw)
5.     for each batch of 100 chunks B:
6.         E <- BGE.embed(B)
7.         for each (t, v) in (B, E):
8.             build Point(id=uuid, vector=v, payload={source, text=t})
9.         upsert points into Qdrant.
10. print "Ingestion complete."
```

**7.3.3 Query-Time Algorithm.**

1. User enters a query *q* in the Streamlit UI.
2. CrewAI launches the sequential pipeline: Retrieval → Research → Reasoning.
3. Retrieval Agent invokes `document_search_tool(q)` which returns the top-5 chunks.
4. Research Agent (optionally) augments with a Serper web search.
5. Reasoning Agent constructs a prompt with the query and retrieved context and calls Ollama LLaMA 3.2.
6. The final answer is rendered in the UI; sources are shown in the debug panel.

### 7.4 Business Logic and Software Architecture

The business logic enforces the following invariants: (a) every answer must be grounded in retrieved chunks; (b) the source of each chunk is always preserved through the payload; (c) no user query is sent to a remote LLM; and (d) the user always has the option to inspect retrieved sources.

### 7.5 Advantages and Disadvantages

**Advantages**

- 100% local execution preserves user privacy.
- Open-source stack with zero recurring cost.
- Source-grounded answers reduce hallucination.
- Modular agent architecture is easily extensible.

**Disadvantages**

- Quality of generated text is bounded by LLaMA 3.2 capability.
- Initial ingestion requires a few minutes.
- English-only in the current version.

### 7.6 Applications

- Quick reference tool for law students preparing for examinations.
- Research aid for practising lawyers.
- Legal helpdesk for paralegals and journalists.
- Educational demonstration of RAG in a regulated domain.

---

## 8. Software Testing

### 8.1 Introduction

Testing was performed at three levels: unit, integration and acceptance. The aim was to verify both the correctness of individual components and the end-to-end behaviour of the agent pipeline.

### 8.2 Test Cases

**Table 8.1 — Representative Test Cases**

| ID | Test Description | Expected | Result |
|----|------------------|----------|--------|
| TC01 | Ingest `IPC_1860.pdf` and verify count | Points > 0 in collection | Pass |
| TC02 | Query: "punishment for theft" | Retrieval includes IPC sec. 378/379 | Pass |
| TC03 | Query: "what is BNS" | `BNS_2023` chunk in top-5 | Pass |
| TC04 | Query in Hindi | Returns English answer (limitation) | Pass (degraded) |
| TC05 | Ollama unreachable | UI shows error message | Pass |
| TC06 | Empty query | UI shows warning | Pass |
| TC07 | End-to-end query latency | 3–6 seconds | Pass |

---

## 9. Results and Discussion

The system was evaluated on a set of 20 representative legal queries spanning the IPC, CrPC, BNS and BNSS. Key observations:

- **Retrieval accuracy:** Approximately 85–90% of queries produced at least one chunk that a human evaluator marked as "directly relevant" in the top-5 results.
- **Response time:** Mean end-to-end latency of 4.2 seconds (range 3.1–6.0 s) on a laptop with 16 GB RAM and an integrated GPU.
- **Research time saving:** Compared to manual lookup against the printed PDFs, users reported a reduction of roughly 60–70% in time-to-answer.
- **Grounding:** 92% of generated answers cited at least one retrieved source from the local corpus.

**Discussion.** The results validate the hypothesis that a local RAG pipeline using a 3-billion-parameter-class model (LLaMA 3.2) is sufficient for high-quality answers when the retriever supplies high-quality context. The most common failure mode is over-generalization when the retrieved chunks are short or fragmented, suggesting that improved chunking or hybrid keyword + semantic retrieval could further increase accuracy.

### 9.1 Application Snapshots

The following snapshots illustrate the working Streamlit-based web interface of the Indian Criminal Law RAG Agent, captured during a representative test run. Figure 9.1 shows the input interface where the user submits a natural language legal query. Figure 9.2 shows the agent's grounded response, which cites specific sections of the IPC and CrPC retrieved by the local RAG pipeline.

![Streamlit query input](docs/figures/ui-query-input.png)

*Figure 9.1 — Streamlit web interface showing the query input panel of the Indian Criminal Law RAG Agent.*

![Grounded answer with citations](docs/figures/ui-grounded-answer.png)

*Figure 9.2 — Generated grounded answer with citations to the Indian Penal Code and Code of Criminal Procedure.*

---

## 10. Deployment and Maintenance

### 10.1 Installation and Un-installation

**Installation**

1. Install Python 3.11 and Ollama; run `ollama pull llama3.2`.
2. Clone the repository, create a virtual environment and run `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and edit values.
4. Run `python ingestion.py` once.
5. Launch `streamlit run app.py` and open `http://localhost:8501`.

**Un-installation**

Delete the project directory along with the `qdrant_db/` folder, deactivate the virtual environment and optionally run `ollama rm llama3.2`.

### 10.2 User Help

The web interface includes inline placeholders and a sidebar with configuration help. The README file in the project root documents common errors such as "Ollama Connection Error", "Qdrant Errors" and "Memory Issues during Ingestion".

---

## 11. Conclusion and Future Scope

### 11.1 Conclusion

This project successfully demonstrates the design and implementation of a fully local, open-source Retrieval-Augmented Generation system for Indian Criminal Law. By combining CrewAI multi-agent orchestration, Qdrant vector storage, semantic chunking and Ollama-hosted LLaMA 3.2, the system delivers grounded, citation-aware answers to natural language legal queries while preserving user privacy. Empirical observations show meaningful reductions in research time and acceptable retrieval accuracy on representative queries.

### 11.2 Future Scope

- **Multilingual support:** Add Hindi, Marathi and other regional languages through fine-tuned multilingual encoders.
- **Voice queries:** Integrate speech-to-text for hands-free legal research.
- **Live court database:** Stream real-time judgements and cause lists into the vector store.
- **Advanced LLMs:** Upgrade to legal-specialized LLMs such as Legal-LLaMA or instruction-tuned variants.
- **Hybrid retrieval:** Combine BM25 keyword search with semantic search to improve recall on rare statutory terms.
- **Citation verification:** Add a verification step that re-checks every cited section against the source PDF before presenting the answer.

---

## References

1. Lewis, P. et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *NeurIPS*, 2020.
2. Reimers, N. and Gurevych, I., "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," *EMNLP*, 2019.
3. Johnson, J., Douze, M. and Jégou, H., "Billion-Scale Similarity Search with GPUs," *arXiv:1702.08734*, 2017.
4. Government of India, *Indian Penal Code*, 1860.
5. Government of India, *The Code of Criminal Procedure*, 1973.
6. Government of India, *Bharatiya Nyaya Sanhita*, 2023.
7. Government of India, *Bharatiya Nagarik Suraksha Sanhita*, 2023.
8. Qdrant Documentation, <https://qdrant.tech/documentation/>.
9. Ollama Documentation, <https://github.com/ollama/ollama>.
10. CrewAI Documentation, <https://docs.crewai.com>.
11. BAAI, "BGE: BAAI General Embedding," Hugging Face, 2023.
12. Streamlit Documentation, <https://docs.streamlit.io>.
13. Chalkidis, I. et al., "LEGAL-BERT: The Muppets straight out of Law School," *Findings of EMNLP*, 2020.
14. Karpukhin, V. et al., "Dense Passage Retrieval for Open-Domain Question Answering," *EMNLP*, 2020.
15. Touvron, H. et al., "LLaMA: Open and Efficient Foundation Language Models," *arXiv*, 2023.

---

## Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation, an architecture that combines a retriever with a generative model. |
| **LLM** | Large Language Model. |
| **IPC** | Indian Penal Code, 1860. |
| **CrPC** | Code of Criminal Procedure, 1973. |
| **BNS** | Bharatiya Nyaya Sanhita, 2023. |
| **BNSS** | Bharatiya Nagarik Suraksha Sanhita, 2023. |
| **TAO** | Thought–Action–Observation reasoning loop. |
| **BGE** | BAAI General Embedding, a family of text embedding models. |
| **Qdrant** | An open-source vector database used for semantic search. |
| **CrewAI** | A Python framework for orchestrating multi-agent LLM workflows. |
| **Ollama** | A runtime that hosts and serves open-weight LLMs locally. |
| **Streamlit** | A Python framework for rapid construction of data and ML web apps. |

---

> **Disclaimer:** This system is an educational and research tool. It does not constitute legal advice. All responses must be verified by a qualified legal professional before being relied upon.

# DocuMind — 5-Minute Portfolio Demonstration Runbook

## Executive Summary & Narrative

**Demonstration Premise:**
DocuMind is an enterprise-grade, local-first RAG application engineered to eliminate the two biggest risks in enterprise AI: **data privacy exposure** and **unverifiable hallucinations**.

**Key Technical Differentiators:**
1. **100% Offline & Free:** Runs locally on consumer CPU hardware under a **₹0 / $0 cloud compute budget**.
2. **Deterministic Attribution:** Enforces a short-token citation contract (`[S1]`, `[S2]`) resolved via application-level cryptographic metadata to actual physical PDF page numbers.
3. **Dual-Tier Abstention:** Rejects out-of-domain or ungrounded questions in **sub-20 milliseconds** (~16.6 ms mean, 9.08–16.60 ms measured), without wasting costly CPU generation cycles.
4. **Crash-Resilient Indexing:** Immutable FAISS snapshots with atomic pointer recovery across application restarts.

---

## Scripted 5-Minute Live Demonstration

### Scene 1: Multi-PDF Ingestion & Sub-Second Indexing (0:00 – 1:00)

**Speaker Action:**
1. Open browser to `http://localhost:8501`.
2. Direct attention to the left sidebar showing the **Document Management** panel.
3. Point out that the system has already hydrated 3 indexed documents (`BASWE-15-RAG-Concepts-Guide.pdf`, `Everything-to-know-about-RAG.pdf`, and `FINZAA.pdf`) comprising **436 vectors**.
4. In the file uploader, demonstrate uploading an additional PDF or explain how PyMuPDF extracts text, computes content SHA-256 digests, chunks in sliding character windows (1000 chars, 150 overlap), embeds via `all-MiniLM-L6-v2` (384-d), and writes to FAISS.

**Verbal Talking Point:**
> *"Notice the instant startup. Rather than re-reading or re-embedding documents on every session launch, DocuMind uses atomic disk snapshots. The entire 436-vector FAISS index and registry hydrate in under 200 milliseconds. Every chunk retains immutable provenance: document hash, physical 1-indexed page boundaries, and character offsets."*

---

### Scene 2: Cross-Document Factual Retrieval & Citation Accordions (1:00 – 2:15)

**Speaker Action:**
1. In the chat input, type:
   ```text
   What is chunk overlap and what is the typical starting percentage?
   ```
2. Watch the response generate.
3. Point to the generated answer containing `[S1]` citations:
   ```text
   Chunk overlap is usually 10 to 20 percent of chunk size [S1].
   ```
4. Click on the expandable **Sources ([S1])** accordion below the message.
5. Highlight the resolved metadata:
   - **Document:** `BASWE-15-RAG-Concepts-Guide.pdf`
   - **Pages:** Physical pages 2–3
   - **Evidence Snippet:** Direct verbatim chunk excerpt.

**Verbal Talking Point:**
> *"Notice that the small 0.5B parameter language model was not asked to output fragile filenames or arbitrary page strings. Instead, it emits a strict token `[S1]`. Our post-generation verification engine intercepts this token, validates it against retrieved context chunks using regex and metadata checks, and verifies that the claim has character-level evidence support in the source PDF."*

---

### Scene 3: Fast-Path Negative Abstention Defense (~16.6 ms) (2:15 – 3:15)

**Speaker Action:**
1. In the chat input, ask an out-of-domain distractor question:
   ```text
   How do you tune an automotive carburetor for cold weather?
   ```
2. Observe the near-instantaneous response:
   ```text
   Abstained — No Conclusive Evidence.
   
   Reason: No retrieved document passages exceeded the 0.35 similarity threshold. 0 LLM tokens generated.
   ```
3. Expand the **Telemetry & Performance** accordion to show:
   - **Retrieval Latency:** ~16 ms
   - **Generation Latency:** 0.0 ms
   - **LLM Tokens Generated:** 0

**Verbal Talking Point:**
> *"Most naive RAG implementations retrieve whatever is closest in vector space, pass irrelevant passages to an LLM, and produce confident hallucinations. DocuMind implements a dual-tier abstention defense. In tier 1, our fast-path pre-generation filter detects that no passage exceeds our cosine similarity threshold of 0.35. We abort generation in 16 milliseconds, saving the user from waiting 60 seconds on CPU for a hallucinated answer."*

---

### Scene 4: Complex Multi-Page Document Grounding (The FINZAA Test) (3:15 – 4:15)

**Speaker Action:**
1. In the chat input, ask:
   ```text
   Who are the authors of FINZAA.pdf? Cite the source page.
   ```
2. Show that DocuMind correctly identifies:
   - **Author:** Anjali Singh
   - **Citation:** `[S1]` linking to `FINZAA.pdf`, pages 2–7.

**Verbal Talking Point:**
> *"This query illustrates our hybrid metadata-aware retrieval. In technical and scientific PDFs, page 1 may contain cover art or unextracted layout headers, with author attribution appearing on page 2. DocuMind's metadata router detects document references in the user query and blends semantic similarity with metadata priors, reliably surfacing author and affiliation evidence without hallucination."*

---

### Scene 5: Persistence Across Restarts & Safe Confirmed Reset (4:15 – 5:00)

**Speaker Action:**
1. In the left sidebar, expand the **⚠️ Danger Zone (Reset Index)** expander.
2. Demonstrate the two-step safety gate:
   - The "Clear Index & Reset" button is **disabled** by default.
   - Checking "Confirm permanent deletion of indexed vectors" enables the button.
3. Explain the invariant:
   - Reset purges only `data/uploads/` and `data/index/snapshots/`.
   - The heavy offline model weights in `models/llm/` and all project source code are strictly protected and never touched.

**Verbal Talking Point:**
> *"To ensure operational safety, DocuMind enforces strict blast-radius controls. Resetting an index wipes ephemeral uploaded PDFs and vectors, but model caches and codebase files are architecturally immutable. When the app restarts, atomic snapshot pointers guarantee consistent state recovery."*

---

## Common Technical Interview Questions & Answers

### Q1: Why use a 0.5B model instead of 7B or 8B?
**Answer:**
> *"DocuMind was designed to prove that grounded, verifiable RAG can run completely offline on commodity 16 GB laptops without a dedicated GPU. A 7B parameter model in FP16 requires 14 GB of RAM solely for weights, leaving zero headroom for OS, FAISS, and inference activations. Qwen2.5-0.5B requires only ~990 MB of memory and peaks at 1.55 GB active RSS, allowing instant deployment anywhere at ₹0 cost."*

### Q2: Why use FAISS IndexFlatIP rather than HNSW or an approximate index?
**Answer:**
> *"IndexFlatIP performs exact brute-force Maximum Inner Product search. Because our embeddings are unit-normalized ($L_2 = 1.0$), inner product is mathematically identical to cosine similarity. For document collections up to 50,000 vectors, exact flat search takes 1.2 milliseconds on CPU. Approximate algorithms like HNSW introduce recall loss, higher memory overhead, and parameter tuning without measurable latency benefits at this scale."*

### Q3: How do you prevent the LLM from leaking internal prompt formatting or chunk IDs?
**Answer:**
> *"We employ a two-fold mitigation: first, our system prompt strictly specifies output delimiters and forbids repeating passage headers. Second, our GroundedGenerator passes all raw generation through a post-processing regex sanitizer (`_sanitize_answer`) that strips any accidental echoes of system headers, chunk tags, or internal metadata before UI rendering."*

### Q4: How is thread-safety handled if multiple users access the Streamlit UI?
**Answer:**
> *"Streamlit runs a multi-threaded Python process where each browser connection has an independent `session_state`. We isolate user messages and active document filters in `session_state`. Heavy model weights are loaded once via `@st.cache_resource` and shared across sessions. To prevent concurrent CPU cache thrashing during PyTorch forward passes, we serialize inference using a process-wide `_INFERENCE_LOCK`."*

# Phase 5 — Streamlit Web Application Acceptance Runbook

**Document ID:** `DOCU-RUN-05`  
**Milestone:** DocuMind v1.0.0 (WP-06)  
**Target Environment:** Localhost (`http://localhost:8501`), Windows 11, Local CPU Execution  
**Hardware Profile:** Local Intel/AMD CPU (8 threads), 16 GB+ RAM  
**Cost / Cloud Invariant:** Strictly ₹0 Cloud Cost • 100% Offline • Zero Telemetry Calls  

---

## 1. Purpose & Scope

This runbook defines the human acceptance testing protocol for the DocuMind Streamlit web application. It verifies end-to-end ingestion, document management, conversational grounded generation with verifiable citations, fast-path abstention, and safe index reset using the real, locally cached `Qwen2.5-0.5B-Instruct` model and `MiniLM-L6-v2` embeddings.

---

## 2. Prerequisites & Pre-Flight Checks

Before starting the web application, verify that your local environment is correctly configured:

### 2.1 Virtual Environment Activation
Open PowerShell in the project root directory (`D:\rag`):
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2.2 Verify Model Weights
Confirm that model weights are present in the offline cache (`models/llm`):
```powershell
python -c "from pathlib import Path; assert (Path('models/llm') / 'models--Qwen--Qwen2.5-0.5B-Instruct').exists(); print('Real model weights present.')"
```

### 2.3 Verify Port and Host Configuration
Inspect `.streamlit/config.toml` to ensure strict offline binding:
- `server.address = "localhost"`
- `server.port = 8501`
- `browser.gatherUsageStats = false`
- `server.enableXsrfProtection = true`

---

## 3. Starting the Application

Launch the Streamlit web server:
```powershell
$env:PYTHONPATH = "src"
streamlit run src/documind/app.py
```

Expected Startup Output:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: (Offline / Disabled)
```

Open a web browser and navigate to `http://localhost:8501`.

---

## 4. Acceptance Test Scenarios

### Test 1: Ingestion & Document Manager (APP-01, APP-02)

**Goal:** Verify multi-PDF ingestion, chunk indexing, progress telemetry, snapshot persistence, and duplicate detection.

**Steps:**
1. In the sidebar under **1. Ingest Documents**, click **Load Samples** (or drag and drop `research/technical_references/BASWE-15-RAG-Concepts-Guide.pdf` and `research/technical_references/Everything-to-know-about-RAG.pdf` into the file uploader and click **Index Uploads**).
2. Observe the real-time status expander tracking file saving, chunk extraction, vector embedding, and snapshot serialization.
3. Check **2. Document Registry**:
   - Total Documents: `2 Documents`
   - Total Chunks / Vectors: `195 Chunks / Vectors` (30 chunks from BASWE guide + 165 chunks from Everything guide).
   - Expand the **Active Documents List**: verify filenames, physical page counts, chunk counts, and file sizes are displayed.
4. Re-upload or re-index `BASWE-15-RAG-Concepts-Guide.pdf`:
   - Verify an informative toast/message appears: `Skipped (already indexed with identical content hash)`.
   - Verify vector count does **not** duplicate or increase beyond 195.

**Acceptance Criteria:**
- [ ] Ingestion completes without timeouts or exceptions.
- [ ] Document registry accurately lists both PDFs and displays exactly 195 indexed chunks.
- [ ] Duplicate uploads are detected via SHA-256 hash and safely skipped.

---

### Test 2: Factual Grounded Query with Citation Inspection (APP-03)

**Goal:** Verify synchronous verified answer generation, green verified badging, and accurate physical page citation resolution.

**Steps:**
1. In the chat input bar at the bottom, type:
   ```text
   What is the typical recommended range for chunk overlap percentage?
   ```
   (Alternatively, click the **What is chunk overlap?** quick-start pill).
2. Press Enter.
3. Observe the spinner: `Retrieving evidence & generating grounded answer...`.
4. Inspect the generated response:
   - **Grounding Badge:** Displays `✅ Grounding Status: Verified (All claims supported by citations)`.
   - **Answer Text:** States clearly that chunk overlap typically ranges from 10% to 20% (or similar factual phrasing grounded in the BASWE guide).
   - **Inline Citations:** Contains `[S1]` or similar bracketed citation keys.
5. Expand the **📚 Sources & Verifiable Citations** accordion:
   - Click `[S1] BASWE-15-RAG-Concepts-Guide.pdf — p. 2`.
   - Verify the 3-Layer Badges:
     - `• Syntax (Layer 1): Valid ✅`
     - `• Metadata Linkage (Layer 2): Mapped to ContextReference ✅`
     - `• Evidence Support (Layer 3): Claim terms confirmed in chunk ✅`
   - Verify the excerpt shows the exact sentence: `"A typical starting point is 10% to 20% overlap..."`.
6. Expand **⏱️ Performance Telemetry & Hardware Diagnostics**:
   - Verify retrieval latency is reported in milliseconds (~10-25 ms).
   - Verify generation latency and throughput (typically 1.2 - 2.5 tok/s on CPU) are displayed.

**Acceptance Criteria:**
- [ ] Answer is factually correct (10-20% overlap).
- [ ] Green `Verified ✅` badge is displayed.
- [ ] Citation accordion correctly identifies `BASWE-15-RAG-Concepts-Guide.pdf` at Page 2 with chunk text excerpt.
- [ ] Unsupported claims are never marked as verified.

---

### Test 3: Fast-Path Distractor Abstention (APP-03, APP-04)

**Goal:** Verify instant fast-path pre-generation abstention (< 15 ms) when zero chunks exceed the 0.35 similarity threshold.

**Steps:**
1. In the chat input, enter an unanswerable distractor question unrelated to the indexed documents:
   ```text
   What is the proper water temperature for sourdough bread?
   ```
2. Press Enter.
3. Observe that the response appears **immediately** (no multi-second LLM spinner).
4. Inspect the assistant message:
   - **Grounding Badge:** Displays `ℹ️ Grounding Status: Abstained (No Conclusive Evidence)`.
   - **Callout Banner:** States: `I cannot answer this question based on the provided documents.`
   - **Reasoning:** Mentions `Fast-path pre-generation rejection: No retrieved document passages exceeded the 0.35 similarity threshold. 0 LLM tokens generated.`
5. Expand telemetry:
   - Verify `Completion Tokens: 0`.
   - Verify total latency is < 30 ms.

**Acceptance Criteria:**
- [ ] Model does not hallucinate answers to off-topic questions.
- [ ] Fast-path rejection triggers with 0 LLM tokens generated.
- [ ] Visual abstention callout clearly explains the lack of evidence.

---

### Test 4: Safe Index Reset (APP-04, Safeguard 8)

**Goal:** Verify two-step confirmed index reset, wiping session state and disk snapshots while strictly preserving model files.

**Steps:**
1. In the sidebar, scroll down to **3. Index Management** and expand **⚠️ Danger Zone (Reset Index)**.
2. Verify that the **Clear Index & Reset** button is disabled by default.
3. Check the confirmation checkbox: `Confirm permanent deletion of indexed vectors`.
4. Verify the **Clear Index & Reset** button is now enabled.
5. Click **Clear Index & Reset**.
6. After rerun, inspect the application state:
   - Active Documents: `0 Documents`.
   - Indexed Knowledge Base: `0 Chunks / Vectors`.
   - Chat history: Completely cleared.
7. Verify file system preservation in PowerShell:
   ```powershell
   # Confirm model weights directory is intact
   Test-Path "models\llm\models--Qwen--Qwen2.5-0.5B-Instruct"  # Must return True
   # Confirm upload directory is empty
   (Get-ChildItem "data\uploads").Count  # Must return 0
   ```

**Acceptance Criteria:**
- [ ] Two-step confirmation prevents accidental deletion.
- [ ] In-memory vectors and document registry reset to 0.
- [ ] Uploaded PDFs in `data/uploads/` and snapshots in `data/index/snapshots/` are cleared.
- [ ] `models/` directory and all source code remain completely untouched.

---

## 5. Acceptance Sign-Off Matrix

| Test Case | Requirement | Expected Result | Human Tester Verification |
|:---|:---|:---|:---:|
| 1. Ingestion & Manager | APP-01, APP-02 | Ingests 2 PDFs (195 chunks), detects duplicates | [ ] PASS |
| 2. Grounded Query | APP-03 | Verified answer (10-20%), [S1] p. 2 citation, green badge | [ ] PASS |
| 3. Distractor Abstention | APP-03, APP-04 | Fast-path abstention (< 15 ms, 0 tokens) | [ ] PASS |
| 4. Safe Reset | APP-04 | 2-step confirmed wipe, resets to 0, preserves `models/` | [ ] PASS |

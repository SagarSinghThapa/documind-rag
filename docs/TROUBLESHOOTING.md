# DocuMind — Technical Troubleshooting & Diagnostics Guide

This guide compiles verified solutions for common runtime, operating system, network, and model issues encountered during the development, testing, and deployment of DocuMind.

---

## 1. Port 8501 Conflict ("Port is already in use")

### Symptom
When starting Streamlit via `streamlit run src/documind/app.py`, the console reports:
```text
Port 8501 is not available. Please specify an unused port with --server.port.
```

### Root Cause
A previously orphaned Streamlit server or background Python process is still bound to TCP port 8501 on `127.0.0.1` / `localhost`.

### Diagnosis
In a PowerShell terminal, identify the conflicting process:
```powershell
Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | Select-Object OwningProcess, State
```

Inspect the owning process name and command line:
```powershell
Get-Process -Id <PID> | Select-Object Id, ProcessName, Path
```

### Resolution
Terminate the orphaned process:
```powershell
taskkill /PID <PID> /F
```
Alternatively, launch DocuMind on an alternate port:
```powershell
streamlit run src/documind/app.py --server.port 8502
```

---

## 2. Windows Loopback Socket Restrictions in Agent Sandboxes

### Symptom
When running `pytest tests/test_ui_app.py tests/test_ui_state.py` inside an AI sandbox or restricted container, tests hang indefinitely or fail with:
```text
Failed to establish connection to 127.0.0.1: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions.
```

### Root Cause
Streamlit's `AppTest` framework starts an ephemeral in-process Tornado HTTP server that binds to a dynamic high-numbered loopback port (`127.0.0.1`). Restricted sandboxes often intercept or block Win32 loopback socket creation.

### Resolution
1. **Never run AppTest inside restricted sandboxes:**
   - Execute unit and component tests (128 tests) inside the sandbox:
     ```powershell
     pytest tests/test_citations.py tests/test_generator.py tests/test_retriever.py ...
     ```
2. **Execute AppTest in a native host terminal:**
   - Open standard Windows PowerShell or Windows Terminal on the host workstation and run:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     pytest tests/test_ui_app.py tests/test_ui_state.py -v
     ```
   - All 11 UI tests will pass in under 20 seconds.

---

## 3. Streamlit CORS and allowedHosts Warnings

### Symptom
When launching Streamlit, the browser or terminal displays:
```text
WARNING: streamlit.server.server: Disallowing request from unknown host...
```
Or browser cross-origin websockets fail to connect.

### Root Cause
Streamlit v1.38+ enforces strict host validation when `enableCORS = true`. Accessing via `localhost` vs `127.0.0.1` or a custom hostname without explicit whitelisting causes connection rejections.

### Resolution
Verify that `.streamlit/config.toml` includes both `localhost` and `127.0.0.1` in `allowedHosts`:
```toml
[server]
port = 8501
address = "localhost"
enableCORS = true
enableXsrfProtection = true
allowedHosts = ["localhost", "127.0.0.1"]

[browser]
gatherUsageStats = false
```

---

## 4. Intel OpenMP Duplicate Library Error (`libiomp5md.dll`)

### Symptom
During import of `torch` or `faiss`, Python crashes with:
```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint: This may cause crashes or silently produce incorrect results.
```

### Root Cause
Both `torch` (CPU distribution) and `faiss-cpu` (compiled with MSVC/Intel MKL) link against independent OpenMP runtimes (`libiomp5md.dll` or `vcomp140.dll`).

### Resolution
DocuMind automatically sets `KMP_DUPLICATE_LIB_OK="TRUE"` inside `src/documind/__init__.py`.
For standalone scripts or test executions outside the package entrypoint, export the variable in PowerShell before running Python:
```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

---

## 5. Scanned PDFs & Image-Only Page Handling (e.g. FINZAA Page 1)

### Symptom
When asking questions about text visibly rendered on the title page of a PDF (such as author names or dates), the retriever returns passages starting on Page 2 or abstains.

### Root Cause
Many academic papers (such as `FINZAA.pdf`) render Page 1 as a raster image or scanned graphic without underlying digital text streams. Standard PDF text extraction libraries (`fitz.get_text()`) return empty strings for scanned pages unless an Optical Character Recognition (OCR) engine (such as Tesseract) is invoked.

In `FINZAA.pdf`:
- **Physical Page 1:** Scanned graphic title page (0 digital characters).
- **Physical Page 2:** First digital text page containing paper abstract, author name ("Anjali Singh"), and affiliations.
- **Chunk 0:** Covers physical pages 2–7.

### Resolution
DocuMind's metadata router and sliding-window chunker handle this gracefully:
1. Chunks retain multi-page spans (`start_page=2`, `end_page=7`).
2. When the query references the document title, the metadata-prioritized retriever boosts chunks from the target document.
3. For pure scanned PDFs lacking any digital text streams, Milestone 2 will incorporate optional Tesseract OCR preprocessing.

---

## 6. MemoryError Prevention During Unit Tests

### Symptom
Running a large test suite triggers `MemoryError` or system slowdown during fixture setup.

### Root Cause
Instantiating real Transformer models (`SentenceTransformer` and `AutoModelForCausalLM`) in every isolated test function consumes multiple gigabytes of RAM.

### Resolution
DocuMind implements dependency injection for all testing:
- **`MockEmbedder`:** Returns fixed-dimension, deterministic L2-normalized numpy arrays (`384-d`) in `< 0.1 ms` with zero RAM overhead.
- **`MockLLM`:** Returns canned answers and deterministic `[S1]` citations without allocating neural network weights.
- Real models are isolated strictly to end-to-end evaluation scripts (`verify_real_llm.py`, `profile_hardware_envelope.py`, `verify_e2e_acceptance.py`).

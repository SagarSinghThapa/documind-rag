# DocuMind — Windows 11 Clean-Room Setup Guide

This guide provides step-by-step instructions for deploying and running DocuMind on a clean Windows 11 (x64) workstation.

DocuMind is designed for **100% offline, local-first execution** with **₹0 cloud spend**, running entirely on consumer CPU hardware (16 GB RAM recommended, tested on 10-core Intel Core i7 / AMD Ryzen).

---

## 1. System Requirements

| Specification | Minimum | Recommended | Tested System |
|:---|:---|:---|:---|
| **Operating System** | Windows 10 x64 (Build 19041+) | Windows 11 x64 (22H2+) | Windows 11 Pro 64-bit |
| **CPU** | 4 Cores (x86_64) | 8+ Cores with AVX2 support | 10 Physical / 12 Logical Cores |
| **System Memory (RAM)** | 8 GB | 16 GB | 15.65 GB Total Physical RAM |
| **Disk Storage** | 5 GB free SSD space | 10 GB free SSD space | NVMe PCIe SSD |
| **Network** | None during runtime | Offline after model provisioning | 100% air-gapped runtime |

---

## 2. Prerequisites Installation

### 2.1 Python 3.11 Installation
DocuMind requires **Python 3.11** (tested against Python 3.11.9).

1. Download the official 64-bit installer from [python.org](https://www.python.org/downloads/release/python-3119/):
   - `Windows installer (64-bit)`
2. Run the installer:
   - Check **"Add python.exe to PATH"**.
   - Select **"Customize installation"** and ensure `pip` and standard development tools are checked.
   - Click **Install Now**.
3. Verify the installation in PowerShell:
   ```powershell
   py -3.11 --version
   # Expected: Python 3.11.9
   ```

### 2.2 PowerShell Execution Policy
If PowerShell blocks virtual environment activation scripts, enable local script execution for the current user:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 3. Repository & Virtual Environment Setup

### 3.1 Clone or Open Repository
```powershell
cd D:\rag
```

### 3.2 Create and Activate Isolated Virtual Environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Confirm that the virtual environment interpreter is active:
```powershell
Get-Command python | Select-Object -ExpandProperty Source
# Expected: D:\rag\.venv\Scripts\python.exe
```

### 3.3 Upgrade Core Packaging Tools
```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 3.4 Install Dependencies
Install all pinned production and evaluation dependencies from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

---

## 4. Windows-Specific DLL & Runtime Handling

### 4.1 PyTorch & FAISS OpenMP Duplicate Library Resolution
On Windows, both PyTorch (`torch_cpu.dll`) and `faiss-cpu` bundle independent copies of the Intel OpenMP runtime (`vcomp140.dll` or `libiomp5md.dll`). In certain environments, this triggers a fatal runtime conflict:
```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

**Resolution in DocuMind:**
DocuMind programmatically configures the OpenMP runtime environment at module initialization in `src/documind/__init__.py`:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
If executing custom scripts directly in PowerShell, export this environment variable:
```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

### 4.2 Win32 Loopback Socket Boundary (Antigravity / Sandbox Note)
When running automated Streamlit tests (`tests/test_ui_app.py`, `tests/test_ui_state.py`), the Antigravity sandbox restricts Windows loopback socket connections (`127.0.0.1`).
- **Unit and regression tests (128 tests):** Execute seamlessly inside any standard sandbox or terminal.
- **Headless Streamlit AppTests (11 tests):** Must be run in a native host PowerShell terminal outside restrictive socket sandboxes:
  ```powershell
  pytest tests/test_ui_app.py tests/test_ui_state.py -v
  ```

---

## 5. Offline Model Provisioning

DocuMind runs strictly offline with zero paid APIs. Language model weights and embedding models must be cached locally before disconnected execution.

### 5.1 Embedding Model (`sentence-transformers/all-MiniLM-L6-v2`)
The 384-dimensional dense embedder (~90 MB) is automatically cached in the standard Hugging Face cache upon first initialization or can be pre-downloaded:
```powershell
python scripts/download_model.py
```

### 5.2 Language Model (`Qwen/Qwen2.5-0.5B-Instruct`)
The local CPU reasoning model is stored directly in `models/llm/`:
```powershell
python scripts/download_llm.py
```

**Directory Verification:**
Verify that `models/llm/` contains the complete Safetensors snapshot:
```powershell
Get-ChildItem -Path models\llm\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots
# Must contain: config.json, model.safetensors (~990 MB), tokenizer.json, etc.
```

When `local_files_only=True` is configured in `GenerationConfig`, DocuMind guarantees that zero outbound network traffic is generated during generation.

---

## 6. Streamlit Server Security & Configuration

The application is pre-configured with secure localhost bindings in `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "localhost"
headless = true
enableCORS = true
enableXsrfProtection = true
allowedHosts = ["localhost", "127.0.0.1"]

[browser]
gatherUsageStats = false
```

### Key Security Properties:
- `address = "localhost"` binds the server strictly to the local loopback interface, preventing exposure to the local area network (LAN).
- `allowedHosts` blocks DNS rebinding attacks.
- `enableXsrfProtection = true` protects against cross-site request forgery.
- `gatherUsageStats = false` disables all outbound telemetry.

---

## 7. Starting DocuMind

Launch the application:
```powershell
streamlit run src/documind/app.py
```

Open your browser to:
```text
http://localhost:8501
```

### Verifying Normal Operation:
1. **Sidebar:** Shows "DocuMind" title, file uploader, indexed documents count, and memory telemetry.
2. **Persistence:** Existing indexed documents (`BASWE`, `Everything RAG`, `FINZAA`) are loaded in `< 200 ms` on startup.
3. **Chat Interface:** Type a question (e.g. *"What is chunk overlap?"*) and verify response generation with expandable `[S1]` citation badges.


---

# ✅ **TROUBLESHOOTING_UBUNTU_22_04.md**


# Troubleshooting Notes – Ubuntu 22.04 LTS

These notes document **my personal experience** installing and using Chatterbox-TTS on Ubuntu 22.04.  
They are **not official instructions**, and they may not apply to your system.

They are provided only for users who encounter similar issues.

---

## 🖥 System Information

- **OS:** Ubuntu 22.04.3 LTS  
- **Python versions:** 3.10 and 3.11 available  
- **GPU:** NVIDIA (CUDA-enabled)  
- **Chatterbox version:** as released on PyPI (chatterbox-tts)  
- **Environment:** Virtual environment (`venv`) with Python 3.11  

---

## ⚠️ Issues Encountered

### 1. `pkuseg` build failure  
Chatterbox depends on `pkuseg`, which failed to build without NumPy present.

Symptoms:


Resolution:
- Install NumPy **before** installing Chatterbox.
- Ensure build tools (`build-essential`) are installed.

---

### 2. Missing `longintrepr.h` for Python 3.11  
During compilation:


Notes:
- Python 3.11 changed internal header locations.
- Some third-party dependencies still expect old header paths.

Resolution:
- Install `python3.11-dev` so headers exist.
- Ensure build tools were installed (`sudo apt install build-essential`).

---

### 3. Wheel-building failures for `s3tokenizer`
Error:

---

### 4. Additional CUDA/SDPA warnings  
Chatterbox emits warnings about attention kernels:


These are benign; no action required.

---

### 5. First-run model download failures  
Occasionally, initial Chatterbox runs failed while downloading voice or model components.

Resolution:
- Restart the script once.
- Ensure internet connectivity during first run.

---

## ✔ Summary

Installing Chatterbox-TTS on Ubuntu 22.04 required:

- Installing NumPy beforehand  
- Installing Python header files (`python3.11-dev`)  
- Installing build tools (`build-essential`)  
- Updating wheel/setuptools  
- Re-running the script to trigger model downloads  

These notes are provided in case they help other Ubuntu users, but they should not be considered official or complete guidance.


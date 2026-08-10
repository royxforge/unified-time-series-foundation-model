# Installation Guide for TSFM Backends

This guide explains how to install each of the 5 Time Series Foundation
Model packages that UniTSFM wraps.  Some packages have known build issues
on Windows or require specific build tools -- this document provides
tested workarounds.

> **Tip:** If you only need one or two models, install only those
> packages.  UniTSFM gracefully handles missing backends and reports
> which models are available.


## Prerequisites

### Universal

- Python 3.11 or later
- PyTorch 2.1+ ([pytorch.org](https://pytorch.org/get-started/locally/))
- pip 23.0+ (`python -m pip install --upgrade pip`)
- 8 GB+ disk space for model weight caches (~2-5 GB per model)

### Windows Only

Two packages (`uni2ts`, `lag-llama`) may require build tools on Windows.

**Option A -- Microsoft Visual C++ Build Tools (recommended):**

1. Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer and select "Desktop development with C++"
3. Restart your terminal

**Option B -- Windows Subsystem for Linux (WSL2):**

```bash
wsl --install -d Ubuntu
# Then run all pip commands inside WSL2
```

**Option C -- Conda / Mamba (handles binaries better than pip on Windows):**

```bash
conda create -n uniftsm python=3.11
conda activate uniftsm
# Then run the pip commands below
```


## Package Installation

| Model | Package | PyPI / Source | Status |
|-------|---------|---------------|--------|
| Chronos | `chronos-forecasting` | PyPI | ✅ Verified |
| TimesFM | `timesfm` | PyPI | ✅ Verified |
| MOIRAI | `uni2ts` | PyPI + GitHub | ⚠️ Build tools needed |
| Lag-Llama | `lag-llama` | GitHub only | ⚠️ Git + build tools |
| TTM | `granite-tsfm` | PyPI | ✅ Verified |


### 1. Chronos (Amazon)

```bash
pip install chronos-forecasting
```

**Works on:** Linux, macOS, Windows (verified)
**Model sizes:** `tiny`, `small`, `base`, `large`
**HuggingFace repo:** `amazon/chronos-t5-*`
**Approx. download:** 300 MB (tiny) -- 2.5 GB (large)

> **Note:** The PyPI package is `chronos-forecasting`, not `chronos`.
> The `chronos` package on PyPI is unrelated.


### 2. TimesFM (Google)

```bash
pip install timesfm
```

**Works on:** Linux, macOS, Windows (verified)
**Model sizes:** TimesFM 2.x ships a single open 200M checkpoint
**HuggingFace repo:** `google/timesfm-2.5-200m-pytorch`
**Approx. download:** ~700 MB
**HuggingFace token:** Some model versions require authentication.
If you see a `401` error, log in with:

```bash
huggingface-cli login
# Then paste your HuggingFace token
```

> **Note:** The `timesfm` package installs TimesFM 2.x.  UniTSFM's
> wrapper handles both the 2.x (via `from_pretrained()`) and legacy
> (direct constructor) APIs.


### 3. MOIRAI (Salesforce)

**Basic install (may fail without build tools):**

```bash
pip install uni2ts
```

**If the above fails with `Cannot import 'mesonpy'`:**

This error occurs because `uni2ts` requires `meson-python` and a C++
compiler to build from source on Windows.  Install the build
dependencies first:

```bash
pip install meson-python ninja wheel
pip install uni2ts --no-build-isolation
```

**If you still get build errors, install pre-built wheels via conda:**

```bash
conda install -c conda-forge uni2ts
```

**Alternatively, install directly from GitHub (latest):**

```bash
pip install git+https://github.com/SalesforceAIResearch/uni2ts.git
```

**Works on:** Linux (native), macOS (native), Windows (with VS Build Tools
or WSL2)
**Model sizes:** `small`, `base`, `large`
**HuggingFace repo:** `Salesforce/moirai-*`
**Approx. download:** 300 MB -- 1.5 GB

> **Note:** The HuggingFace `transformers` library also supports MOIRAI
> checkpoints.  UniTSFM's wrapper tries `uni2ts` first, then falls back
> to `transformers`.


### 4. Lag-Llama (Time Series Foundation Models)

**Install from GitHub (requires git):**

```bash
pip install git+https://github.com/time-series-foundation-models/lag-llama.git
```

**Works on:** Linux (native), macOS (native), Windows (with VS Build Tools
or WSL2).  There is no PyPI package.

**Model sizes:** `base` (only size available)
**HuggingFace repo:** `time-series-foundation-models/lag-llama`
**Approx. download:** 800 MB

**Windows troubleshooting:**

If the GitHub install fails with a compilation error:

```bash
# 1. Install git for Windows from https://git-scm.com/
# 2. Ensure git is in your PATH, then retry:
pip install git+https://github.com/time-series-foundation-models/lag-llama.git

# 3. If that still fails, use WSL2:
wsl --install -d Ubuntu
# Inside WSL2:
pip install git+https://github.com/time-series-foundation-models/lag-llama.git
```

> **Tip:** Lag-Llama uses a LLaMA-style architecture from the
> `transformers` library.  If the GitHub install is problematic, you
> can often load the model weights directly via HuggingFace
> `LlamaForCausalLM` without the `lag-llama` package itself.


### 5. TinyTimeMixers / TTM (IBM)

**Install via the `granite-tsfm` package (not `tinytimemixers`):**

```bash
pip install granite-tsfm
```

**Works on:** Linux, macOS, Windows (verified -- provides pre-built wheels)
**Model sizes:** `small`, `medium`, `large` (all resolve to the r2 release)
**HuggingFace repo:** `ibm-granite/granite-timeseries-ttm-r2`
**Approx. download:** ~50 MB

> **Note:** The package name on PyPI is `granite-tsfm`, NOT
> `tinytimemixers`.  The `tinytimemixers` name does not exist on PyPI.
> UniTSFM's `pyproject.toml` has been updated to reflect this.


## Installing All Models at Once

```bash
# Via UniTSFM optional dependency groups:
pip install uniftsm[all]

# Or manually:
pip install chronos-forecasting timesfm granite-tsfm
pip install uni2ts          # may need build tools
pip install git+https://github.com/time-series-foundation-models/lag-llama.git
```

## Quick Smoke Test

After installing, verify that UniTSFM can see the models:

```python
from uniftsm.core.registry import registry
import uniftsm.models.chronos
import uniftsm.models.timesfm
import uniftsm.models.moirai
import uniftsm.models.lag_llama
import uniftsm.models.ttm

print("Available models:", registry.list_models())
# Expected: ['chronos', 'lag_llama', 'moirai', 'timesfm', 'ttm']
```

To test a real forecast with Chronos (smallest download):

```python
import pandas as pd
import numpy as np
from uniftsm.models.chronos import ChronosForecaster

y = pd.Series(
    np.cumsum(np.random.randn(100)),
    index=pd.date_range("2020-01-01", periods=100, freq="D"),
)
model = ChronosForecaster(model_size="tiny", device="cpu")
model.fit(y)
pred = model.predict(horizon=10)
print(pred)
```

## Disk Space Requirements

| Model | Weights Download | RAM Usage |
|-------|-----------------|-----------|
| Chronos-tiny | ~300 MB | ~1 GB |
| Chronos-base | ~1.5 GB | ~4 GB |
| Chronos-large | ~2.5 GB | ~8 GB |
| TimesFM | ~2 GB | ~6 GB |
| MOIRAI | ~1.5 GB | ~4 GB |
| Lag-Llama | ~800 MB | ~3 GB |
| TTM-small | ~50 MB | ~500 MB |

**Total (all models, tiny/small variants):** ~5 GB download, ~15 GB RAM

## Common Issues

### `pip install` fails with "Microsoft Visual C++ 14.0 is required"

Install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
with the "Desktop development with C++" workload, or use WSL2.

### `uni2ts` fails with "Cannot import 'mesonpy'"

```bash
pip install meson-python ninja
pip install uni2ts --no-build-isolation
```

### Model weights fail to download / 401 Unauthorized

Some models (TimesFM) require HuggingFace authentication:

```bash
huggingface-cli login
```

### `CUDA out of memory`

Use a smaller model variant or reduce context length:

```python
model = ChronosForecaster(model_size="tiny", context_length=256)
```

### `tinytimemixers` package not found

Install `granite-tsfm` instead:

```bash
pip install granite-tsfm  # NOT tinytimemixers
```

### `lag-llama` package not found

Install directly from GitHub:

```bash
pip install git+https://github.com/time-series-foundation-models/lag-llama.git
```

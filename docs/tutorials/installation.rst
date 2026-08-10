Installation Guide
==================

Requirements
------------

* Python 3.11 or later
* PyTorch 2.1+ (CPU or CUDA)
* 8 GB+ RAM (16 GB+ recommended for ensemble mode)


Standard Installation
---------------------

Install the core library from PyPI:

.. code-block:: bash

    pip install uniftsm

This installs the core framework and all non-TSFM dependencies (numpy,
pandas, torch, scikit-learn, matplotlib, etc.).


Installing TSFM Backends
-------------------------

UniTSFM wraps multiple Time Series Foundation Models, each with their
own package.  Install only the ones you need:

.. code-block:: bash

    # Single backends
    pip install uniftsm[chronos]
    pip install uniftsm[timesfm]
    pip install uniftsm[moirai]
    pip install uniftsm[lag_llama]
    pip install uniftsm[ttm]

    # All backends
    pip install uniftsm[all]

.. note::

    TSFM packages can be large (1-5 GB each when downloaded).  The
    ``all`` extra is convenient for local development but may not be
    suitable for CI environments.


Development Installation
------------------------

Clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/royxforge/uniftsm.git
    cd uniftsm
    python -m venv venv
    source venv/bin/activate  # or venv\\Scripts\\activate on Windows
    pip install -e ".[dev]"


Verifying the Installation
--------------------------

.. code-block:: python

    import uniftsm
    print(uniftsm.__version__)  # 0.1.0

    from uniftsm.core.registry import registry
    print(registry.list_models())  # ['chronos', 'lag_llama', 'moirai', 'timesfm', 'ttm']


Troubleshooting
---------------

**ImportError: No module named 'chronos'**

Install the missing backend:

.. code-block:: bash

    pip install uniftsm[chronos]

**CUDA out of memory**

Reduce the context length or use a smaller model variant (e.g.
``model_size="tiny"`` instead of ``"base"``).

**Model weights fail to download**

Ensure you have a working internet connection and at least 2 GB of free
disk space in your HuggingFace cache directory (``~/.cache/huggingface/``).

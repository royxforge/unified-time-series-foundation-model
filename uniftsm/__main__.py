"""Module entry point — ``python -m uniftsm``.

This mirrors the ``uniftsm`` console script so the CLI is reachable
through either ``uniftsm ...`` or ``python -m uniftsm ...``.
"""

from uniftsm.cli.main import main

if __name__ == "__main__":
    main()

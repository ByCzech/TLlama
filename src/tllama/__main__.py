"""Support `python -m tllama`.

The package is the obvious thing to name when running from a checkout,
and without this it is the one spelling that does not work.
"""

from tllama.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

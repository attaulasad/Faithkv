"""Allow the documented ``python -m kvcot`` command form."""
from kvcot.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

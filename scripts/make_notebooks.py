"""Compatibility entry point: build the classroom presentation, not CLI copies."""

import bootstrap  # Resolve src/ and the project root.
from presentation_notebooks import main

if __name__ == '__main__':
    main()

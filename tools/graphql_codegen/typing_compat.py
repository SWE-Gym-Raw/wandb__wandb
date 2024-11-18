"""Definitions to ensure compatibility with all supported python versions."""

import sys

if sys.version_info >= (3, 12):
    from typing import Annotated, override
else:
    from typing_extensions import Annotated, override


Annnotated = Annotated
override = override

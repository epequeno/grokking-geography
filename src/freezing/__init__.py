"""
Freezing API for surgical component interventions.

Implementation: zero-gradient hooks (NOT requires_grad=False).
This preserves optimizer state (Adam moments) across freeze/unfreeze events.
See manager.py for details.
"""

from .manager import FreezeManager

__all__ = ["FreezeManager"]

"""Mocks for testing behavioral plumbing without claiming model results."""
from __future__ import annotations
class ScriptedProvider:
    def __init__(self,responses):
        self.responses=iter(responses)
    def __call__(self,messages):
        return {"text":next(self.responses),"model":"synthetic","provider":"synthetic"}

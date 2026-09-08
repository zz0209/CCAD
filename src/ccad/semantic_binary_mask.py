"""The existing RAVEL DBM gate on frozen SAE codes, with residual retained.

The mask is unconstrained in selected-member count; its measured binary cost
must accompany any comparison with a projected sparse native control.
"""
from ccad.ravel_controls import MultiDBM


class SemanticBinaryMask(MultiDBM):
    def forward(self,difference,decoder,control):
        return super().forward(difference,None,control)@decoder

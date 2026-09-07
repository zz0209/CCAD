"""Portable source-member operations from an absolute target-code map."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .component_correspondence import component_update


@dataclass
class ComponentOperation:
    target_members: np.ndarray
    source_members: np.ndarray
    allocation: np.ndarray
    source_decoder: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.target_members=np.asarray(self.target_members,dtype=np.int64)
        self.source_members=np.asarray(self.source_members,dtype=np.int64)
        self.allocation=np.asarray(self.allocation,dtype=np.float64)
        self.source_decoder=np.asarray(self.source_decoder,dtype=np.float64)
        for members in [self.source_members,self.target_members]:
            if members.ndim!=1 or not len(members) or np.any(members<0) or len(set(members.tolist()))!=len(members):
                raise ValueError('Distinct nonnegative native member IDs required')
        if self.allocation.shape!=(len(self.target_members),len(self.source_members)):
            raise ValueError('Allocation shape differs from member identity')
        if self.source_decoder.ndim!=2 or len(self.source_decoder)!=len(self.source_members):
            raise ValueError('Source decoder shape differs from member identity')
        if not np.isfinite(self.allocation).all() or not np.isfinite(self.source_decoder).all():
            raise ValueError('Finite operation arrays required')
        if self.metadata.get('state_projection','identity') not in ['identity','clip','cone_half']:
            raise ValueError('Unknown absolute-state projection')

    def apply(self,selected_codes,scales=None,consumer='removal',donor_codes=None,dose=1.):
        z=np.asarray(selected_codes,dtype=np.float64)
        if z.ndim<1 or z.shape[-1]!=len(self.target_members) or not np.isfinite(z).all():
            raise ValueError('Finite target codes in the recorded member order required')
        if not np.isfinite(dose):raise ValueError('Finite dose required')
        projection=self.metadata.get('state_projection','identity')
        if projection!='identity':
            from .source_state_projection import project_source_state
            allocation=project_source_state(z@self.allocation,self.source_decoder,projection)[0]
            if consumer=='contrast':
                donor=np.asarray(donor_codes,dtype=np.float64)
                if donor.shape!=z.shape or not np.isfinite(donor).all():raise ValueError('Matching donor codes required for contrast')
                allocation=project_source_state(donor@self.allocation,self.source_decoder,projection)[0]-allocation
            elif consumer=='removal':allocation=-allocation
            elif consumer!='contribution':raise ValueError('Unknown consumer')
            scales=np.ones(len(self.source_members)) if scales is None else scales
            return dose*component_update(allocation,self.source_decoder,scales)
        if consumer=='contrast':
            donor=np.asarray(donor_codes,dtype=np.float64)
            if donor.shape!=z.shape or not np.isfinite(donor).all():raise ValueError('Matching donor codes required for contrast')
            z=donor-z
        elif consumer=='removal':z=-z
        elif consumer!='contribution':raise ValueError('Unknown consumer')
        allocation=z@self.allocation
        scales=np.ones(len(self.source_members)) if scales is None else scales
        return dose*component_update(allocation,self.source_decoder,scales)

    def save(self,path):
        with Path(path).open('xb') as f:
            np.savez_compressed(f,target_members=self.target_members,source_members=self.source_members,
                allocation=self.allocation,source_decoder=self.source_decoder,metadata_json=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            return cls(*(data[key].copy() for key in ['target_members','source_members','allocation','source_decoder']),json.loads(str(data['metadata_json'])))

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BackboneSpec:
    train_script: str
    eval_script: str
    hp_file: str
    hp_key: str
    result_tag: str
BACKBONES: dict[str, BackboneSpec] = {'ionet_a': BackboneSpec(train_script='train_ionet_a.py', eval_script='evaluate_ionet_a.py', hp_file='hyperparams_ionet_a.json', hp_key='ionet_a', result_tag='ionet_a')}
MAIN_BACKBONE = 'ionet_a'

def resolve_backbone(name: str) -> BackboneSpec:
    if name not in BACKBONES:
        raise ValueError(f'Unknown backbone {name!r}; choose from {list(BACKBONES)}')
    return BACKBONES[name]

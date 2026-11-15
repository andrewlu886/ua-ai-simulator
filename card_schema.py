"""Pure card data and hook contracts. Hooks may return generators for decisions."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

CARD_HOOKS: dict[tuple[str, str], list[Callable]] = {}

def register_card_hook(card_id: str, hook_name: str):
    def register(fn):
        bucket = CARD_HOOKS.setdefault((card_id, hook_name), [])
        if fn not in bucket:
            bucket.append(fn)
        return fn
    return register

class Resistance(str, Enum):
    IMMUNE = 'IMMUNE'
    TAX_HAND_1 = 'TAX_HAND_1'
    TAX_HAND_2 = 'TAX_HAND_2'
    TAX_AP_1 = 'TAX_AP_1'

@dataclass(frozen=True)
class Protection:
    kind: Resistance
    applies_to_snipe: bool = False

@dataclass
class CardBase:
    id: str
    name: str
    card_type: str = 'CHARACTER'
    color: str = 'BLUE'
    ap_cost: int = 1
    energy_req: int = 0
    energy_gen: int = 1
    bp: int = 1500
    damage: int = 1
    impact: int = 0
    is_raid: bool = False
    is_snipe: bool = False
    is_step: bool = False
    double_attack: bool = False
    double_block: bool = False
    impact_nullify: bool = False
    raid_targets: list[str] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)
    trigger: str = 'NONE'
    protections: list[Protection] = field(default_factory=list)
    color_trigger_max_bp: int = 3500
    color_trigger_target_color: str | None = None
    color_trigger_dest: str = 'DROP'
    canonical_name: str = ''

    def __post_init__(self):
        if not self.canonical_name:
            self.canonical_name = self.name

@dataclass
class CardInstance:
    base: CardBase
    uid: int
    owner: int = 0
    zone: str = 'DECK'
    under_cards: list['CardInstance'] = field(default_factory=list)
    is_face_down: bool = False
    effects_enabled: bool = True
    is_active: bool = True
    is_raided: bool = False
    cannot_attack: bool = False
    cannot_block: bool = False
    can_attack_from_energy: bool = False
    unblockable_by_bp_at_least: int | None = None
    temp_bp_mod: int = 0
    temp_damage_mod: int = 0
    temp_impact_mod: int = 0
    temp_energy_mod: int = 0
    attack_count_this_turn: int = 0
    block_count_this_turn: int = 0
    used_once_per_turn: set[str] = field(default_factory=set)

    def iter_hooks(self, name, state, allow_disabled=False, **kwargs):
        if self.effects_enabled or allow_disabled:
            for fn in CARD_HOOKS.get((self.base.id, name), []):
                yield fn(self, state, **kwargs)

    def trigger_hook(self, name, state, **kwargs):
        return list(self.iter_hooks(name, state, **kwargs))

    def _value(self, state, base, temporary, hook):
        return max(0, base + temporary + sum(x for x in self.trigger_hook(hook, state) if x is not None))

    def get_current_bp(self, state):
        return self._value(state, self.base.bp, self.temp_bp_mod, 'MOD_BP')

    def get_current_damage(self, state):
        return self._value(state, self.base.damage, self.temp_damage_mod, 'MOD_DAMAGE')

    def get_current_impact(self, state):
        return self._value(state, self.base.impact, self.temp_impact_mod, 'MOD_IMPACT')

    def get_energy_requirement(self, state):
        return self._value(state, self.base.energy_req, self.temp_energy_mod, 'MOD_ENERGY_REQ')

    def clear_temporary(self):
        self.temp_bp_mod = self.temp_damage_mod = self.temp_impact_mod = self.temp_energy_mod = 0
        self.cannot_attack = self.cannot_block = False
        self.can_attack_from_energy = False
        self.unblockable_by_bp_at_least = None
        self.attack_count_this_turn = self.block_count_this_turn = 0
        self.used_once_per_turn.clear()

    def leave_field_reset(self):
        self.clear_temporary()
        self.is_active = self.effects_enabled = True
        self.is_raided = self.is_face_down = False

@dataclass
class AttackEvent:
    attacker_card: CardInstance
    target: CardInstance | None = None
    is_snipe: bool = False
    blocked: bool = False
    cancelled: bool = False

@dataclass
class Choice:
    kind: str
    card: CardInstance | None = None
    target: CardInstance | None = None
    zone: str = ''
    amount: int = 0

@dataclass
class Decision:
    player: int
    kind: str
    choices: list[Choice]


def load_deck(path):
    """Load the explicit user card profile, validating counts and trigger limits."""
    import json
    from collections import Counter
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    deck = []
    for row in data['cards']:
        row = dict(row)
        count = row.pop('count')
        if not isinstance(count, int) or not 1 <= count <= 4:
            raise ValueError('Card count must be 1..4')
        card = CardBase(**row)
        deck.extend([card] * count)
    if len(deck) != 50 or max(Counter(c.id for c in deck).values()) > 4:
        raise ValueError('Deck must have 50 cards and at most four copies per ID')
    for trigger in ('COLOR', 'FINAL', 'SPECIAL'):
        if sum(c.trigger == trigger for c in deck) > 4:
            raise ValueError(f'Too many {trigger} triggers')
    return deck

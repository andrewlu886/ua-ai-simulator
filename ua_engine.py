"""Headless UA rules profile, resumable decisions and fixed-size numeric Gym API.

Each step belongs to info['acting_player']; terminal reward is from that actor's
perspective. Use SelfPlayEnv below for PPO: raw alternating-player trajectories
must not be fed to a conventional single-agent discounted-return algorithm.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from types import GeneratorType
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from card_schema import CARD_HOOKS, CardBase, CardInstance, AttackEvent, Choice, Decision, Resistance
import plugins  # Explicit current-contract plugin registration.

ZONES = ('DECK', 'HAND', 'FRONT', 'ENERGY', 'LIFE', 'DROP', 'REMOVE')
KINDS = ('PASS', 'NEXT', 'DRAW', 'PLAY', 'RAID', 'MOVE', 'SWAP', 'ATTACK',
         'SNIPE', 'BLOCK', 'PAY', 'DISCARD', 'TARGET', 'TRIGGER', 'ACTIVE',
         'HAND', 'SHIELD', 'DETACH', 'REMOVE', 'KEEP', 'MULLIGAN', 'ACTIVATE_MAIN', 'HAND_ACTIVATE_MAIN', 'TOP', 'DROP', 'BUFF', 'REST', 'COUNT')
DECISIONS = ('MOVE', 'MAIN', 'ATTACK', 'BLOCK', 'TAX', 'DISCARD', 'TARGET',
             'TRIGGER', 'ACTIVE', 'RAID', 'RAID_MOVE', 'SPACE', 'END', 'SHIELD', 'DETACH', 'MULLIGAN', 'EFFECT', 'PEEK', 'COST', 'RAID_STACK')
TRIGGERS = ('NONE', 'DRAW', 'ACTIVE', 'COLOR', 'SPECIAL', 'FINAL', 'RAID', 'GET')
COLORS = ('BLUE', 'RED', 'GREEN', 'YELLOW', 'PURPLE', 'WHITE')
MAX_ACTIONS = 512
MAX_NAMES = 64
MAX_TRAITS = 32
CARD_FEATURES = 36 + MAX_TRAITS
# Field 16 + hand 50 + revealed 7 + public piles 200 + private view 50 + battle 2 + under 100.
CARD_SLOTS = 425
OBS_SIZE = 38 + 2 * MAX_NAMES * 5 + CARD_SLOTS * CARD_FEATURES + MAX_ACTIONS * 7

@dataclass
class PlayerState:
    index: int
    zones: dict = field(default_factory=lambda: {z: [] for z in ZONES})
    ap: int = 0
    max_ap: int = 0
    turns: int = 0
    extra_drawn: bool = False
    raids_this_turn: int = 0
    attacks_this_turn: int = 0
    max_attacks_per_turn: int | None = None
    hand_dropped_this_turn: int = 0
    used_named: set[tuple[str, str]] = field(default_factory=set)

    def __getattr__(self, name):
        zone = 'REMOVE' if name == 'remove_area' else name.upper()
        if zone in ZONES:
            return self.zones[zone]
        raise AttributeError(name)

    def energy_for(self, color):
        return sum(c.base.energy_gen for c in self.energy if c.base.color == color)

    @property
    def total_energy(self):
        return sum(c.base.energy_gen for c in self.energy)

class FastUAEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, deck_a, deck_b, record_replay=False, max_steps=10000):
        super().__init__()
        if len(deck_a) != 50 or len(deck_b) != 50:
            raise ValueError('Each deck must contain exactly 50 cards')
        self.templates = (list(deck_a), list(deck_b))
        self.card_codes = {v: i + 1 for i, v in enumerate(sorted({c.id for d in self.templates for c in d}))}
        names = sorted({c.canonical_name for d in self.templates for c in d})
        traits = sorted({t for d in self.templates for c in d for t in c.traits})
        if len(names) > MAX_NAMES or len(traits) > MAX_TRAITS:
            raise ValueError('Observation vocabulary capacity exceeded')
        self.name_codes = {name: i + 1 for i, name in enumerate(names)}
        self.trait_codes = {name: i for i, name in enumerate(traits)}
        self.record_replay = record_replay
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)
        self.pending = None
        self._flow = None

    @property
    def p1(self): return self.players[0]
    @property
    def p2(self): return self.players[1]
    @property
    def active_player(self): return self.players[self.turn_player]
    @property
    def phase(self): return self.pending.kind if self.pending else 'FINISHED'

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(2**31))
            super().reset(seed=seed)
        self.seed_value = seed
        self.players = [PlayerState(0), PlayerState(1)]
        self.cards = []
        self.revealed = []
        self.private_views = {0: [], 1: []}
        self.replay_events = []
        self.actions = []
        self.steps = self.turn = 0
        self.turn_player = self.deciding_player = 0
        self.winner = None
        self.reason = None
        self.truncated = False
        self.battle_event = None
        for p, deck in zip(self.players, self.templates):
            cards = [CardInstance(b, p.index * 50 + i, p.index) for i, b in enumerate(deck)]
            self.cards.extend(cards)
            self.np_random.shuffle(cards)
            p.deck.extend(cards)
            self.draw(p.index, 7)
        self._flow = self.setup_game()
        self._advance()
        return self.observation(), self.info()

    def setup_game(self):
        # No life cards are dealt until BOTH players have made their one-time choice.
        for p in self.players:
            pick = yield from self.choose(p.index, 'MULLIGAN', [Choice('KEEP'), Choice('MULLIGAN')])
            if pick.kind == 'MULLIGAN':
                held = list(p.hand)
                p.hand.clear()
                # Atomic: no decision or hook can observe the temporarily set-aside cards.
                self.draw(p.index, 7)
                for card in held:
                    self.transfer(card, 'DECK')
                self.np_random.shuffle(p.deck)
                self.log('MULLIGAN_SHUFFLE', player=p.index)
        for p in self.players:
            for _ in range(7):
                self.transfer(p.deck[0], 'LIFE')
        yield from self.game()

    def info(self):
        return {'deciding_player': self.deciding_player, 'turn_player': self.turn_player,
                'phase': self.phase, 'winner': self.winner, 'reason': self.reason}

    def log(self, event, **data):
        if self.record_replay:
            self.replay_events.append({'turn': self.turn, 'event': event, **data})

    def transfer(self, card, zone, *, allow_removed_to_drop=False, track_hand_drop=True):
        """Low-level movement. Field departure must go through leave_field()."""
        if card.zone == 'REMOVE' and zone != 'REMOVE' and not (zone == 'DROP' and allow_removed_to_drop):
            raise ValueError('Removed cards cannot be reused')
        p = self.players[card.owner]
        if track_hand_drop and card.zone == 'HAND' and zone == 'DROP':
            p.hand_dropped_this_turn += 1
        for pile in p.zones.values():
            if card in pile:
                pile.remove(card)
                break
        if card in self.revealed:
            self.revealed.remove(card)
        card.zone = zone
        card.is_face_down = zone == 'LIFE'
        p.zones[zone].append(card)
        self.log('ZONE', uid=card.uid, zone=zone)

    def draw(self, player, count=1, start_phase=False):
        p = self.players[player]
        for _ in range(count):
            if not p.deck:
                if start_phase:
                    self.winner, self.reason = 1 - player, 'DECK_OUT'
                return
            self.transfer(p.deck[0], 'HAND')

    def choose(self, player, kind, choices):
        if not choices or len(choices) > MAX_ACTIONS:
            raise ValueError(f'Invalid decision size: {kind}/{len(choices)}')
        return (yield Decision(player, kind, choices))

    def hooks(self, card, name, allow_disabled=False, **kwargs):
        results = []
        for result in card.iter_hooks(name, self, allow_disabled=allow_disabled, **kwargs):
            if isinstance(result, GeneratorType):
                result = yield from result
            results.append(result)
            if name == 'ON_BEFORE_LEAVE_FIELD' and result is True:
                break
        self.log(name, uid=card.uid)
        return results

    def _advance(self, choice=None):
        try:
            self.pending = next(self._flow) if choice is None else self._flow.send(choice)
            self.deciding_player = self.pending.player
        except StopIteration:
            self.pending = None
        if self.winner is not None:
            self.pending = None

    def action_masks(self):
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        if self.pending and self.winner is None and not self.truncated:
            mask[:len(self.pending.choices)] = True
        return mask

    def step(self, action):
        if not self.action_space.contains(action) or not self.action_masks()[int(action)]:
            raise ValueError(f'Illegal action {action} in {self.phase}')
        actor = self.deciding_player
        choice = self.pending.choices[int(action)]
        self.actions.append(int(action))
        self.log('DECISION', player=actor, phase=self.phase, action=int(action),
                 kind=choice.kind, card=choice.card.uid if choice.card else None,
                 target=choice.target.uid if choice.target else None, zone=choice.zone)
        self.steps += 1
        self._advance(choice)
        if self.steps >= self.max_steps and self.winner is None:
            self.truncated, self.reason = True, 'STEP_LIMIT'
        terminated = self.winner is not None
        reward = (1.0 if self.winner == actor else -1.0) if terminated else 0.0
        info = self.info()
        info['acting_player'] = actor
        return self.observation(), reward, terminated, self.truncated, info

    def protections(self, card):
        values = list(card.base.protections)
        if card.zone in ('FRONT', 'ENERGY'):
            for ally in self.players[card.owner].front + self.players[card.owner].energy:
                for result in ally.trigger_hook('PROTECT_ALLY', self, target=card):
                    values.extend(result or [])
        # Identical passive abilities do not multiply the same tax.
        return list(dict.fromkeys(values))

    def target_cost(self, target, snipe=False):
        hand = ap = 0
        for r in self.protections(target):
            if snipe and (r.kind == Resistance.IMMUNE or not r.applies_to_snipe):
                continue
            if r.kind == Resistance.IMMUNE:
                return None
            hand += {Resistance.TAX_HAND_1: 1, Resistance.TAX_HAND_2: 2}.get(r.kind, 0)
            ap += int(r.kind == Resistance.TAX_AP_1)
        return hand, ap

    def pay_target(self, actor, target, snipe=False):
        cost = self.target_cost(target, snipe)
        if cost is None:
            return False
        hand, ap = cost
        if not hand and not ap:
            return True
        p = self.players[actor]
        choices = [Choice('PASS')]
        if len(p.hand) >= hand and p.ap >= ap:
            choices.append(Choice('PAY', target))
        pick = yield from self.choose(actor, 'TAX', choices)
        if pick.kind == 'PASS':
            return False
        p.ap -= ap
        for _ in range(hand):
            discard = yield from self.choose(actor, 'DISCARD', [Choice('DISCARD', c) for c in p.hand])
            self.transfer(discard.card, 'DROP')
        return True

    def detach_to_drop(self, parent, source):
        parent.under_cards.remove(source)
        for child in list(source.under_cards):
            self.detach_to_drop(source, child)
        source.leave_field_reset()
        self.transfer(source, 'DROP')

    def leave_field(self, card, dest='DROP', retire=True):
        if card.zone not in ('FRONT', 'ENERGY'):
            return False
        results = yield from self.hooks(card, 'ON_BEFORE_LEAVE_FIELD', dest=dest, is_retire=retire)
        if any(result is True for result in results):
            self.log('LEAVE_REPLACED', uid=card.uid)
            return False
        for child in list(card.under_cards):
            self.detach_to_drop(card, child)
        self.transfer(card, dest)
        if dest == 'DROP' and retire:
            yield from self.hooks(card, 'ON_RETIRE')
        yield from self.hooks(card, 'ON_LEAVE_FIELD', dest=dest)
        card.leave_field_reset()
        return True

    def place(self, card, zone):
        p = self.players[card.owner]
        while len(p.zones[zone]) >= 4:
            pick = yield from self.choose(card.owner, 'SPACE', [Choice('REMOVE', c) for c in p.zones[zone]])
            yield from self.leave_field(pick.card, 'REMOVE', False)
        if card.zone == 'UNDER':
            parent = next(c for c in self.cards if card in c.under_cards)
            parent.under_cards.remove(card)
            card.leave_field_reset()
        self.transfer(card, zone)

    def raid_sources(self, card):
        if not card.base.is_raid:
            return []
        candidates = []
        for source in self.players[card.owner].front + self.players[card.owner].energy:
            if source.base.card_type != 'CHARACTER':
                continue
            default = (source.base.canonical_name in card.base.raid_targets or
                       source.base.name in card.base.raid_targets or
                       bool(set(source.base.traits) & set(card.base.raid_targets)))
            verdicts = card.trigger_hook('CAN_RAID_ON', self, source=source, default=default)
            if (all(verdicts) if verdicts else default):
                candidates.append(source)
        return candidates

    def raid(self, card, source, defensive=False):
        p = self.players[card.owner]
        zone = source.zone
        inherited = card.trigger_hook('RAID_INHERIT_SOURCES', self, source=source)
        inherited = inherited[0] if inherited else [source]
        # Preserve every layer while disabling the overlapped source.
        p.zones[zone].remove(source)
        card.under_cards = source.under_cards + [source]
        source.under_cards = []
        for under in card.under_cards:
            under.zone = 'UNDER'
            under.effects_enabled = False
        self.transfer(card, zone)
        card.is_raided = card.is_active = True
        p.raids_this_turn += 1
        yield from self.hooks(card, 'ON_RAID_STACK', source=source)
        self.log('RAID_OVERLAY', uid=card.uid, source=source.uid)
        if zone == 'ENERGY':
            choice = yield from self.choose(card.owner, 'RAID_MOVE', [Choice('PASS'), Choice('MOVE', card, zone='FRONT')])
            if choice.kind == 'MOVE':
                yield from self.place(card, 'FRONT')
        for base in inherited:
            yield from self.hooks(base, 'ON_BEING_RAIDED', allow_disabled=True, raid_card=card)
        yield from self.hooks(card, 'ON_PLAY', is_raid=True, is_defensive_trigger=defensive)

    def effect_remove(self, actor, max_bp=float('inf'), color=None, dest='DROP', optional=True):
        targets = [c for c in self.players[1-actor].front
                   if c.base.card_type == 'CHARACTER' and c.get_current_bp(self) <= max_bp and (color is None or c.base.color == color)
                   and self.target_cost(c) is not None]
        if not targets:
            return
        pick = yield from self.choose(actor, 'TARGET', ([Choice('PASS')] if optional else []) + [Choice('TARGET', c) for c in targets])
        if pick.kind != 'PASS' and (yield from self.pay_target(actor, pick.card)):
            yield from self.leave_field(pick.card, dest, dest == 'DROP')

    def damage(self, player, amount):
        if amount <= 0:
            return
        p = self.players[player]
        # Resolve the entire trigger batch before checking the zero-life loss condition.
        if not p.life:
            self.winner, self.reason = 1-player, 'LIFE_DAMAGE'
            return
        batch = p.life[:amount]
        for card in batch:
            p.life.remove(card)
            # Remains logically LIFE until its trigger resolves, but publicly revealed.
            card.is_face_down = False
            self.revealed.append(card)
        self.log('LIFE_REVEAL', player=player, cards=[c.uid for c in batch])
        while batch and self.winner is None:
            pick = yield from self.choose(player, 'TRIGGER', [Choice('TRIGGER', c) for c in batch])
            card = pick.card
            batch.remove(card)
            trigger = card.base.trigger
            if (card.base.id, 'ON_TRIGGER') in CARD_HOOKS:
                yield from self.hooks(card, 'ON_TRIGGER')
            elif trigger == 'GET':
                self.transfer(card, 'HAND')
            elif trigger == 'DRAW':
                self.draw(player)
            elif trigger == 'ACTIVE':
                targets = [c for c in p.front + p.energy if c.base.card_type == 'CHARACTER']
                if targets:
                    target = yield from self.choose(player, 'ACTIVE', [Choice('PASS')] + [Choice('ACTIVE', c) for c in targets])
                    if target.kind != 'PASS':
                        target.card.is_active = True
                        target.card.temp_bp_mod += 3000
            elif trigger in ('SPECIAL', 'COLOR'):
                yield from self.effect_remove(player,
                    max_bp=card.base.color_trigger_max_bp if trigger == 'COLOR' else float('inf'),
                    color=card.base.color_trigger_target_color if trigger == 'COLOR' else None,
                    dest=card.base.color_trigger_dest if trigger == 'COLOR' else 'DROP', optional=False)
            elif trigger == 'FINAL':
                if p.life:
                    self.transfer(card, 'HAND')
                elif p.deck:
                    # This is life recovery, not a draw or another trigger check.
                    self.transfer(p.deck[0], 'LIFE')
            elif trigger == 'RAID':
                sources = self.raid_sources(card) if p.energy_for(card.base.color) >= card.get_energy_requirement(self) else []
                options = [Choice('HAND', card)] + [Choice('RAID', card, c) for c in sources]
                result = yield from self.choose(player, 'RAID', options)
                if result.kind == 'RAID':
                    yield from self.raid(card, result.target, defensive=True)
                else:
                    self.transfer(card, 'HAND')
            if card in self.revealed:
                self.transfer(card, 'DROP')
            self.log('TRIGGER_RESOLVED', uid=card.uid, trigger=trigger)
        if not p.life and self.winner is None:
            self.winner, self.reason = 1-player, 'LIFE_DAMAGE'

    def battle(self, event):
        attacker, defender = event.attacker_card, event.target
        if event.cancelled or not self.attack_position(attacker) or defender.zone != 'FRONT':
            event.cancelled = True
            return
        self.log('BATTLE', attacker=attacker.uid, defender=defender.uid, blocked=event.blocked)
        attack_bp, defend_bp = attacker.get_current_bp(self), defender.get_current_bp(self)
        impact = attacker.get_current_impact(self)
        nullify = defender.base.impact_nullify
        if attack_bp >= defend_bp:
            yield from self.hooks(defender, 'ON_BATTLE_LOSE', event=event)
            yield from self.leave_field(defender)
            yield from self.hooks(attacker, 'ON_BATTLE_WIN', event=event)
            if event.blocked and impact and not nullify:
                yield from self.damage(defender.owner, impact)
        else:
            # UA's attacking character is not retired for losing this comparison.
            yield from self.hooks(attacker, 'ON_BATTLE_LOSE', event=event)
            yield from self.hooks(defender, 'ON_BATTLE_WIN', event=event)

    def attack(self, card, target=None):
        p = self.players[card.owner]
        if p.max_attacks_per_turn is not None and p.attacks_this_turn >= p.max_attacks_per_turn:
            return
        if target and not (yield from self.pay_target(card.owner, target, True)):
            return
        p.attacks_this_turn += 1
        card.attack_count_this_turn += 1
        card.is_active = bool(card.base.double_attack and card.attack_count_this_turn == 1)
        event = AttackEvent(card, target, target is not None)
        self.battle_event = event
        yield from self.hooks(card, 'ON_ATTACK', event=event)
        if not self.attack_position(card) or event.cancelled:
            self.battle_event = None
            return
        if target is not None:
            yield from self.hooks(card, 'ON_UNBLOCKED', event=event)
            yield from self.battle(event)
        else:
            defenders = [c for c in self.players[1-card.owner].front
                         if c.base.card_type == 'CHARACTER' and c.is_active and not c.cannot_block
                         and (card.unblockable_by_bp_at_least is None or
                              c.get_current_bp(self) < card.unblockable_by_bp_at_least)]
            result = yield from self.choose(1-card.owner, 'BLOCK', [Choice('PASS')] + [Choice('BLOCK', c) for c in defenders])
            if result.kind == 'BLOCK':
                blocker = result.card
                event.blocked, event.target = True, blocker
                blocker.block_count_this_turn += 1
                blocker.is_active = bool(blocker.base.double_block and blocker.block_count_this_turn == 1)
                yield from self.hooks(blocker, 'ON_BLOCK', event=event)
                yield from self.battle(event)
            else:
                yield from self.hooks(card, 'ON_UNBLOCKED', event=event)
                if self.attack_position(card) and not event.cancelled:
                    yield from self.damage(1-card.owner, card.get_current_damage(self))
        self.battle_event = None

    def attack_position(self, card):
        return card.zone == 'FRONT' or (card.zone == 'ENERGY' and card.can_attack_from_energy)

    def activation_available(self, card, hand=False):
        hook = 'CAN_HAND_ACTIVATE_MAIN' if hand else 'CAN_ACTIVATE_MAIN'
        return any(card.trigger_hook(hook, self))

    def state_checks(self):
        # BP reduction to zero retires a character through the normal replacement path.
        for p in self.players:
            for card in list(p.front + p.energy):
                if card.base.card_type == 'CHARACTER' and card.get_current_bp(self) <= 0:
                    yield from self.leave_field(card)

    def game(self):
        while self.winner is None:
            p = self.players[self.turn_player]
            self.turn += 1
            p.turns += 1
            base = 1 if self.turn == 1 else 2 if self.turn in (2, 3) else 3
            p.max_ap = base
            p.ap, p.extra_drawn = p.max_ap, False
            for card in p.front + p.energy:
                card.is_active = True
            if self.turn != 1:
                self.draw(p.index, start_phase=True)
            if self.winner is not None:
                break
            self.log('START_TURN', player=p.index, ap=p.ap)
            moved = set()
            while True:
                choices = [Choice('NEXT')]
                for card in p.front + p.energy:
                    if card.uid in moved or card.base.card_type != 'CHARACTER':
                        continue
                    zone = 'FRONT' if card.zone == 'ENERGY' else 'ENERGY'
                    if card.zone == 'ENERGY' or card.base.is_step:
                        choices.append(Choice('MOVE', card, zone=zone))
                    if card.base.is_step:
                        choices += [Choice('SWAP', card, c) for c in p.zones[zone]
                                    if c.base.card_type == 'CHARACTER' and c.uid not in moved]
                pick = yield from self.choose(p.index, 'MOVE', choices)
                if pick.kind == 'NEXT':
                    break
                moved.add(pick.card.uid)
                if pick.kind == 'MOVE':
                    yield from self.place(pick.card, pick.zone)
                else:
                    a, b = pick.card, pick.target
                    za, zb = a.zone, b.zone
                    ia, ib = p.zones[za].index(a), p.zones[zb].index(b)
                    p.zones[za][ia], p.zones[zb][ib] = b, a
                    a.zone, b.zone = zb, za
                    moved.add(b.uid)
            while self.winner is None:
                choices = [Choice('NEXT')]
                if p.ap and not p.extra_drawn and self.turn != 1:
                    choices.append(Choice('DRAW'))
                choices += [Choice('ACTIVATE_MAIN', c) for c in p.front + p.energy
                            if self.activation_available(c)]
                for card in p.hand:
                    if self.activation_available(card, hand=True):
                        choices.append(Choice('HAND_ACTIVATE_MAIN', card))
                    if p.ap < card.base.ap_cost or p.energy_for(card.base.color) < card.get_energy_requirement(self):
                        continue
                    choices.append(Choice('PLAY', card, zone='ENERGY'))
                    if card.base.card_type == 'CHARACTER':
                        choices.append(Choice('PLAY', card, zone='FRONT'))
                        choices += [Choice('RAID', card, c) for c in self.raid_sources(card)]
                pick = yield from self.choose(p.index, 'MAIN', choices)
                if pick.kind == 'NEXT':
                    break
                if pick.kind == 'DRAW':
                    p.ap -= 1
                    p.extra_drawn = True
                    self.draw(p.index)
                    continue
                card = pick.card
                if pick.kind in ('ACTIVATE_MAIN', 'HAND_ACTIVATE_MAIN'):
                    yield from self.hooks(card, pick.kind)
                    yield from self.state_checks()
                    continue
                p.ap -= card.base.ap_cost
                if pick.kind == 'RAID':
                    yield from self.raid(card, pick.target)
                elif card.base.card_type == 'EVENT':
                    self.transfer(card, 'DROP', track_hand_drop=False)
                    yield from self.hooks(card, 'ON_PLAY', is_raid=False)
                else:
                    yield from self.place(card, pick.zone)
                    card.is_active = False
                    yield from self.hooks(card, 'ON_PLAY', is_raid=False)
                yield from self.state_checks()
            while self.winner is None:
                choices = [Choice('NEXT')]
                if self.turn != 1 and (p.max_attacks_per_turn is None or p.attacks_this_turn < p.max_attacks_per_turn):
                    for card in p.front + p.energy:
                        if not self.attack_position(card):
                            continue
                        if card.base.card_type != 'CHARACTER' or not card.is_active or card.cannot_attack:
                            continue
                        choices.append(Choice('ATTACK', card))
                        if card.base.is_snipe:
                            choices += [Choice('SNIPE', card, c) for c in self.players[1-p.index].front
                                        if c.base.card_type == 'CHARACTER']
                pick = yield from self.choose(p.index, 'ATTACK', choices)
                if pick.kind == 'NEXT':
                    break
                yield from self.attack(pick.card, pick.target)
            if self.winner is not None:
                break
            for card in list(p.front + p.energy):
                if card.zone in ('FRONT', 'ENERGY'):
                    yield from self.hooks(card, 'ON_END_ATTACK_PHASE')
            while len(p.hand) > 8:
                pick = yield from self.choose(p.index, 'END', [Choice('REMOVE', c) for c in p.hand])
                self.transfer(pick.card, 'REMOVE')
            self.end_turn_cleanup(p)
            self.turn_player = 1-self.turn_player

    def end_turn_cleanup(self, current):
        # All this-turn effects expire, then only the turn player's field becomes active.
        for player in self.players:
            player.raids_this_turn = player.attacks_this_turn = player.hand_dropped_this_turn = 0
            player.max_attacks_per_turn = None
            player.used_named.clear()
        for card in self.cards:
            card.clear_temporary()
        for card in current.front + current.energy:
            if card.base.card_type in ('CHARACTER', 'FIELD'):
                card.is_active = True
        # AP is intentionally untouched.
        self.log('END_ACTIVE_RESET', player=current.index)

    def card_vector(self, card, viewer=None):
        if card is None:
            return [0.0] * CARD_FEATURES
        viewer = self.deciding_player if viewer is None else viewer
        if card.zone == 'UNDER' and card.is_face_down and card.owner != viewer:
            return [1.0] + [0.0] * (CARD_FEATURES - 1)
        res = self.protections(card)
        return [1, self.card_codes[card.base.id] / 100, card.get_current_bp(self) / 10000,
                card.get_current_damage(self) / 10, card.get_current_impact(self) / 10,
                card.get_energy_requirement(self) / 10, card.base.energy_gen / 10,
                card.base.ap_cost / 10, float(card.is_active), float(card.is_raided),
                float(card.base.is_snipe), float(card.base.is_step), float(card.base.double_attack),
                float(card.base.double_block), float(card.base.impact_nullify),
                len(card.under_cards) / 50, sum(c.is_face_down for c in card.under_cards) / 50,
                *[float(any(r.kind == k for r in res)) for k in Resistance],
                float(any(r.applies_to_snipe for r in res)),
                card.attack_count_this_turn / 2, card.block_count_this_turn / 2,
                float(card.cannot_attack), float(card.cannot_block),
                TRIGGERS.index(card.base.trigger) / (len(TRIGGERS) - 1),
                (COLORS.index(card.base.color) + 1) / 6 if card.base.color in COLORS else 0,
                float(card.base.is_raid), float(bool(card.used_once_per_turn)),
                ('CHARACTER', 'FIELD', 'EVENT').index(card.base.card_type) / 2,
                float(card.effects_enabled), self.name_codes[card.base.canonical_name] / MAX_NAMES,
                float(card.can_attack_from_energy), (card.unblockable_by_bp_at_least or 0) / 10000,
                float(any(n == card.base.canonical_name for n, _ in self.players[card.owner].used_named)),
                *[float(t in card.base.traits) for t in self.trait_codes],
                *([0.0] * (MAX_TRAITS - len(self.trait_codes)))]

    def observation(self, ego=None):
        ego = self.deciding_player if ego is None else ego
        own, other = self.players[ego], self.players[1-ego]
        header = [float(self.turn_player == ego), self.turn / 100,
                  (DECISIONS.index(self.phase) + 1) / len(DECISIONS) if self.phase in DECISIONS else 0,
                  self.steps / self.max_steps]
        for p in (own, other):
            header += [len(p.life)/7, len(p.hand)/50, len(p.deck)/50, p.ap/10, p.max_ap/10,
                       0.0, float(p.extra_drawn), p.turns/50,
                       len(p.drop)/50, len(p.remove_area)/50]
        cards = []
        def append_slots(pile, capacity):
            cards.extend(pile)
            cards.extend([None] * (capacity - len(pile)))
        for p in (own, other):
            append_slots(p.front, 4)
            append_slots(p.energy, 4)
        append_slots(own.hand, 50)
        append_slots(self.revealed, 7)
        for p in (own, other):
            append_slots(p.drop, 50)
            append_slots(p.remove_area, 50)
        append_slots(self.private_views[ego], 50)
        event = self.battle_event
        append_slots([event.attacker_card, event.target] if event else [], 2)
        for p in (own, other):
            append_slots([c for c in self.cards if c.owner == p.index and c.zone == 'UNDER'], 50)
        header += [float(event.is_snipe) if event else 0, float(event.blocked) if event else 0,
                   float(event.cancelled) if event else 0,
                   float(self.revealed[0].owner == ego) if self.revealed else 0]
        for p in (own, other):
            header += [p.raids_this_turn / 50, p.attacks_this_turn / 50,
                       -1 if p.max_attacks_per_turn is None else p.max_attacks_per_turn / 50,
                       p.hand_dropped_this_turn / 50, len(p.used_named) / 50]
        resources = np.zeros((2, MAX_NAMES, 5), dtype=np.float32)
        for side, p in enumerate((own, other)):
            for zi, zone in enumerate(('FRONT', 'ENERGY', 'DROP', 'REMOVE')):
                for c in p.zones[zone]:
                    resources[side, self.name_codes[c.base.canonical_name] - 1, zi] += 1 / 50
            for name, _ in p.used_named:
                resources[side, self.name_codes[name] - 1, 4] += 1
        references = {c.uid: i + 1 for i, c in enumerate(cards) if c is not None}
        # Context-dependent action indices are explicitly described in the observation.
        actions = np.zeros((MAX_ACTIONS, 7), dtype=np.float32)
        if self.pending:
            for i, ch in enumerate(self.pending.choices):
                actions[i] = [1, (KINDS.index(ch.kind)+1)/len(KINDS),
                              references.get(ch.card.uid, 0)/CARD_SLOTS if ch.card else 0,
                              references.get(ch.target.uid, 0)/CARD_SLOTS if ch.target else 0,
                              (ZONES.index(ch.zone)+1)/7 if ch.zone in ZONES else 0,
                              float(ch.card.is_face_down) if ch.card and ch.card.zone == 'UNDER' else 0, ch.amount / 50]
        result = np.concatenate([np.array(header, np.float32), resources.ravel(),
                                 np.array([self.card_vector(c, ego) for c in cards], np.float32).ravel(), actions.ravel()])
        return result.astype(np.float32, copy=False)

    def assert_invariants(self):
        seen = []
        def visit(card, owner, zone):
            assert card.owner == owner and card.zone == zone
            seen.append(card.uid)
            for under in card.under_cards:
                assert not under.effects_enabled
                visit(under, owner, 'UNDER')
        for p in self.players:
            assert 0 <= p.ap and len(p.front) <= 4 and len(p.energy) <= 4
            for z, pile in p.zones.items():
                for card in pile:
                    visit(card, p.index, z)
        for c in self.revealed:
            visit(c, c.owner, 'LIFE')
        assert len(seen) == 100 and len(set(seen)) == 100, 'Card conservation failed'

    def replay_data(self):
        """Return one complete, JSON-serializable game replay."""
        return {'seed': self.seed_value, 'winner': self.winner,
            'reason': self.reason, 'truncated': self.truncated, 'total_turns': self.turn,
            'actions': list(self.actions), 'events': list(self.replay_events)}

    def export_replay(self, filepath, winner=None):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'schema_version': 3,
            'rules_profile': 'user-specified-decks-v3', **self.replay_data(),
            'name_codes': self.name_codes, 'trait_codes': self.trait_codes,
            'decks': [[c.id for c in d] for d in self.templates]},
            ensure_ascii=False, indent=2), encoding='utf-8')


class SelfPlayEnv(gym.Env):
    """A fixed-seat learner vs an opponent callback, preserving correct PPO returns.

    opponent(obs, mask) -> action. Supply a frozen MaskablePPO policy snapshot for
    self-play; default is a seeded random legal opponent. Reward is learner-relative.
    """
    def __init__(self, engine, opponent=None, learner=0):
        super().__init__()
        self.engine, self.opponent, self.learner = engine, opponent, learner
        self.action_space, self.observation_space = engine.action_space, engine.observation_space

    def action_masks(self):
        return self.engine.action_masks()

    def _opponents(self, obs, terminated=False, truncated=False, info=None):
        while not (terminated or truncated) and self.engine.deciding_player != self.learner:
            mask = self.action_masks()
            action = (self.opponent(obs, mask) if self.opponent else
                      int(self.np_random.choice(np.flatnonzero(mask))))
            obs, _, terminated, truncated, info = self.engine.step(int(action))
        if terminated or truncated:
            obs = self.engine.observation(self.learner)
        reward = (1.0 if self.engine.winner == self.learner else -1.0) if terminated else 0.0
        return obs, reward, terminated, truncated, info or self.engine.info()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.engine.reset(seed=seed, options=options)
        obs, _, terminated, truncated, info = self._opponents(obs, info=info)
        if terminated or truncated:
            raise RuntimeError('Opponent ended episode before learner received a decision')
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.engine.step(action)
        return self._opponents(obs, terminated, truncated, info)

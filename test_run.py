"""Rule regressions, 100 seeded legal random games, replay and PPO smoke test.

Run: python test_run.py
Optional: python test_run.py --train-steps 256
"""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import argparse
import json
import numpy as np
from gymnasium.utils.env_checker import check_reset_seed_determinism
from gymnasium.utils.passive_env_checker import env_step_passive_checker
from card_schema import load_deck, CardBase, Choice, AttackEvent, Protection, Resistance, register_card_hook
from ua_engine import FastUAEnv, SelfPlayEnv


def build_test_deck():
    # 13 distinct synthetic cards; at most four of any ID, all six triggers.
    cards = [
        CardBase('BASE_A', 'Scout', traits=['unit'], trigger='DRAW'),
        CardBase('BASE_B', 'Runner', bp=2000, is_step=True, traits=['unit'], trigger='ACTIVE'),
        CardBase('BASE_C', 'Generator', bp=2500, energy_req=1, energy_gen=2, traits=['unit']),
        CardBase('SNIPE_DRAW', 'Sniper', bp=3000, energy_req=2, is_snipe=True, trigger='COLOR'),
        CardBase('AURA', 'Ward', bp=2000, energy_req=1, trigger='SPECIAL'),
        CardBase('SHIELD', 'Layered ace', bp=4000, energy_req=2, is_raid=True,
                 raid_targets=['unit'], traits=['unit'], trigger='RAID'),
        CardBase('DETACH', 'Releaser', bp=3500, energy_req=2, is_raid=True,
                 raid_targets=['unit'], traits=['unit'], trigger='RAID'),
        CardBase('DISCOUNT', 'Reserve', bp=3500, energy_req=3, trigger='FINAL'),
        CardBase('GROW', 'Veteran', bp=3000, energy_req=2, damage=2, impact=1,
                 double_attack=True, traits=['unit'], trigger='ACTIVE'),
        CardBase('GUARD', 'Guard', bp=3000, energy_req=1, double_block=True, impact_nullify=True,
                 protections=[Protection(Resistance.TAX_HAND_1, True)], trigger='DRAW'),
        CardBase('TAX_TWO', 'Fortress', bp=3500, energy_req=2,
                 protections=[Protection(Resistance.TAX_HAND_2, True)], trigger='NONE'),
        CardBase('TAX_AP', 'Gate', bp=3000, energy_req=1,
                 protections=[Protection(Resistance.TAX_AP_1)], trigger='NONE'),
        CardBase('EVENT', 'Removal', card_type='EVENT', energy_gen=0, energy_req=2, trigger='DRAW'),
    ]
    deck = [c for c in cards for _ in range(4 if c.id != 'EVENT' else 2)]
    assert len(deck) == 50 and max(Counter(c.id for c in deck).values()) <= 4
    return deck


def keep_opening(env):
    while env.phase == 'MULLIGAN':
        env.step(0)


def fixture():
    deck = build_test_deck()
    env = FastUAEnv(deck, list(reversed(deck)), record_replay=True)
    env.reset(seed=17)
    keep_opening(env)
    # End the game generator before a focused isolated rule probe.
    env._flow.close()
    env.pending = None
    for p in env.players:
        for card in list(p.hand + p.life):
            env.transfer(card, 'DECK')
    return env


def get(env, player, card_id, zone):
    card = next(c for c in env.players[player].deck if c.base.id == card_id)
    env.transfer(card, zone)
    return card


def resolve(flow, policy=None):
    decisions = []
    try:
        decision = next(flow)
        while True:
            decisions.append(decision)
            pick = policy(decision) if policy else decision.choices[0]
            decision = flow.send(pick)
    except StopIteration as stop:
        return stop.value, decisions


def event_count(env, name):
    return sum(e['event'] == name for e in env.replay_events)


def opening_and_final_regressions():
    d = build_test_deck()
    # Every keep/redraw combination, correct seats and no early life allocation.
    for picks in ((0, 0), (0, 1), (1, 0), (1, 1)):
        e = FastUAEnv(d, d, record_replay=True)
        obs, info = e.reset(seed=41)
        assert info['phase'] == 'MULLIGAN' and e.turn == 0
        assert e.observation_space.contains(obs)
        expected = []
        for seat, action in enumerate(picks):
            assert e.phase == 'MULLIGAN' and e.deciding_player == seat
            assert [c.kind for c in e.pending.choices] == ['KEEP', 'MULLIGAN']
            np.testing.assert_array_equal(np.flatnonzero(e.action_masks()), [0, 1])
            assert all(len(p.life) == 0 and len(p.hand) == 7 for p in e.players)
            p = e.players[seat]
            original = {c.uid for c in p.hand}
            wanted = [c.uid for c in (p.deck[:7] if action else p.hand)]
            expected.append(wanted)
            e.step(action)
            assert [c.uid for c in p.hand] == wanted
            if action:
                assert original.isdisjoint(wanted)
                assert original <= {c.uid for c in p.deck + p.life}
            e.assert_invariants()
        assert e.turn == 1 and e.phase == 'MOVE' and e.deciding_player == 0
        assert all(len(p.deck) == 36 and len(p.life) == 7 for p in e.players)
        assert all(c.is_face_down for p in e.players for c in p.life)
        assert all([c.uid for c in p.hand] == hand for p, hand in zip(e.players, expected))
        # Seed and decisions reproduce shuffle, hand, and life identities exactly.
        replay = FastUAEnv(d, d, record_replay=True)
        replay.reset(seed=41)
        for action in picks: replay.step(action)
        assert replay.replay_events == e.replay_events
        assert [[c.uid for c in p.deck + p.hand + p.life] for p in replay.players] == [
            [c.uid for c in p.deck + p.hand + p.life] for p in e.players]

    # Both self-play seats receive their own opening decision.
    for seat in (0, 1):
        adapter = SelfPlayEnv(FastUAEnv(d, d), learner=seat, opponent=lambda obs, mask: 0)
        obs, _ = adapter.reset(seed=5)
        assert adapter.engine.deciding_player == seat and adapter.engine.phase == 'MULLIGAN'
        obs, _, terminated, truncated, _ = adapter.step(1)
        assert not terminated and not truncated and adapter.observation_space.contains(obs)
        assert adapter.engine.deciding_player == seat

    # FINAL on the last life recovers exactly the top card, face down, with no trigger check.
    e = fixture()
    final = get(e, 1, 'DISCOUNT', 'LIFE')
    top = e.p2.deck[0]
    top.base = replace(top.base, trigger='DRAW')
    ap = (e.p2.ap, e.p2.max_ap)
    size = len(e.p2.deck)
    _, decisions = resolve(e.damage(1, 1))
    assert e.p2.life == [top] and top.is_face_down and final.zone == 'DROP'
    assert len(e.p2.deck) == size - 1 and not e.p2.hand and e.winner is None
    assert [d.kind for d in decisions] == ['TRIGGER']
    assert event_count(e, 'TRIGGER_RESOLVED') == 1 and (e.p2.ap, e.p2.max_ap) == ap
    e.assert_invariants()

    # Two simultaneous FINAL triggers: the first recovers, the second goes to hand.
    for reverse in (False, True):
        e = fixture()
        finals = [get(e, 1, 'DISCOUNT', 'LIFE') for _ in range(2)]
        top = e.p2.deck[0]
        _, decisions = resolve(e.damage(1, 2), lambda d: d.choices[-1 if reverse else 0])
        first, second = (finals[::-1] if reverse else finals)
        assert first.zone == 'DROP' and second.zone == 'HAND' and not second.is_face_down
        assert e.p2.life == [top] and e.winner is None
        assert len(decisions) == 2 and event_count(e, 'TRIGGER_RESOLVED') == 2
        e.assert_invariants()

    # Empty deck cannot supply life; the last FINAL resolves then the owner loses.
    e = fixture()
    final = get(e, 1, 'DISCOUNT', 'LIFE')
    for c in list(e.p2.deck): e.transfer(c, 'DROP')
    resolve(e.damage(1, 1))
    assert final.zone == 'DROP' and not e.p2.life and e.winner == 0
    assert e.reason == 'LIFE_DAMAGE'
    e.assert_invariants()

    # Cleanup affects both players' temporary modifiers, but only the current field's activity.
    e = fixture()
    own = [get(e, 0, 'BASE_A', 'FRONT'), get(e, 0, 'BASE_B', 'ENERGY')]
    own[1].base = replace(own[1].base, card_type='FIELD')
    opposing = get(e, 1, 'BASE_A', 'FRONT')
    for c in own + [opposing]:
        c.is_active = False
        c.temp_bp_mod = 3000
        c.used_once_per_turn.add('used')
    e.p1.ap, e.p2.ap = 0, 2
    e.end_turn_cleanup(e.p1)
    assert all(c.is_active for c in own) and not opposing.is_active
    assert all(c.temp_bp_mod == 0 and not c.used_once_per_turn for c in own + [opposing])
    assert (e.p1.ap, e.p2.ap) == (0, 2)

    # Integration: a freshly played resting character becomes a legal blocker next turn.
    e = FastUAEnv(d, d)
    e.reset(seed=3)
    keep_opening(e)
    card = get(e, 0, 'BASE_A', 'HAND')
    e.step(0)  # MOVE -> MAIN
    action = next(i for i, ch in enumerate(e.pending.choices)
                  if ch.kind == 'PLAY' and ch.card is card and ch.zone == 'FRONT')
    e.step(action)
    assert not card.is_active and e.p1.ap == 0
    while e.turn == 1: e.step(0)
    assert card.is_active and e.p1.ap == 0
    attacker = get(e, 1, 'BASE_A', 'FRONT')
    while e.phase != 'ATTACK': e.step(0)
    action = next(i for i, ch in enumerate(e.pending.choices) if ch.kind == 'ATTACK' and ch.card is attacker)
    e.step(action)
    assert e.phase == 'BLOCK' and e.deciding_player == 0
    assert any(ch.kind == 'BLOCK' and ch.card is card for ch in e.pending.choices)
    e.assert_invariants()
    print('Mulligan, FINAL and end-phase regressions passed')


def regressions():
    # Opening AP, no first-player opening draw/extra draw, no skipped P1 T2 restriction.
    d = build_test_deck()
    e = FastUAEnv(d, d)
    e.reset(seed=1)
    keep_opening(e)
    aps = []
    for turn in range(1, 6):
        aps.append(e.active_player.ap)
        assert len(e.active_player.hand) == (7 if turn == 1 else 8 if turn in (2, 3) else 9)
        while e.turn == turn:
            choices = e.pending.choices
            if e.turn == 1:
                assert all(c.kind != 'DRAW' for c in choices)
            action = next((i for i,c in enumerate(choices) if c.kind == 'NEXT'), 0)
            e.step(action)
    assert aps == [1, 2, 2, 3, 3]

    # Snipe bypasses immunity, doesn't rest/block, still draws on unblock and loses battle.
    e = fixture()
    sniper = get(e, 0, 'SNIPE_DRAW', 'FRONT')
    defender = get(e, 1, 'BASE_A', 'FRONT')
    defender.temp_bp_mod = 5000
    get(e, 1, 'AURA', 'ENERGY')
    assert e.target_cost(defender) is None and e.target_cost(defender, True) == (0, 0)
    resolve(e.attack(sniper, defender))
    assert defender.is_active and len(e.p1.hand) == 1
    assert event_count(e, 'ON_BLOCK') == 0 and event_count(e, 'ON_UNBLOCKED') == 1
    assert event_count(e, 'ON_BATTLE_LOSE') == 1
    e.assert_invariants()

    # Each hand tax is an explicit payment then sequential card selections; refusal is free.
    e = fixture()
    target = get(e, 1, 'TAX_TWO', 'FRONT')
    for _ in range(2): get(e, 0, 'BASE_A', 'HAND')
    accepted, choices = resolve(e.pay_target(0, target, True), lambda d: d.choices[-1])
    assert accepted and len(e.p1.drop) == 2 and [d.kind for d in choices] == ['TAX','DISCARD','DISCARD']
    rejected, _ = resolve(e.pay_target(0, target, True))
    assert not rejected
    ap_target = get(e, 1, 'TAX_AP', 'FRONT')
    e.p1.ap = 1
    accepted, _ = resolve(e.pay_target(0, ap_target), lambda d: d.choices[-1])
    assert accepted and e.p1.ap == 0

    # Layered raids, source suppression, replacement, bounce resets and silent bottom discard.
    e = fixture()
    source = get(e, 0, 'BASE_A', 'FRONT')
    first = get(e, 0, 'SHIELD', 'HAND')
    second = get(e, 0, 'SHIELD', 'HAND')
    resolve(e.raid(first, source))
    resolve(e.raid(second, first))
    assert len(second.under_cards) == 2 and all(c.is_face_down for c in second.under_cards)
    retired = event_count(e, 'ON_RETIRE')
    result, _ = resolve(e.leave_field(second), lambda d: d.choices[-1])
    assert not result and second.zone == 'FRONT' and len(second.under_cards) == 1
    second.temp_bp_mod = 200
    second.used_once_per_turn.add('test')
    resolve(e.leave_field(second, 'HAND', False))
    assert second.zone == 'HAND' and not second.under_cards and second.temp_bp_mod == 0
    assert not second.used_once_per_turn and event_count(e, 'ON_RETIRE') == retired
    e.assert_invariants()

    # Full field eviction is selected; removed cards cannot return to hand.
    e = fixture()
    for _ in range(4): get(e, 0, 'BASE_A', 'FRONT')
    incoming = get(e, 0, 'BASE_B', 'HAND')
    victim = e.p1.front[2]
    resolve(e.place(incoming, 'FRONT'), lambda d: d.choices[2])
    assert victim in e.p1.remove_area and len(e.p1.front) == 4
    try:
        e.transfer(victim, 'HAND')
        raise AssertionError('REMOVE was reusable')
    except ValueError:
        pass
    e.assert_invariants()

    # All life cards reveal before first decision; ordering, FINAL, ACTIVE and fallback RAID.
    e = fixture()
    active = get(e, 1, 'BASE_A', 'FRONT')
    final = get(e, 1, 'DISCOUNT', 'LIFE')
    get(e, 1, 'BASE_B', 'LIFE')
    get(e, 1, 'SHIELD', 'LIFE')
    flow = e.damage(1, 2)
    dec = next(flow)
    assert len(e.revealed) == 2 and len(e.p2.life) == 1
    dec = flow.send(dec.choices[1])
    assert dec.kind == 'ACTIVE'
    dec = flow.send(dec.choices[-1])
    try: flow.send(dec.choices[0])
    except StopIteration: pass
    assert active.temp_bp_mod == 3000 and final.zone == 'HAND' and not e.revealed
    assert e.winner is None
    raid = e.p2.life[0]
    resolve(e.damage(1, 1))
    assert raid.zone == 'HAND'
    resolve(e.damage(1, 1))
    assert e.winner == 0
    e.assert_invariants()

    # Defensive RAID activates, resolves ON_PLAY, and can move from energy to front.
    e = fixture()
    get(e, 1, 'BASE_C', 'ENERGY')
    raid = get(e, 1, 'SHIELD', 'LIFE')
    _, choices = resolve(e.damage(1, 1), lambda d: d.choices[-1])
    assert raid.zone == 'FRONT' and raid.is_active and raid.under_cards[0].is_face_down
    assert any(d.kind == 'RAID' for d in choices)
    e.assert_invariants()

    # Impact only applies after a declared block, and nullification prevents it.
    for blocked, nullify, expected in [(False,False,0),(True,False,1),(True,True,0)]:
        e = fixture()
        atk = get(e, 0, 'GROW', 'FRONT')
        target = get(e, 1, 'BASE_A', 'FRONT')
        target.base = replace(target.base, impact_nullify=nullify)
        get(e, 1, 'BASE_C', 'LIFE')
        resolve(e.battle(AttackEvent(atk, target, not blocked, blocked)))
        assert event_count(e, 'LIFE_REVEAL') == expected
        e.assert_invariants()

    # Full energy line during source release retains conservation at the interrupt.
    e = fixture()
    parent = get(e, 0, 'DETACH', 'HAND')
    source = get(e, 0, 'BASE_A', 'FRONT')
    resolve(e.raid(parent, source))
    for _ in range(4): get(e, 0, 'BASE_C', 'ENERGY')
    flow = e.hooks(parent, 'ON_END_ATTACK_PHASE')
    decision = next(flow)
    decision = flow.send(decision.choices[-1])
    assert decision.kind == 'SPACE' and source in parent.under_cards
    e.assert_invariants()
    try: flow.send(decision.choices[0])
    except StopIteration: pass
    assert source.zone == 'ENERGY' and source.effects_enabled
    e.assert_invariants()

    # Double Attack/Block reactivation; subsequent manual reactivation is still legal.
    e = fixture()
    atk = get(e, 0, 'GROW', 'FRONT')
    blocker = get(e, 1, 'GUARD', 'FRONT')
    blocker.temp_bp_mod = 5000
    resolve(e.attack(atk), lambda decision: decision.choices[-1])
    assert atk.is_active and blocker.is_active
    resolve(e.attack(atk), lambda decision: decision.choices[-1])
    assert not atk.is_active and not blocker.is_active
    assert atk.attack_count_this_turn == blocker.block_count_this_turn == 2

    # Dynamic values are recalculated on every query (base + temporary + hooks).
    e = fixture()
    c = get(e, 0, 'GROW', 'FRONT')
    c.temp_bp_mod, c.temp_damage_mod, c.temp_impact_mod = 1000, 1, 2
    assert (c.get_current_bp(e), c.get_current_damage(e), c.get_current_impact(e)) == (4000, 3, 3)
    c = get(e, 0, 'DISCOUNT', 'HAND')
    c.temp_energy_mod = 2
    before = c.get_energy_requirement(e)
    get(e, 0, 'BASE_A', 'HAND')
    get(e, 0, 'BASE_B', 'HAND')
    get(e, 0, 'BASE_C', 'HAND')
    assert c.get_energy_requirement(e) < before

    # Normal retirement fires only after arrival in DROP.
    e = fixture()
    victim = get(e, 1, 'BASE_A', 'FRONT')
    resolve(e.leave_field(victim))
    assert victim.zone == 'DROP' and event_count(e, 'ON_RETIRE') == 1

    # COLOR and SPECIAL target actual opposing front-line cards with resistance checks.
    for trigger_id in ('SNIPE_DRAW', 'AURA'):
        e = fixture()
        victim = get(e, 0, 'BASE_A', 'FRONT')
        get(e, 1, trigger_id, 'LIFE')
        resolve(e.damage(1, 1), lambda decision: decision.choices[-1])
        assert victim.zone == 'DROP'

    # Empty deck loses at start; AP follows the standard curve.
    e = FastUAEnv(d, d)
    e.reset(seed=33)
    keep_opening(e)
    while e.turn == 1:
        e.step(0)
    assert e.p2.max_ap == 2
    for c in list(e.p1.deck): e.transfer(c, 'DROP')
    e.draw(0)  # Non-start draw simply has no available card.
    assert e.winner is None
    while e.winner is None:
        e.step(0)
    assert e.winner == 1 and e.reason == 'DECK_OUT'

    # Movement choices include Step swaps; full lines swap without removing either unit.
    e = FastUAEnv(d, d)
    e.reset(seed=37)
    e._flow.close()
    for p in e.players:
        for c in list(p.front + p.energy + p.hand + p.life): e.transfer(c, 'DECK')
    stepper = get(e, 0, 'BASE_B', 'FRONT')
    for _ in range(4): get(e, 0, 'BASE_A', 'ENERGY')
    e._flow = e.game()
    e._advance()
    choice = next(i for i,c in enumerate(e.pending.choices) if c.kind == 'SWAP')
    e.step(choice)
    assert stepper.zone == 'ENERGY' and len(e.p1.energy) == 4 and not e.p1.remove_area
    e.assert_invariants()

    # Hidden opposing hand/deck identities do not enter an observation.
    e = FastUAEnv(d, d)
    obs, _ = e.reset(seed=7)
    a, b = e.p2.hand[:2]
    a.base, b.base = b.base, a.base
    np.testing.assert_array_equal(obs, e.observation())
    assert obs.dtype == np.float32 and e.observation_space.contains(obs)
    # Ego invariance under relabeling the players.
    for p in e.players:
        p.index = 1-p.index
    for c in e.cards:
        c.owner = 1-c.owner
    e.players.reverse()
    e.turn_player = e.deciding_player = e.pending.player = 1
    np.testing.assert_array_equal(obs, e.observation())
    print('Rule regressions passed')


def real_fixture():
    root = Path(__file__).parent
    e = FastUAEnv(load_deck(root / 'data/bleach_purple.json'),
                  load_deck(root / 'data/ttgl_red.json'), record_replay=True)
    e.reset(seed=73)
    e._flow.close()
    e.pending = None
    for p in e.players:
        for c in list(p.hand + p.life): e.transfer(c, 'DECK')
    e.turn = 2
    return e


def real_regressions():
    from ua_engine import MAX_NAMES
    from plugins.common import peek_sort
    from card_schema import CARD_HOOKS
    e = real_fixture()
    assert len(e.p1.deck) == len(e.p2.deck) == 50
    # Every effectful character has registered behavior; vanilla cards and events are explicit.
    vanilla = {'UA08BT/BLC-1-039', 'UA56BT/TTG-1-042'}
    for base in {c.id:c for deck in e.templates for c in deck}.values():
        assert base.canonical_name and (base.id in vanilla or any(k[0] == base.id for k in CARD_HOOKS))

    # Same-name resource encodings aggregate distinct printings and keep yellow energy separate.
    ids = ['UA08BT/BLC-1-039','UA08BT/BLC-1-040','UA08BT/BLC-1-041','UA08ST/BLC-1-103']
    removed = [get(e,0,id,'REMOVE') for id in ids]
    name = '黑崎 一護'
    assert len({e.card_vector(c)[32] for c in removed}) == 1
    obs = e.observation(0)
    offset = 38 + (e.name_codes[name]-1)*5 + 3
    assert np.isclose(obs[offset], 4/50)
    uryu = get(e,0,'EX15BT/BLC-4-004','HAND')
    other_uryu = get(e,0,'EX15BT/BLC-4-004','HAND')
    e.p1.ap = 1
    assert uryu.base.color == 'YELLOW' and e.p1.energy_for('YELLOW') == 0
    assert e.activation_available(uryu, hand=True)
    resolve(e.hooks(uryu,'HAND_ACTIVATE_MAIN'), lambda d: d.choices[0])
    assert uryu.zone == 'FRONT' and uryu.is_active and e.p1.ap == 0
    assert all(c.zone == 'DROP' for c in removed)
    # Replenishing AP and resources cannot bypass the shared named once-per-turn limit.
    e.p1.ap = 1
    for c in removed: e.transfer(c,'REMOVE')
    assert not e.activation_available(other_uryu, hand=True)
    e.end_turn_cleanup(e.p1)
    assert e.activation_available(other_uryu, hand=True)
    e.assert_invariants()

    # On-being-raided survives source suppression and precedes the top's on-play.
    e = real_fixture()
    source = get(e,0,'UA01DC/BLC-4-083','FRONT')
    raid = get(e,0,'UA08BT/BLC-1-053','HAND')
    def inherited_policy(d):
        return next((c for c in d.choices if c.kind == 'BUFF' or c.card is raid),d.choices[0])
    resolve(e.raid(raid,source),inherited_policy)
    assert raid.unblockable_by_bp_at_least == 4000 and not source.effects_enabled
    events = [(r['event'],r.get('uid')) for r in e.replay_events]
    assert events.index(('ON_BEING_RAIDED',source.uid)) < events.index(('ON_PLAY',raid.uid))
    high = get(e,1,'UA56BT/TTG-1-046','FRONT')
    low = get(e,1,'UA56BT/TTG-1-042','FRONT')
    attack = e.attack(raid)
    decision = next(attack)
    assert decision.kind == 'BLOCK'
    assert high not in [c.card for c in decision.choices] and low in [c.card for c in decision.choices]
    attack.close()
    e.assert_invariants()

    # Custom Raid: both names can be on ANY character's raid sources.
    e = real_fixture(); e.turn_player = 1
    k = get(e,1,'UA56BT/TTG-1-045','FRONT')
    g = get(e,1,'UA56BT/TTG-1-072','HAND')
    simon = get(e,1,'UA56BT/TTG-1-051','ENERGY')
    resolve(e.raid(g,k))
    ace = get(e,1,'UA56BT/TTG-1-073','HAND')
    assert g in e.raid_sources(ace)
    before_hand = len(e.p2.hand)
    resolve(e.raid(ace,g))
    assert g.is_face_down and len(ace.under_cards) == 2 and not g.effects_enabled
    assert ace.get_current_bp(e) == 5000 and ace.get_current_impact(e) == 1
    assert e.p2.raids_this_turn == 2 and len(e.p2.hand) == before_hand  # ace leaves hand, inheritance draws one
    assert sum(r['event']=='ON_BEING_RAIDED' and r.get('uid')==k.uid for r in e.replay_events) == 2
    retired = event_count(e,'ON_RETIRE')
    stopped,_ = resolve(e.leave_field(ace),lambda d:d.choices[-1])
    assert not stopped and ace.zone == 'FRONT' and g.zone == 'DROP'
    assert ace.get_current_bp(e)==4000 and ace.get_current_impact(e)==0
    assert event_count(e,'ON_RETIRE')==retired
    e.assert_invariants()
    # No Kamina anywhere => custom Raid is unavailable, even on a Simon.
    e = real_fixture()
    simon = get(e,1,'UA56BT/TTG-1-051','FRONT')
    ace = get(e,1,'UA56BT/TTG-1-073','HAND')
    assert e.raid_sources(ace)==[]

    # Energy-line attacks enter a real battle; total attack cap survives a source leaving.
    e = real_fixture(); e.turn_player=1
    yoko = get(e,1,'UA56BT/TTG-1-068','ENERGY')
    second = get(e,1,'UA56BT/TTG-1-068','ENERGY')
    get(e,1,'UA56BT/TTG-1-051','HAND')
    blocker = get(e,0,'UA08BT/BLC-1-039','FRONT'); blocker.temp_bp_mod=10000
    e.p2.raids_this_turn=2
    assert yoko.get_current_bp(e)==4000 and e.activation_available(yoko)
    resolve(e.hooks(yoko,'ACTIVATE_MAIN'))
    assert yoko.can_attack_from_energy and e.p2.max_attacks_per_turn==4
    assert not e.activation_available(second)
    for _ in range(4):
        yoko.is_active=blocker.is_active=True
        resolve(e.attack(yoko),lambda d:d.choices[-1])
    assert e.p2.attacks_this_turn==4 and event_count(e,'BATTLE')==4
    yoko.is_active=True
    resolve(e.attack(yoko))
    assert e.p2.attacks_this_turn==4
    resolve(e.leave_field(yoko))
    assert not yoko.can_attack_from_energy and e.p2.max_attacks_per_turn==4
    # Mask also respects the cap, rather than relying only on attack() rejection.
    e._flow=e.game();e._advance()
    e.step(0);e.step(0)
    assert e.phase=='ATTACK' and [c.kind for c in e.pending.choices]==['NEXT']
    e.end_turn_cleanup(e.p2)
    assert e.p2.max_attacks_per_turn is None and e.p2.raids_this_turn==0
    e.assert_invariants()

    # Private deck inspection is encoded for its owner, never for the opponent.
    e = real_fixture()
    flow=peek_sort(e,0,2);dec=next(flow);e.pending=dec;e.deciding_player=0
    before=e.observation(1);own=e.observation(0)
    a,b=e.p1.deck[0],next(c for c in e.p1.deck[2:] if c.base.id!=e.p1.deck[0].base.id)
    a.base,b.base=b.base,a.base
    np.testing.assert_array_equal(before,e.observation(1))
    assert not np.array_equal(own,e.observation(0))
    a.base,b.base=b.base,a.base
    first,second=e.p1.deck[:2]
    dec=flow.send(next(c for c in dec.choices if c.kind=='TOP' and c.card is second))
    try:flow.send(next(c for c in dec.choices if c.kind=='TOP'))
    except StopIteration:pass
    assert e.p1.deck[:2]==[second,first] and not e.private_views[0]
    e.assert_invariants()

    # Mill/banish choices can include zero cards; BP reaching zero retires normally.
    for banish in (False,True):
        e=real_fixture()
        raid=get(e,0,'UA08BT/BLC-1-041','FRONT')
        victim=get(e,1,'UA56BT/TTG-1-051','FRONT')
        get(e,0,'UA08BT/BLC-1-039','DROP')
        resolve(e.hooks(raid,'ON_PLAY'),lambda d:d.choices[-1] if banish else d.choices[0])
        assert victim.zone==('DROP' if banish else 'FRONT')
        assert len(e.p1.remove_area)==(3 if banish else 0)
        e.assert_invariants()

    # Purple COLOR revives with on-play; GET moves the revealed card into hand.
    e=real_fixture()
    color=get(e,0,'UA08BT/BLC-1-040','LIFE')
    get_card=get(e,0,'UA08BT/BLC-1-039','LIFE')
    target=get(e,0,'UA08BT/BLC-1-034','DROP')
    resolve(e.damage(0,1),lambda d:d.choices[-1])
    assert target.zone=='FRONT' and target.is_active and color.zone=='DROP'
    resolve(e.damage(0,1))
    assert get_card.zone=='HAND'
    e.assert_invariants()

    # Rest-to-mill and rest-to-loot consume activity and mask further activations.
    for id in ('UA56BT/TTG-1-053','UA56BT/TTG-1-076'):
        e=real_fixture();c=get(e,1,id,'ENERGY')
        assert e.activation_available(c)
        size=len(e.p2.deck)
        resolve(e.hooks(c,'ACTIVATE_MAIN'))
        assert not c.is_active and not e.activation_available(c) and len(e.p2.deck)==size-1
        e.assert_invariants()

    # AP-recovery events never exceed available AP and do not count as hand discard.
    for owner,id in ((0,'UA08BT/BLC-1-066'),(1,'UA56BT/TTG-1-078')):
        e=real_fixture();c=get(e,owner,id,'HAND');p=e.players[owner]
        p.max_ap=3;p.ap=0
        e.transfer(c,'DROP',track_hand_drop=False)
        resolve(e.hooks(c,'ON_PLAY'),lambda d:d.choices[-1])
        assert p.ap==2 and p.hand_dropped_this_turn==0

    # Shared Kamina reactivation is once per name; separate cards may still choose buff.
    e=real_fixture();e.turn_player=1;e.p2.raids_this_turn=1
    one=get(e,1,'UA56BT/TTG-1-046','FRONT');two=get(e,1,'UA56BT/TTG-1-046','FRONT')
    one.is_active=two.is_active=False
    resolve(e.hooks(one,'ACTIVATE_MAIN'),lambda d:next((c for c in d.choices if c.kind=='ACTIVE'),d.choices[0]))
    assert one.is_active and not e.activation_available(one) and e.activation_available(two)
    flow=e.hooks(two,'ACTIVATE_MAIN');dec=next(flow)
    assert all(c.kind!='ACTIVE' for c in dec.choices)
    flow.close()
    print('Real-deck hooks, canonical observations and attack restrictions passed')


def random_games():
    deck_files = ['data/bleach_purple.json', 'data/ttgl_red.json']
    deck_names = [json.loads((Path(__file__).parent / filename).read_text(
        encoding='utf-8'))['name'] for filename in deck_files]
    deck_a = load_deck(Path(__file__).parent / 'data/bleach_purple.json')
    deck_b = load_deck(Path(__file__).parent / 'data/ttgl_red.json')
    wins = Counter()
    decisions = Counter()
    total_steps = 0
    replay_games = []
    game_results = []
    rng = np.random.default_rng(20261115)
    for episode in range(100):
        env = FastUAEnv(deck_a, deck_b, record_replay=True)
        obs, _ = env.reset(seed=episode)
        while True:
            env.assert_invariants()
            assert env.observation_space.contains(obs) and np.isfinite(obs).all()
            decisions[env.phase] += 1
            actions = np.flatnonzero(env.action_masks())
            assert actions.size
            obs, reward, terminated, truncated, info = env.step(int(rng.choice(actions)))
            total_steps += 1
            if terminated or truncated:
                assert not truncated, f'Unexpected truncation: seed={episode}'
                assert reward in (-1, 1)
                env.assert_invariants()
                wins[info['winner']] += 1
                game_results.append({'seed': episode, 'winner': info['winner'],
                    'winner_deck': deck_names[info['winner']], 'reason': env.reason})
                replay_games.append({'game_number': episode + 1,
                    'winner_deck': deck_names[info['winner']], **env.replay_data()})
                break
        if (episode+1) % 20 == 0:
            print(f'{episode+1}/100 games passed', flush=True)
    # Replay action indices against the exact same deck order and seed.
    replay = replay_games[0]
    copy = FastUAEnv(deck_a, deck_b, record_replay=True)
    copy.reset(seed=replay['seed'])
    for action in replay['actions']:
        copy.step(action)
    assert copy.winner == replay['winner'] and copy.replay_events == replay['events']
    leaders = [player for player in range(2) if wins[player] == max(wins.values())]
    summary = {
        'policy': 'random_legal_actions', 'action_rng_seed': 20261115,
        'total_games': len(game_results), 'total_steps': total_steps, 'truncations': 0,
        'decks': [{'player': player, 'name': name, 'file': deck_files[player],
                   'wins': wins[player], 'win_rate': wins[player] / len(game_results)}
                  for player, name in enumerate(deck_names)],
        'result': 'tie' if len(leaders) > 1 else 'most_wins',
        'most_wins_decks': [deck_names[player] for player in leaders],
        'games': game_results,
    }
    replay = {'schema_version': 4, 'rules_profile': 'user-specified-decks-v3',
              'replay_scope': 'All 100 games are available in games.',
              'decks': [[c.id for c in d] for d in (deck_a, deck_b)],
              'deck_names': deck_names, 'test_summary': summary, 'games': replay_games}
    replay_path = Path('replays/bleach_vs_ttgl.json')
    replay_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding='utf-8')
    saved = json.loads(replay_path.read_text(encoding='utf-8'))
    assert sum(d['wins'] for d in saved['test_summary']['decks']) == len(game_results)
    assert Counter(g['winner'] for g in saved['test_summary']['games']) == wins
    assert len(saved['games']) == 100
    assert all(g['events'] and g['actions'] for g in saved['games'])
    print(f'100 games completed: wins={dict(wins)}, steps={total_steps}, truncations=0')
    print(f'Decision coverage: {dict(decisions)}')
    print('Deterministic replay verified: replays/bleach_vs_ttgl.json')


def training_smoke(steps):
    from sb3_contrib import MaskablePPO
    import torch
    torch.set_num_threads(1)
    a = load_deck(Path(__file__).parent / 'data/bleach_purple.json')
    b = load_deck(Path(__file__).parent / 'data/ttgl_red.json')
    env = SelfPlayEnv(FastUAEnv(a, b))
    model = MaskablePPO('MlpPolicy', env, n_steps=32, batch_size=16, n_epochs=1,
                        policy_kwargs={'net_arch': [32]}, seed=2, verbose=0)
    model.learn(total_timesteps=max(32, steps))
    # Use the very same trained policy for both seats, with a fixed policy during rollout.
    env.opponent = lambda obs, mask: int(model.predict(obs, action_masks=mask, deterministic=True)[0])
    obs, _ = env.reset(seed=8)
    action, _ = model.predict(obs, action_masks=env.action_masks())
    env.step(int(action))
    if steps:
        Path('models').mkdir(exist_ok=True)
        model.save('models/ua_maskable_ppo')
    print('MaskablePPO training and same-model opponent smoke passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-steps', type=int, default=0)
    args = parser.parse_args()
    opening_and_final_regressions()
    regressions()
    checked = FastUAEnv(build_test_deck(), build_test_deck())
    check_reset_seed_determinism(checked)
    env_step_passive_checker(checked, int(np.flatnonzero(checked.action_masks())[0]))
    real_regressions()
    random_games()
    training_smoke(args.train_steps)

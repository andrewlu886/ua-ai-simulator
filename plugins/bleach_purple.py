"""User-specified BLEACH card profile. All named synergies stay in this plugin."""
from card_schema import register_card_hook as hook, Choice
from .common import (field, choose_card, discard, move_selection, mill, peek_sort,
                     target_enemy, buff_ally, ready_ap, loot)

ICHIGO = '黑崎 一護'
RUKIA = '朽木 露琪亞'
URYU = '石田 雨龍'

@hook('EX07BT/BLC-2-054', 'MOD_BP')
def discarded_bonus(card, env):
    return 500 if env.turn_player == card.owner and env.players[card.owner].hand_dropped_this_turn else 0

@hook('EX15BT/BLC-4-004', 'CAN_HAND_ACTIVATE_MAIN')
def can_hand_play(card, env):
    p = env.players[card.owner]
    return p.ap >= 1 and (URYU, 'hand_play') not in p.used_named and sum(
        c.base.canonical_name == ICHIGO for c in p.remove_area) >= 4

@hook('EX15BT/BLC-4-004', 'HAND_ACTIVATE_MAIN')
def hand_play(card, env):
    p = env.players[card.owner]
    p.ap -= 1
    p.used_named.add((URYU, 'hand_play'))
    yield from move_selection(env, card.owner,
        [c for c in p.remove_area if c.base.canonical_name == ICHIGO],
        'DROP', minimum=4, allow_removed=True)
    yield from env.place(card, 'FRONT')
    card.is_active = True
    yield from env.hooks(card, 'ON_PLAY', is_raid=False)

@hook('UA08BT/BLC-1-040', 'ON_PLAY')
@hook('UA01DC/BLC-4-083', 'ON_PLAY')
def look_one(card, env, **kwargs):
    yield from peek_sort(env, card.owner, 1)

@hook('UA08BT/BLC-1-040', 'ON_TRIGGER')
def purple_color(card, env):
    p = env.players[card.owner]
    target = yield from choose_card(env, card.owner,
        [c for c in p.drop if c.base.card_type == 'CHARACTER' and c.base.color == 'PURPLE'
         and c.get_energy_requirement(env) <= 2 and c.base.ap_cost == 1], optional=False)
    if target:
        yield from env.place(target, 'FRONT')
        target.is_active = True
        yield from env.hooks(target, 'ON_PLAY', is_raid=False)
        yield from env.state_checks()

@hook('UA08BT/BLC-1-041', 'ON_PLAY')
def mill_banish_debuff(card, env, **kwargs):
    mill(env, card.owner, 2)
    moved = yield from move_selection(env, card.owner, env.players[card.owner].drop, 'REMOVE')
    target = yield from target_enemy(env, card.owner)
    if target:
        target.temp_bp_mod -= 500 * len(moved)
        yield from env.state_checks()

@hook('UA08BT/BLC-1-053', 'ON_PLAY')
def loot_three(card, env, **kwargs):
    env.draw(card.owner, 3)
    yield from discard(env, card.owner, min(2, len(env.players[card.owner].hand)))

@hook('UA08BT/BLC-1-053', 'CAN_ACTIVATE_MAIN')
def can_buff(card, env):
    return bool(env.players[card.owner].hand) and 'main' not in card.used_once_per_turn

@hook('UA08BT/BLC-1-053', 'ACTIVATE_MAIN')
def buff(card, env):
    card.used_once_per_turn.add('main')
    yield from discard(env, card.owner)
    yield from buff_ally(env, card.owner, 500)

hook('UA08BT/BLC-1-066', 'ON_PLAY')(ready_ap)

@hook('UA08ST/BLC-1-103', 'ON_ATTACK')
def attack_power(card, env, **kwargs):
    card.temp_bp_mod += 1000

@hook('UA08ST/BLC-1-104', 'CAN_ACTIVATE_MAIN')
def can_impact(card, env):
    return len(env.players[card.owner].drop) >= 4 and 'main' not in card.used_once_per_turn

@hook('UA08ST/BLC-1-104', 'ACTIVATE_MAIN')
def banish_impact(card, env):
    card.used_once_per_turn.add('main')
    yield from move_selection(env, card.owner, env.players[card.owner].drop, 'REMOVE', 4, 4)
    env.draw(card.owner)
    card.temp_impact_mod += 1

@hook('UA01DC/BLC-4-082', 'ON_PLAY')
def look_two(card, env, **kwargs):
    dropped = yield from peek_sort(env, card.owner, 2)
    if any(c.get_energy_requirement(env) >= 3 for c in dropped):
        env.draw(card.owner)

@hook('UA01DC/BLC-4-083', 'MOD_ENERGY_REQ')
def empty_field_discount(card, env):
    return -2 if card.zone == 'HAND' and not field(env, card.owner) else 0

@hook('UA01DC/BLC-4-083', 'ON_BEING_RAIDED')
def inherited_choice(card, env, raid_card):
    pick = yield from env.choose(card.owner, 'EFFECT', [Choice('TOP', card), Choice('BUFF', raid_card)])
    if pick.kind == 'TOP':
        yield from peek_sort(env, card.owner, 1)
    else:
        target = yield from choose_card(env, card.owner,
            [c for c in field(env, card.owner) if c.base.canonical_name == RUKIA])
        if target:
            target.unblockable_by_bp_at_least = 4000

hook('UA08BT/BLC-1-034', 'ON_PLAY')(loot)

@hook('UA08BT/BLC-1-064', 'ON_PLAY')
def getsuga(card, env, **kwargs):
    limit = 5000 if any(c.base.canonical_name == ICHIGO for c in field(env, card.owner)) else 3000
    yield from env.effect_remove(card.owner, max_bp=limit, optional=False)

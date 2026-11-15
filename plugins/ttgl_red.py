"""User-specified red Dai-Gurren deck; all title/card-specific rules live here."""
from card_schema import register_card_hook as hook, Choice
from .common import (field, choose_card, discard, mill, choose_number, target_enemy,
                     buff_ally, peek_take_one, ready_ap, loot)

TEAM = '大紅蓮團'
KAMINA = '卡米拿'
SIMON = '施蒙'
YOKO = '陽子'
GURREN = '紅蓮'
LAGANN = '螺巖'
ACE = '紅蓮螺巖'

def team(c):
    return TEAM in c.base.traits

@hook('UA56BT/TTG-1-045', 'ON_BEING_RAIDED')
def kamina_inherit(card, env, raid_card):
    if env.turn_player == card.owner:
        yield from buff_ally(env, card.owner, 1000, team)
        if any(c.is_face_down for c in raid_card.under_cards):
            env.draw(card.owner)

@hook('UA56BT/TTG-1-046', 'CAN_ACTIVATE_MAIN')
def can_kamina(card, env):
    return card.zone == 'FRONT' and env.players[card.owner].raids_this_turn > 0 and 'main' not in card.used_once_per_turn

@hook('UA56BT/TTG-1-046', 'ACTIVATE_MAIN')
def kamina_main(card, env):
    card.used_once_per_turn.add('main')
    p = env.players[card.owner]
    chosen = set()
    for _ in range(2):
        choices = [Choice('PASS')]
        if 'ready' not in chosen and (KAMINA, 'ready') not in p.used_named:
            choices.append(Choice('ACTIVE', card))
        if 'buff' not in chosen:
            choices.append(Choice('BUFF', card))
        pick = yield from env.choose(card.owner, 'EFFECT', choices)
        if pick.kind == 'PASS':
            break
        if pick.kind == 'ACTIVE':
            chosen.add('ready')
            p.used_named.add((KAMINA, 'ready'))
            card.is_active = True
        else:
            chosen.add('buff')
            yield from buff_ally(env, card.owner, 1000, team, optional=False)
            break  # Printed order: activation precedes the buff if both are chosen.

hook('UA56BT/TTG-1-051', 'ON_PLAY')(loot)

@hook('UA56BT/TTG-1-052', 'ON_PLAY')
def salvage(card, env, **kwargs):
    if not (yield from discard(env, card.owner, optional=True)):
        return
    target = yield from choose_card(env, card.owner,
        [c for c in env.players[card.owner].drop if c.base.color == 'RED' and c.get_energy_requirement(env) <= 2])
    if target:
        env.transfer(target, 'HAND')

@hook('UA56BT/TTG-1-053', 'CAN_ACTIVATE_MAIN')
@hook('UA56BT/TTG-1-076', 'CAN_ACTIVATE_MAIN')
def can_rest(card, env):
    return card.is_active

@hook('UA56BT/TTG-1-053', 'ACTIVATE_MAIN')
def rest_mill(card, env):
    card.is_active = False
    mill(env, card.owner, 1)

@hook('UA56BT/TTG-1-055', 'ON_PLAY')
def mill_recover(card, env, **kwargs):
    count = yield from choose_number(env, card.owner, min(2, len(env.players[card.owner].deck)))
    batch = mill(env, card.owner, count)
    if batch and (yield from discard(env, card.owner, optional=True)):
        target = yield from choose_card(env, card.owner, [c for c in batch if c.zone == 'DROP'])
        if target:
            env.transfer(target, 'HAND')

@hook('UA56BT/TTG-1-068', 'MOD_BP')
def yoko_power(card, env):
    return 1000 * env.players[card.owner].raids_this_turn if env.turn_player == card.owner else 0

@hook('UA56BT/TTG-1-068', 'CAN_ACTIVATE_MAIN')
def can_yoko(card, env):
    p = env.players[card.owner]
    return card.zone == 'ENERGY' and bool(p.hand) and (YOKO, 'energy_attack') not in p.used_named

@hook('UA56BT/TTG-1-068', 'ACTIVATE_MAIN')
def yoko_main(card, env):
    p = env.players[card.owner]
    p.used_named.add((YOKO, 'energy_attack'))
    yield from discard(env, card.owner)
    card.can_attack_from_energy = True
    p.max_attacks_per_turn = min(p.max_attacks_per_turn, 4) if p.max_attacks_per_turn is not None else 4

@hook('UA56BT/TTG-1-072', 'MOD_BP')
def gurren_power(card, env):
    return 500 if env.turn_player == card.owner else 0

@hook('UA56BT/TTG-1-072', 'ON_PLAY')
def draw_rest(card, env, **kwargs):
    env.draw(card.owner)
    target = yield from target_enemy(env, card.owner, 3000)
    if target:
        target.is_active = False

@hook('UA56BT/TTG-1-073', 'CAN_RAID_ON')
def ace_raid(card, env, source, default):
    names = {c.base.canonical_name for c in field(env, card.owner)}
    names.update(c.base.canonical_name for top in field(env, card.owner)
                 for c in top.under_cards if not c.is_face_down)
    has_both = {SIMON, KAMINA} <= names
    return has_both and (source.base.canonical_name in (SIMON, KAMINA) or
                        source.is_raided and source.base.canonical_name in (LAGANN, GURREN))

@hook('UA56BT/TTG-1-073', 'RAID_INHERIT_SOURCES')
def inherited_sources(card, env, source):
    if source.is_raided and source.base.canonical_name in (LAGANN, GURREN):
        return [c for c in source.under_cards if not c.is_face_down
                and c.base.canonical_name in (SIMON, KAMINA)]
    return [source]

@hook('UA56BT/TTG-1-073', 'ON_RAID_STACK')
def fold_mecha(card, env, source):
    # Choosing this alternative raid source commits to folding its top card.
    if source.is_raided and source.base.canonical_name in (LAGANN, GURREN):
        source.is_face_down = True
        card.under_cards.remove(source)
        card.under_cards.insert(0, source)

@hook('UA56BT/TTG-1-073', 'MOD_BP')
def shield_power(card, env):
    return 1000 if any(c.is_face_down for c in card.under_cards) else 0

@hook('UA56BT/TTG-1-073', 'MOD_IMPACT')
def shield_impact(card, env):
    return 1 if any(c.is_face_down for c in card.under_cards) else 0

@hook('UA56BT/TTG-1-073', 'ON_PLAY')
def ace_remove(card, env, **kwargs):
    if any(c.is_face_down for c in card.under_cards):
        yield from env.effect_remove(card.owner, max_bp=5000)

@hook('UA56BT/TTG-1-073', 'ON_BEFORE_LEAVE_FIELD')
def shield_replace(card, env, **kwargs):
    if card.get_current_bp(env) < 500:
        return False
    target = yield from choose_card(env, card.owner, [c for c in card.under_cards if c.is_face_down])
    if target:
        env.detach_to_drop(card, target)
        return True
    return False

@hook('UA56BT/TTG-1-076', 'ON_PLAY')
def lagann_look(card, env, **kwargs):
    yield from peek_take_one(env, card.owner, 2)

@hook('UA56BT/TTG-1-076', 'ACTIVATE_MAIN')
def lagann_main(card, env):
    card.is_active = False
    yield from loot(card, env)

hook('UA56BT/TTG-1-078', 'ON_PLAY')(ready_ap)

@hook('UA56BT/TTG-1-081', 'ON_PLAY')
def giga_drill(card, env, **kwargs):
    named = [c for top in field(env, card.owner) for c in [top] + top.under_cards if not c.is_face_down]
    limit = 5000 if any(c.base.canonical_name in (SIMON, KAMINA) for c in named) else 3000
    yield from env.effect_remove(card.owner, max_bp=limit, optional=False)

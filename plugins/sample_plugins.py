"""Synthetic cards demonstrating hooks; no card-specific code belongs in the engine."""
from card_schema import Choice, Protection, Resistance, register_card_hook

@register_card_hook('SNIPE_DRAW', 'ON_UNBLOCKED')
def snipe_draw(card, env, **kwargs):
    env.draw(card.owner)

@register_card_hook('AURA', 'PROTECT_ALLY')
def aura(card, env, target, **kwargs):
    return [Protection(Resistance.IMMUNE)] if target.owner == card.owner else []

@register_card_hook('SHIELD', 'ON_PLAY')
def prepare_shields(card, env, **kwargs):
    for source in card.under_cards:
        source.is_face_down = True

@register_card_hook('SHIELD', 'ON_BEFORE_LEAVE_FIELD')
def substitute(card, env, **kwargs):
    sources = [c for c in card.under_cards if c.is_face_down]
    if not sources:
        return False
    choice = yield from env.choose(card.owner, 'SHIELD', [Choice('PASS')] + [Choice('SHIELD', c) for c in sources])
    if choice.kind == 'PASS':
        return False
    env.detach_to_drop(card, choice.card)
    return True

@register_card_hook('DETACH', 'ON_END_ATTACK_PHASE')
def detach(card, env, **kwargs):
    if not card.under_cards:
        return
    choice = yield from env.choose(card.owner, 'DETACH', [Choice('PASS')] + [Choice('DETACH', c) for c in card.under_cards])
    if choice.kind != 'PASS':
        source = choice.card
        yield from env.place(source, 'ENERGY')

@register_card_hook('DISCOUNT', 'MOD_ENERGY_REQ')
def discount(card, env, **kwargs):
    return -len(env.players[card.owner].hand) // 3

@register_card_hook('GROW', 'MOD_BP')
def stack_power(card, env, **kwargs):
    return 500 * len(card.under_cards)

@register_card_hook('EVENT', 'ON_PLAY')
def removal(card, env, **kwargs):
    yield from env.effect_remove(card.owner, max_bp=3500)

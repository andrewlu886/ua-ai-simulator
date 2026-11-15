"""Reusable plugin effect building blocks; every strategic selection is yielded."""
from card_schema import Choice


def field(env, owner):
    p = env.players[owner]
    return p.front + p.energy


def choose_card(env, owner, cards, optional=True, kind='TARGET'):
    cards = list(cards)
    choices = ([Choice('PASS')] if optional else []) + [Choice(kind, c) for c in cards]
    if not choices:
        return None
    result = yield from env.choose(owner, 'EFFECT', choices)
    return result.card


def discard(env, owner, count=1, optional=False):
    if len(env.players[owner].hand) < count:
        return False
    for i in range(count):
        card = yield from choose_card(env, owner, env.players[owner].hand,
                                      optional=optional and i == 0, kind='DISCARD')
        if card is None:
            return False
        env.transfer(card, 'DROP')
    return True


def move_selection(env, owner, cards, dest, minimum=0, maximum=None, allow_removed=False):
    remaining = list(cards)
    chosen = []
    maximum = len(remaining) if maximum is None else maximum
    while remaining and len(chosen) < maximum:
        card = yield from choose_card(env, owner, remaining, optional=len(chosen) >= minimum)
        if card is None:
            break
        remaining.remove(card)
        chosen.append(card)
        env.transfer(card, dest, allow_removed_to_drop=allow_removed)
    return chosen


def mill(env, owner, count):
    cards = list(env.players[owner].deck[:count])
    for card in cards:
        env.transfer(card, 'DROP')
    return cards


def choose_number(env, owner, maximum):
    pick = yield from env.choose(owner, 'EFFECT', [Choice('COUNT', amount=i) for i in range(maximum + 1)])
    return pick.amount


def peek_sort(env, owner, count):
    """Choose next card and TOP/DROP; TOP choices establish final top-to-bottom order."""
    p = env.players[owner]
    viewed = list(p.deck[:count])
    env.private_views[owner] = list(viewed)
    kept, dropped = [], []
    try:
        while viewed:
            choices = [Choice(kind, c) for c in viewed for kind in ('TOP', 'DROP')]
            pick = yield from env.choose(owner, 'PEEK', choices)
            viewed.remove(pick.card)
            if pick.kind == 'DROP':
                env.transfer(pick.card, 'DROP')
                dropped.append(pick.card)
            else:
                kept.append(pick.card)
        for card in kept:
            p.deck.remove(card)
        p.deck[:0] = kept
    finally:
        env.private_views[owner] = []
    return dropped


def peek_take_one(env, owner, count):
    viewed = list(env.players[owner].deck[:count])
    if not viewed:
        return
    env.private_views[owner] = viewed
    try:
        card = yield from choose_card(env, owner, viewed, optional=False, kind='HAND')
        for c in viewed:
            env.transfer(c, 'HAND' if c is card else 'DROP')
    finally:
        env.private_views[owner] = []


def target_enemy(env, owner, max_bp=float('inf')):
    cards = [c for c in env.players[1-owner].front
             if c.base.card_type == 'CHARACTER' and c.get_current_bp(env) <= max_bp
             and env.target_cost(c) is not None]
    card = yield from choose_card(env, owner, cards)
    if card is not None and (yield from env.pay_target(owner, card)):
        return card
    return None


def buff_ally(env, owner, amount, predicate=lambda c: True, optional=True):
    cards = [c for c in field(env, owner) if c.base.card_type == 'CHARACTER' and predicate(c)]
    card = yield from choose_card(env, owner, cards, optional=optional)
    if card:
        card.temp_bp_mod += amount
    return card


def ready_ap(card, env, **kwargs):
    p = env.players[card.owner]
    count = yield from choose_number(env, card.owner, min(2, p.max_ap - p.ap))
    p.ap += count


def loot(card, env, **kwargs):
    env.draw(card.owner)
    yield from discard(env, card.owner)

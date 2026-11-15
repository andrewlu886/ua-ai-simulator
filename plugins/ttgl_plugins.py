@register_card_hook("TTGL_GURREN_LAGANN", "MOD_BP")
def gurren_bp_boost(self_card, state, **kwargs):
    # 底下有背面向上的卡時 BP+1000
    face_down_count = sum(1 for c in self_card.under_cards if c.is_face_down)
    return 1000 if face_down_count > 0 else 0

@register_card_hook("TTGL_GURREN_LAGANN", "MOD_IMPACT")
def gurren_impact_boost(self_card, state, **kwargs):
    face_down_count = sum(1 for c in self_card.under_cards if c.is_face_down)
    return 1 if face_down_count > 0 else 0

@register_card_hook("TTGL_GURREN_LAGANN", "ON_BEFORE_LEAVE_FIELD")
def gurren_substitute(self_card, state, **kwargs):
    # 規則：BP500或以上離開場上時，可改為將下面1張背面向上的卡放場外
    if self_card.get_current_bp(state) >= 500:
        face_down_cards = [c for c in self_card.under_cards if c.is_face_down]
        if face_down_cards:
            sacrificed = face_down_cards.pop(0)
            self_card.under_cards.remove(sacrificed)
            sacrificed.zone = "DROP"
            state.get_owner(self_card).drop.append(sacrificed)
            return True  # 成功替代離場！
    return False
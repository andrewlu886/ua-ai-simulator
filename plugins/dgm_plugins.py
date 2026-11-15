from card_schema import register_card_hook

# 1. 手牌6張以上減3費
@register_card_hook("DGM_HAND_REDUCE", "CHECK_ENERGY_REQ")
def hand_reduce_energy(self_card, state, **kwargs):
    p = state.active_player
    other_hand = len([c for c in p.hand if c.uid != self_card.uid])
    if other_hand >= 6:
        return max(0, self_card.base.energy_req - 3)
    return None

# 2. 突襲〈艾倫・獲加〉：在前線時，其他原BP3500以下免疫效果選擇（Snipe可繞過）
@register_card_hook("DGM_ALLEN_RAID", "PASSIVE_TARGET_IMMUNITY")
def allen_target_immunity(self_card, state, target_card=None, **kwargs):
    if self_card.zone == "FRONT" and target_card is not None and target_card.uid != self_card.uid:
        if target_card.base.bp <= 3500:
            return True
    return False

# 3. 突襲〈雷亞・D・坎貝爾〉：脫離突襲源後變傷害2，攻擊結束封鎖對手
@register_card_hook("DGM_TYKI_RAID", "OVERRIDE_DAMAGE")
def tyki_damage_two(self_card, state, **kwargs):
    return 2 if not self_card.is_raided else 1

@register_card_hook("DGM_TYKI_RAID", "ON_END_ATTACK_PHASE")
def tyki_detach_and_lock(self_card, state, **kwargs):
    if self_card.is_raided:
        self_card.is_raided = False
        if self_card.raid_source:
            src = self_card.raid_source
            src.zone = "DROP"
            state.active_player.drop.append(src)
            self_card.raid_source = None
        state.pending_output.disable_attack_count += 1

# 4. 狙擊測試卡：沒有被阻擋時抽1張；戰敗時自身激活
@register_card_hook("TEST_SNIPE_DRAW", "ON_UNBLOCKED")
def snipe_unblocked_draw(self_card, state, **kwargs):
    state.active_player.draw(1)

@register_card_hook("TEST_SNIPE_DRAW", "ON_BATTLE_LOSE")
def snipe_lose_activate(self_card, state, **kwargs):
    self_card.is_active = True
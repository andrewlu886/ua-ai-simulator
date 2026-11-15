# UA Headless 模擬平台開發最高準則

本專案以使用者指定的 Ver 1.1 規則設定為驗收契約；若與地區版官方裁定不同，須記錄差異，不可暗中更改。此檔為專案開發最高準則（仍受上層指令約束）。

## 交付結構與順序
依序實作 AGENTS.md、requirements.txt、card_schema.py、plugins/__init__.py、plugins/sample_plugins.py、ua_engine.py、test_run.py；保留 replays/.gitkeep。依賴 gymnasium>=0.29.0、numpy>=1.24.0、sb3-contrib>=2.2.0、torch>=2.0.0。

## 資料與架構
- 純數字、無 UI 的 Gymnasium 環境。核心不得包含單卡 ID／作品判斷。特殊效果只由 plugins 下的 @register_card_hook(card_id, hook_name) 註冊。
- BP、Damage、Impact、能源需求每次查詢加總基礎、臨時修正與外掛結果。
- CardInstance 維護多層 under_cards 與 is_face_down；底卡效果無效。
- 所有博弈選擇（含阻擋、抗性費用、棄哪張牌、Raid Trigger、滿場移除、Trigger 順序與目標）必須暫停交給 deciding_player，不得用威脅分數或啟發式代選。
- observation 為固定維度 np.float32、永遠以 deciding_player 為自身視角；逐格公開雙方前線、能源線、BP、抗性、底卡層數。隱藏手牌與牌庫不得洩漏。action_masks 支援 MaskablePPO，同模型可決策雙方。

## 七區與資源
- DECK/HAND/FRONT/ENERGY/LIFE/DROP/REMOVE 嚴格分離。前線與能源線各最多4，開局7生命；DROP可回收，REMOVE不可使用。
- 登場或移動至滿場先選既有卡移至REMOVE。結束手牌超過8，選多餘牌進REMOVE。
- 先攻第一回合1AP、不抽牌、不可額外抽；後攻第一回合2AP抽1；先攻第二回合2AP抽1；往後3AP。每回合最多一次支付1AP額外抽1。
- 能源只檢查門檻不消耗。移動階段能源至前線；Step可前後移動與交換位置且交換不移除。

## Raid 與離場
- 順序為重疊元體、元體效果無效、活動、能源線可選擇移前線、ON_PLAY。
- ON_BEFORE_LEAVE_FIELD 回傳True時取消離場；例：丟1張背面底卡替死。
- 真正進DROP的退場／戰敗才觸發ON_RETIRE。回手／移除只移動頂卡，所有底卡進DROP且不觸發ON_RETIRE；底卡本身不作場上角色處理。離場重置場上狀態與每回合一次旗標。

## 戰鬥
- ON_ATTACK、ON_BLOCK、Battle 三層獨立。狙擊可選對手任何前線，穿透效果IMMUNE；不改變目標活動狀態、不觸發ON_BLOCK；必觸發ON_UNBLOCKED，再比較BP，攻擊方較低觸發ON_BATTLE_LOSE。
- 支援Damage X、Impact X（僅有阻擋且戰勝）、Impact Nullify、Double Attack、Double Block。
- IMMUNE防效果指定；TAX_HAND_1/TAX_HAND_2/TAX_AP_1，資料標記是否適用狙擊。支付或放棄交給進攻者，手牌代價逐張選擇。

## 傷害與 Trigger
- 多傷害一次批次翻開所有對應生命，防守方任意選順序逐張解決；一般進DROP。
- DRAW抽1；ACTIVE選角色活動且本回合BP+3000；COLOR依資料顏色／BP限制除去對手前線；SPECIAL除去對手前線；FINAL結算時Life為0則從牌庫頂補1張背面Life且不確認Trigger，否則該FINAL入手；不增加AP；RAID能源符合且有源時可防守中即時突襲並ON_PLAY，否則入手。

## 驗證
- test_run.py建立兩副各50張測試牌組，涵蓋展開、狙擊未阻擋抽牌、防指定光環、多層替死、全部Trigger。
- 執行 python test_run.py，100場依action_masks隨機合法對戰，零錯誤並產生 replays/sample_replay.json。
- 加入關鍵規則回歸、卡片守恆、觀察維度／隱藏資訊、確定性重播檢查。不得把步數截斷當勝利。
- 保留舊版未使用外掛檔，不自動載入不符合新契約的模組。

## 後續規則修正（優先於原始設定）
- 結束階段清理臨時修正後，將當前回合玩家前線及能源線的角色／場域置為活動；AP不重置，對手卡片不因此活動。
- reset抽7張後、放Life前，依序讓雙方各做一次MULLIGAN：0保留，1原手牌放旁、抽7張後將原手牌洗回牌庫；兩人都決定後才放7張Life。使用環境RNG並納入遮罩、觀察與Replay。

## 實戰牌組擴充
- data/bleach_purple.json 與 data/ttgl_red.json 各50張；保留指定槽位及張數，核對官方卡號與顏色，記錄原編號對照及來源，不能以虛構別名冒充官方卡號。
- CardBase.canonical_name 與 traits 為通用資料；觀察加入本名、特徵及雙方各本名公開區資源計數。不得在引擎判斷特定人名。
- CAN_RAID_ON、ON_BEING_RAIDED、RAID_INHERIT_SOURCES、ON_RAID_STACK、場上與手牌主起動、GET、ON_TRIGGER 皆以通用hook提供；特殊效果僅放在外掛。
- 能源線攻擊權限屬於個體；攻擊上限、突襲計數及同名一次限制屬於玩家，並於回合結束清除。
- REMOVE預設禁止重用，但明示卡片效果可透過 allow_removed_to_drop=True 僅進行REMOVE至DROP。
- test_run.py 執行既有及新牌組回歸、100場BLEACH對天元突破與PPO smoke test，輸出replays/bleach_vs_ttgl.json，檢查逐步守恆與確定性重播。

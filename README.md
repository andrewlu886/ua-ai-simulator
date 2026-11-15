# UA Headless 自我對弈平台

Python / Gymnasium 純數字環境。核心提供通用規則及可中斷決策；卡片效果全部由 `plugins/` 註冊。開發準則見 [AGENTS.md](AGENTS.md)。

## 執行與驗證

```powershell
.\.venv\Scripts\Activate.ps1
python test_run.py
# 額外訓練並保存 models/ua_maskable_ppo.zip
python test_run.py --train-steps 4096
```

初次安裝：`python -m venv .venv`，再用虛擬環境執行 `python -m pip install -r requirements.txt`。依賴已安裝在本專案 `.venv`。

預設執行原有規則回歸、新牌組效果回歸、Gymnasium reset/step 檢查、兩副牌的 100 場固定種子隨機對戰、逐步卡片守恆、觀察空間驗證、完整重播一致性及 MaskablePPO 訓練 smoke test。輸出 `replays/bleach_vs_ttgl.json`。隨機勝率與短程訓練只驗證管線，不能證明牌組強度或模型對戰能力。

## 牌組與官方編號校正

- `data/bleach_purple.json`：紫死神 50 張，混入 2 張黃色 `EX15BT/BLC-4-004` 雨龍。普通登場仍需黃色能源；其手牌主起動按效果支付 AP 與移除區的一護，無須黃色門檻。
- `data/ttgl_red.json`：紅大紅蓮團 50 張。
- `data/card_id_corrections.json`：使用者原始編號 → 效果對應的官方編號。兩份牌組 JSON 同時記錄校正、官方來源及核對日期。
- 兩副牌保留指定的 13 個角色／效果槽位與各槽張數；每副共 50 張，同編號最多 4 張，COLOR／FINAL／SPECIAL 各不超過 4 張。沒有以自創 `_ICHIGO` 編號冒充正式卡片。

部分重要校正如下；完整對照見 JSON：

| 原清單 | 實作編號 | 卡片 |
|---|---|---|
| UA08BT/BLC-1-040_ICHIGO | UA08BT/BLC-1-039 | 0 費一護 |
| UA08BT/BLC-1-042 | UA08BT/BLC-1-040 | 1 費一護 |
| UA08BT/BLC-1-040 | UA08BT/BLC-1-034 | 0 費雨龍 |
| UA08BT/BLC-1-044 | EX07BT/BLC-2-054 | 手牌放場外時強化的露琪亞 |
| UA08BT/BLC-1-047 | UA08BT/BLC-1-053 | 突襲露琪亞 |
| UA08BT/BLC-1-045 | UA08ST/BLC-1-103 | 攻擊時強化的一護 |
| EX07BT/BLC-2-002 | UA08ST/BLC-1-104 | 場外移除 4 張的一護 |
| UA08BT/BLC-1-046 | UA01DC/BLC-4-082 | 看牌庫頂 2 張的一護 |
| UA08BT/BLC-1-068 / 066 | UA08BT/BLC-1-066 / 064 | 卍解／月牙天衝 |
| TTG-1-043 / 056 / 052 | TTG-1-042 / 052 / 053 | 無效果卡米拿／回收施蒙／橫置施蒙 |
| TTG-1-071 / 060 / 077 | TTG-1-072 / 076 / 081 | 紅蓮／螺巖／巨鑽爆擊 |

官方資料亦確認：事件不產能源；回收施蒙 `TTG-1-052` 產 1 能源；巨鑽爆擊的 5000 門檻條件是場上或突襲源存在施蒙或卡米拿（以 FAQ 的前線目標勘誤為準）；紅蓮螺巖的登場除去也必須先持有背面底卡。

來源：[官方卡表](https://www.unionarena-tcg.com/jp/cardlist/)、[黃色雨龍](https://www.unionarena-tcg.com/tc/cardlist/detail_iframe.php?card_no=EX15BT%2FBLC-4-004)、[紅蓮螺巖](https://www.unionarena-tcg.com/jp/cardlist/detail_iframe.php?card_no=UA56BT%2FTTG-1-073)、[天元突破 FAQ](https://www.unionarena-tcg.com/jp/faq/list.php?series=UA56BT)。逐卡來源也保存在牌組 JSON。

這是校正卡號後的指定構築，不宣稱「環境最強」或已通過完整賽事認證。仍保留先前指定的規則設定，例如 Impact 僅在宣告阻擋且戰勝時造成傷害。尚未實作完整官方卡池、所有地區裁定、所有禁止／限制表。被突襲與登場效果目前依「元體 → 頂卡」固定順序結算；可選結算順序是後續擴充範圍。

## 開局、結束與 Trigger

`reset()` 先回傳先攻的 MULLIGAN，後攻隨後選擇；`0` 保留、`1` 原手牌放旁，抽 7 張後將原手牌洗回。兩人皆決定後才放 7 張背面 Life。

結束階段清除本回合效果與限制，將當前玩家前線／能源線的角色與場域活動，AP 不變。FINAL 在 Life 為 0 時從牌庫頂補 1 張背面 Life，不再確認 Trigger；Life 非 0 則該 FINAL 入手。GET 直接將該 Trigger 卡入手。批次 Trigger 全部結算後 Life 仍為 0 才判負。空牌庫在開始階段無法抽牌時判負。

## 觀察與動作契約（v3）

`FastUAEnv` 每次 `step` 只處理一個選擇並運行到下一個微決策。`deciding_player` 隨阻擋、Trigger 等切換。非法動作拋出 `ValueError`；沒有威脅評分或自動替代理挑目標。

- 動作：`Discrete(512)`，`action_masks()` 為固定 bool 向量。動作索引依當前選項解讀，觀察附帶每個動作的類型、卡片、目標、區域、背面標記及數量。
- 觀察：33,162 維 `np.float32`，永遠由當前決策者出發。38 維回合／資源資訊 + 640 維本名資源統計 + 425 個卡片格 × 68 維 + 512 個選項 × 7 維。
- 每張卡有 `canonical_name` 的群組編碼及最多 32 個 trait 的多熱編碼。最多 64 個本名，每個本名對雙方 FRONT／ENERGY／DROP／REMOVE 分別計數，另記錄同名每回合限制使用狀態。可直接學習「移除區的一護」與雨龍的關係，核心不寫死該本名。
- 回合特徵包含突襲次數、合計攻擊次數、攻擊上限、手牌放場外次數。卡片特徵包含能源線攻擊權限及阻擋 BP 限制。
- 公開場面、場外、移除區、已翻 Life、底卡層數逐格編碼。牌庫頂檢視只對擁有者提供 50 格私有檢視槽；對手看不到。自己可以確認自己的背面底卡，對手只能看到存在。
- 詞彙表依兩副牌的卡 ID、本名與 traits 排序建立；交換雙方座位不改變詞彙表。共用訓練模型時須維持相同卡池。v3 維度與編碼已改變，舊模型需重新訓練。

```python
from pathlib import Path
import numpy as np
from card_schema import load_deck
from ua_engine import FastUAEnv

env = FastUAEnv(load_deck('data/bleach_purple.json'), load_deck('data/ttgl_red.json'))
obs, info = env.reset(seed=42)
while True:
    action = int(np.random.choice(np.flatnonzero(env.action_masks())))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

非終局 reward 為 0；終局 reward 屬於本次 `info['acting_player']`，勝 +1、敗 -1。`info['winner']` 為實際勝者。步數上限僅產生 truncated，不指定勝者。

PPO 請使用 `SelfPlayEnv` 固定學習者座位；勿直接將雙方交替觀察輸入單一代理 PPO 的折扣回報。adapter 的 `opponent(obs, mask)` 可使用同模型或歷史策略快照。

## 通用引擎與外掛

`plugins/__init__.py` 載入 `sample_plugins`、`bleach_purple`、`ttgl_red`；舊版實驗外掛保留但不載入。`plugins/common.py` 提供可中斷的棄牌、選卡、看牌排序、抽棄、AP 恢復等效果工具。

- `MOD_BP`／`MOD_DAMAGE`／`MOD_IMPACT`／`MOD_ENERGY_REQ` 為純查詢加算鉤子。
- `CAN_ACTIVATE_MAIN`／`ACTIVATE_MAIN` 供場上主起動；`CAN_HAND_ACTIVATE_MAIN`／`HAND_ACTIVATE_MAIN` 供手牌主起動。前者為無副作用的合法性判定，後者可 yield 多個決策。手牌起動不套用普通登場能源門檻。
- `CAN_RAID_ON` 覆寫特定突襲來源判定。`ON_RAID_STACK` 處理堆疊建立，`RAID_INHERIT_SOURCES` 指定真正被突襲的元體。底卡平時停用效果；僅 `ON_BEING_RAIDED` 在指定事件被顯式執行，隨後執行頂卡 `ON_PLAY`。
- `CardInstance.can_attack_from_energy` 是本回合權限；`PlayerState.max_attacks_per_turn=None` 表示不限次數。攻擊計數屬於玩家，換角色、離場、再次活動都不能繞過上限；攻擊取消指定前不消耗次數。
- 個體每回合一次使用卡片的 `used_once_per_turn`；同名共用限制使用玩家 `used_named`。結束階段統一清理。移除卡本身不能清除玩家的同名限制。
- `ON_TRIGGER` 讓 COLOR 等卡文效果由外掛解決；無外掛時採通用 Trigger。紫 COLOR 蘇生會正常發動被蘇生卡的登場效果。
- `state_checks()` 讓 BP 降到 0 的角色走正常退場流程。紅蓮螺巖低於 500 BP 時不能使用其替死資源。
- `transfer()` 預設禁止重用 REMOVE。只有明確效果可傳 `allow_removed_to_drop=True`，而且只准 REMOVE → DROP；雨龍透過此受限介面支付效果代價。不可將 REMOVE 當普通回收區。
- 正式離場使用 `yield from env.leave_field(...)`；不可先從場上列表刪除。牌庫檢視不改變卡片所屬區，所有微決策點均檢查 100 張實體卡唯一歸屬。

## Replay

### 卡圖與播放器

執行 `.venv\Scripts\python.exe download_cards.py` 下載目前兩副牌組的 26 種卡。工具只讀取指定牌組的官方卡片頁，依繁中、日文、英文順序尋找同一卡號；每筆索引保留官方名稱、牌組中文名稱、卡號、圖片語言與官方來源。圖片存於 `assets/cards/`，索引為 `data/card_catalog.json` 與支援直接開啟 HTML 的 `data/card_catalog.js`。已有頁面與圖片會快取，可中斷後重跑；`--refresh` 重新下載，`--decks data/其他牌組.json` 可指定牌組。需要系統 curl，保留 TLS 驗證並限制請求頻率。

開啟 `visualizer.html`，選擇 `replays/bleach_vs_ttgl.json`，可從場次選單切換全部 100 場；手牌及場上會顯示卡圖、中文名與卡號，點卡圖放大。按住「下一步」超過 0.35 秒會每 70ms 連續前進。也可執行 `.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1`，開啟 `http://127.0.0.1:8765/visualizer.html`，按「載入最新測試重播」。缺少卡圖時仍保留名稱及卡號。卡圖來自官方網站，著作權屬原權利人；資料下載不會更改引擎或 AI 觀察。

播放器仍是根據事件推導的原型，沒有完整還原所有外掛造成的 AP、BP、活動狀態；卡圖顯示不代表盤面狀態已完整重建。

單場匯出維持 `schema_version=3`。100 場測試輸出採 `schema_version=4`，共用兩副牌的 ID 順序，`games` 內保存每場種子、勝者、原因、動作序列及逐事件日誌。需相同卡片資料／外掛版本，使用 reset(seed) 後逐項執行 actions 才可重建；測試逐筆比較首場日誌完全一致。完整日誌只供離線分析，不作為代理 observation。舊版 `sample_replay.json` 保留供歷史參考，新的主要輸出是 `bleach_vs_ttgl.json`。

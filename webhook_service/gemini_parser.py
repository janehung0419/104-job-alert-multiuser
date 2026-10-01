"""呼叫 Google Gemini，把使用者在 LINE 上輸入的自然語言解析成結構化的職缺訂閱條件。

使用新版統一 SDK `google-genai`（舊版 `google-generativeai` 已於 2025-08-31 停止維護）。
預設 model 用 `gemini-flash-latest` 這個別名，Google 會自動把它指向當前最新的
Flash 版本（不是寫死某個具體版號）——具體版號的 model（例如 gemini-2.0-flash、
gemini-2.5-flash-lite）會不定期被下架，用別名可以避免每隔幾個月就要改一次程式碼。
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import errors, types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.area_codes import SUPPORTED_CITY_NAMES  # noqa: E402
from common.mrt_lines import ALL_STATIONS, stations_between  # noqa: E402

# Render 上 GEMINI_MODEL 留空時 os.environ.get 會拿到空字串，用 or 才會退回預設值
GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-flash-latest"

_client = None


def _get_client():
    """第一次用到時才建立 Gemini client。

    google-genai 沒有 API key 時建立 client 會直接丟 ValueError；如果在 import 時就建立，
    漏設 GEMINI_API_KEY 會讓 gunicorn 整個啟動失敗（Render 只顯示 Exited with status 1），
    連健康檢查、取消訂閱這些不需要 Gemini 的功能都無法使用。
    """
    global _client
    if _client is None:
        # google-genai 官方文件也接受 GOOGLE_API_KEY，兩個名稱都認，避免設錯名稱就整個無法使用
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("環境變數 GEMINI_API_KEY 未設定，無法呼叫 Gemini")
        _client = genai.Client(api_key=api_key)
    return _client

_AREA_ENUM = SUPPORTED_CITY_NAMES

# 每分鐘走路概估距離（公尺），用來把「步行 X 分鐘」換算成公里數門檻
_WALK_KM_PER_MINUTE = 0.08

# 通知頻率的合理範圍（小時）；GitHub Actions 排程本身是每小時跑一次，
# 所以 1 小時是能做到的最高頻率，24 小時（一天一次）是上限
_MIN_NOTIFY_INTERVAL_HOURS = 1
_MAX_NOTIFY_INTERVAL_HOURS = 24

# 「每次最多通知幾筆」的合理範圍；預設 5 筆
_MIN_MAX_JOBS_PER_RUN = 1
_MAX_MAX_JOBS_PER_RUN = 20
_DEFAULT_MAX_JOBS_PER_RUN = 5

# Gemini 偶爾會回傳 503（暫時過載），重試個幾次通常就能成功；重試次數用完仍失敗
# 就把例外往外拋，讓呼叫端（app.py）知道這是「服務暫時忙碌」而不是「看不懂使用者的話」，
# 兩者要回覆不同的訊息給使用者
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 1.5

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {"type": "STRING", "enum": ["subscribe_or_update", "unsubscribe", "unclear"]},
        "keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
        "areas": {"type": "ARRAY", "items": {"type": "STRING", "enum": _AREA_ENUM}},
        "min_annual_salary": {"type": "INTEGER"},
        "mrt_start_station": {"type": "STRING", "enum": ALL_STATIONS},
        "mrt_end_station": {"type": "STRING", "enum": ALL_STATIONS},
        "max_walk_minutes": {"type": "INTEGER"},
        "include_remote": {"type": "BOOLEAN"},
        "notify_interval_hours": {"type": "INTEGER"},
        "max_jobs_per_run": {"type": "INTEGER"},
    },
    "required": ["intent"],
}

SYSTEM_INSTRUCTION = f"""你是一個求職職缺通知機器人的訊息解析器。使用者會用自然語言描述他想找的工作，
你要把它轉換成結構化的訂閱條件。

規則：
- 只有使用者明確表示要取消/退訂通知時，才把 intent 設為 "unsubscribe"。
- 如果訊息裡完全找不到任何職稱/職務相關的關鍵字，也沒有提到要調整下面任何一項設定
  （地區、遠端與否、最低年薪、捷運通勤範圍、通知頻率、每次通知筆數），才把 intent 設為 "unclear"。
  只要訊息裡有明確提到「其中任何一項」要調整，就不算 unclear，即使沒有提到職稱關鍵字也一樣。
- 其他情況一律設為 "subscribe_or_update"。
- keywords 是使用者想找的職稱關鍵字列表（例如「前端工程師」「後端工程師」），盡量精簡、每個是一個職稱。
  如果這則訊息只是要調整其他設定、沒有提到職稱，keywords 留空陣列即可。
- areas 是使用者想要的城市清單（可以是一個或多個），每個值只能是這些之一：
  {", ".join(_AREA_ENUM)}。例如使用者說「雙北」，代表「台北市」和「新北市」兩個都要填進去。
  如果使用者提到的城市不在這個清單裡、或使用者表示不限地區、或完全沒提到地區，就把這個欄位省略
  或設為空陣列，絕對不要自己發明代碼或用清單以外的城市名。
- min_annual_salary 是使用者期望的最低年薪（新台幣，整數）。如果使用者講的是月薪，
  換算成年薪時用「月薪 x 14」概估；如果使用者沒有提到薪資，就省略這個欄位。
- mrt_start_station / mrt_end_station：只有當使用者明確描述「捷運某一條線上某站到某站之間」這種
  通勤範圍時才填（例如「頂埔到忠孝敦化」「淡水到北投」「古亭到景安」），兩個欄位都只能是台北捷運
  現有車站名稱之一。如果使用者只提到一個站（例如「頂埔站附近」），兩個欄位都填那一站。如果使用者
  提到的站名不在捷運站清單裡、或完全沒提到捷運通勤範圍，就把這兩個欄位都省略——不要自己亂猜或
  硬套最接近的站名，也不要自己判斷兩站是否同一條線（這由後端程式檢查）。有填這兩個欄位時就不用
  再填 areas。
- max_walk_minutes：使用者說的「步行 X 分鐘內」的 X（整數，分鐘）。只有在有講到步行時間時才填，
  沒提到就省略，不要自己編一個數字。
- include_remote：是否也要收到「不限地點的全遠端」職缺（跟通勤範圍/城市是 OR 的關係，遠端職缺會
  無視地區條件）。使用者明確表示「不要遠端」「只要通勤範圍內的」「不用遠端職缺」時設為 false；
  明確表示「含遠端」「可以遠端」「也要遠端職缺」時設為 true。這裡的遠端只指「完全遠端」，
  部分遠端（需進辦公室）一律照一般職缺的地區條件過濾，所以「不要部分遠端，只要全遠端」也是 true。
  訊息裡完全沒提到遠端相關字眼時，
  就省略這個欄位——省略時會沿用使用者原本的設定（新訂閱者預設 true）。
- notify_interval_hours：使用者想要「多久檢查一次、通知一次」的小時數（整數，介於
  {_MIN_NOTIFY_INTERVAL_HOURS} 到 {_MAX_NOTIFY_INTERVAL_HOURS} 之間）。例如「改成每 2 小時通知我」
  →2、「一天通知一次就好」→24、「恢復成每小時通知」→1。如果訊息裡完全沒提到通知頻率，就省略這個
  欄位——省略時會沿用使用者原本的設定，不會被重置成預設值。
- max_jobs_per_run：使用者想要「一次最多收到幾筆新職缺通知」的數字（整數，介於
  {_MIN_MAX_JOBS_PER_RUN} 到 {_MAX_MAX_JOBS_PER_RUN} 之間，預設 {_DEFAULT_MAX_JOBS_PER_RUN}）。
  例如「一次給我10筆就好」→10、「最多20筆」→20、「恢復預設」→{_DEFAULT_MAX_JOBS_PER_RUN}。
  如果訊息裡完全沒提到這個數字，就省略這個欄位——省略時會沿用使用者原本的設定，不會被重置。
- 如果使用者只是想調整地區、遠端與否、薪資、捷運範圍、通知頻率或每次通知筆數其中任何一項，
  沒有提到任何職稱關鍵字，intent 一樣設為 "subscribe_or_update"，keywords 留空陣列即可，其他有
  提到的欄位照樣填。
"""


# 長站名優先比對，避免「新埔民生」被拆成「新埔」、「大安森林公園」被拆成「大安」
_STATION_PATTERN = re.compile("|".join(re.escape(s) for s in sorted(ALL_STATIONS, key=len, reverse=True)))


def _fallback_mrt_range(text: str) -> tuple[str, str] | None:
    """Gemini 偶爾會漏填捷運起訖站（例如句子裡同時提到遠端條件時），這裡直接從原文找站名補上。

    只在訊息裡有「捷運」或「站」字時才啟用，避免把「板橋」「中山」這類行政區名誤判成車站。
    """
    if "捷運" not in text and "站" not in text:
        return None
    found = _STATION_PATTERN.findall(text)
    if not found:
        return None
    return found[0], found[-1]


def parse(text: str) -> dict | None:
    """解析使用者輸入，回傳
    {"intent", "keywords", "areas", "min_annual_salary", "mrt_stations", "max_walk_km",
     "include_remote", "notify_interval_hours", "max_jobs_per_run"}；
    Gemini 判斷「看不懂使用者在說什麼」時回傳 None；重試用完仍然呼叫失敗（例如 Gemini
    暫時過載）則把例外往外拋，讓呼叫端能分辨這兩種不同情況、回覆不同的訊息給使用者。
    """
    client = _get_client()  # 金鑰沒設定是設定錯誤，重試也沒用，直接往外拋
    result = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    # 網路卡住時最多等 15 秒就放棄、交給重試邏輯，避免整個 request 卡到被
                    # gunicorn worker timeout 強制 SIGKILL（LINE 完全收不到任何回覆）
                    http_options=types.HttpOptions(timeout=15_000),
                ),
            )
            result = json.loads(response.text)
            break
        except errors.ClientError as exc:
            # 4xx（金鑰錯誤/無權限、model 不存在、schema 不合法、額度用完 429）重試幾秒內也不會好，
            # 直接往外拋，不要讓使用者白等好幾輪重試
            print(f"[錯誤] Gemini 拒絕請求（HTTP {exc.code}，不重試）：{exc}", file=sys.stderr)
            raise
        except Exception as exc:  # noqa: BLE001 - 其他 Gemini/網路例外都視為暫時性失敗
            if attempt < _MAX_RETRIES:
                print(
                    f"[警告] Gemini 解析失敗（第 {attempt} 次，{_RETRY_DELAY_SECONDS} 秒後重試）：{exc}",
                    file=sys.stderr,
                )
                time.sleep(_RETRY_DELAY_SECONDS)
            else:
                print(f"[警告] Gemini 解析失敗（已重試 {_MAX_RETRIES} 次，放棄）：{exc}", file=sys.stderr)
                raise

    intent = result.get("intent")
    notify_interval = result.get("notify_interval_hours")
    if notify_interval is not None:
        notify_interval = max(
            _MIN_NOTIFY_INTERVAL_HOURS, min(_MAX_NOTIFY_INTERVAL_HOURS, int(notify_interval))
        )

    max_jobs_per_run = result.get("max_jobs_per_run")
    if max_jobs_per_run is not None:
        max_jobs_per_run = max(
            _MIN_MAX_JOBS_PER_RUN, min(_MAX_MAX_JOBS_PER_RUN, int(max_jobs_per_run))
        )

    include_remote = result.get("include_remote")
    min_annual_salary = result.get("min_annual_salary")
    areas = [a for a in (result.get("areas") or []) if a in SUPPORTED_CITY_NAMES] or None

    mrt_start = result.get("mrt_start_station")
    mrt_end = result.get("mrt_end_station") or mrt_start
    if not mrt_start:
        fallback = _fallback_mrt_range(text)
        if fallback:
            mrt_start, mrt_end = fallback
            print(f"[警告] Gemini 漏填捷運範圍，改用原文比對：{mrt_start}↔{mrt_end}", file=sys.stderr)
    mrt_stations = stations_between(mrt_start, mrt_end) if mrt_start else None
    if mrt_stations:
        areas = None  # 有精確捷運範圍時，地區改用這個判斷，不用城市層級的 areas

    walk_minutes = result.get("max_walk_minutes")
    max_walk_km = round(walk_minutes * _WALK_KM_PER_MINUTE, 2) if walk_minutes else None

    if intent == "unclear":
        return None
    # 訊息裡沒有職稱關鍵字、也沒有提到任何一項可單獨調整的設定，代表真的看不懂在說什麼
    if (
        intent == "subscribe_or_update"
        and not result.get("keywords")
        and areas is None
        and min_annual_salary is None
        and mrt_stations is None
        and include_remote is None
        and notify_interval is None
        and max_jobs_per_run is None
    ):
        return None

    return {
        "intent": intent,
        "keywords": result.get("keywords", []),
        "areas": areas,
        "min_annual_salary": min_annual_salary,
        "mrt_stations": mrt_stations,
        "max_walk_km": max_walk_km,
        "include_remote": include_remote,
        "notify_interval_hours": notify_interval,
        "max_jobs_per_run": max_jobs_per_run,
    }


# ---------------------------------------------------------------------------
# Gemini 無法使用時（額度用完、金鑰失效、服務過載…）的備援解析
# ---------------------------------------------------------------------------

_FALLBACK_SALARY_PATTERN = re.compile(
    r"(月薪|年薪)\s*(?:要|需|至少|最少|最低|起碼)?\s*(\d[\d,]*(?:\.\d+)?)\s*(萬|[kK千])?\s*元?\s*(?:以上|起跳|起)?"
)
_FALLBACK_CITY_PATTERNS = [
    ("雙北", ["台北市", "新北市"]),
    ("新北市", ["新北市"]),
    ("新北", ["新北市"]),
    ("台北市", ["台北市"]),
    ("臺北市", ["台北市"]),
    ("台北", ["台北市"]),
    ("臺北", ["台北市"]),
]
# 這些字眼代表訊息牽涉到捷運範圍、遠端、通知頻率/筆數等較複雜的設定，規則式解析不可靠，交給 Gemini
_FALLBACK_UNSUPPORTED_HINTS = ["捷運", "站", "遠端", "小時", "筆", "通知", "頻率", "步行", "分鐘"]
_FALLBACK_FILLER_PREFIXES = [
    "我要尋找", "我想尋找", "請幫我找", "請幫我", "幫我找", "我要找", "我想找", "想找", "要找",
    "尋找", "幫我", "我要", "我想", "請", "找", "在", "的",
]
_FALLBACK_FILLER_SUFFIXES = ["的工作", "工作", "的職缺", "職缺", "的", "以上"]
_FALLBACK_SEPARATORS = re.compile(r"[\s,，、。.!！?？;；:：/和及與或跟]+")


def _strip_fillers(token: str) -> str:
    changed = True
    while changed and token:
        changed = False
        for prefix in _FALLBACK_FILLER_PREFIXES:
            if token.startswith(prefix):
                token, changed = token[len(prefix):], True
        for suffix in _FALLBACK_FILLER_SUFFIXES:
            if token.endswith(suffix):
                token, changed = token[: -len(suffix)], True
    return token


def fallback_parse(text: str) -> dict | None:
    """不靠 Gemini、用簡單規則解析「城市＋職稱＋薪資」這種最常見的訂閱訊息。

    只處理有明確職稱關鍵字的訊息；牽涉捷運範圍、遠端、通知頻率等設定的訊息規則式解析
    不可靠，回傳 None 讓呼叫端照舊回覆「系統忙碌」。
    """
    if any(hint in text for hint in _FALLBACK_UNSUPPORTED_HINTS):
        return None

    rest = text
    min_annual_salary = None
    salary_match = _FALLBACK_SALARY_PATTERN.search(rest)
    if salary_match:
        kind, number, unit = salary_match.groups()
        amount = float(number.replace(",", ""))
        if unit == "萬":
            amount *= 10_000
        elif unit:
            amount *= 1_000
        min_annual_salary = int(amount * 14) if kind == "月薪" else int(amount)
        rest = rest[: salary_match.start()] + " " + rest[salary_match.end():]

    areas: list[str] = []
    for name, cities in _FALLBACK_CITY_PATTERNS:
        if name in rest:
            areas.extend(c for c in cities if c not in areas)
            rest = rest.replace(name, " ")

    keywords = []
    for token in _FALLBACK_SEPARATORS.split(rest):
        token = _strip_fillers(token.strip())
        if 2 <= len(token) <= 20 and not re.search(r"\d", token) and token not in keywords:
            keywords.append(token)
    if not keywords:
        return None

    return {
        "intent": "subscribe_or_update",
        "keywords": keywords,
        "areas": areas or None,
        "min_annual_salary": min_annual_salary,
        "mrt_stations": None,
        "max_walk_km": None,
        "include_remote": None,
        "notify_interval_hours": None,
        "max_jobs_per_run": None,
    }


if __name__ == "__main__":
    samples = [
        "我想找台北的前端工程師工作，年薪至少100萬",
        "幫我找新北的後端工程師，月薪至少7萬",
        "取消訂閱",
        "asdkjaslkdj123",
        "我想找高雄的前端工程師",  # 高雄不在支援清單中，驗證 areas 應該被省略
        "我想找雙北的軟體工程師工作，年薪至少150萬",  # areas 應該是 ["台北市","新北市"]
        "我想找頂埔到忠孝敦化之間、步行五分鐘內的前後端工程師工作，年薪至少100萬",
        "改成每3小時通知我一次",  # 純調整頻率，keywords 應該是空陣列
        "一次給我10筆就好",  # 純調整筆數，keywords 應該是空陣列
        "不要遠端的職缺，只要通勤範圍內的",  # 純調整 include_remote，keywords 應該是空陣列
        "我想找全遠端",  # 純調整 include_remote=true，keywords 應該是空陣列，不該是 unclear
        "地區請幫我改成新北市、台北市",  # 純調整 areas，keywords 應該是空陣列，不該是 unclear
    ]
    for sample in samples:
        print(f"輸入：{sample}")
        print(f"解析結果：{parse(sample)}")
        print()

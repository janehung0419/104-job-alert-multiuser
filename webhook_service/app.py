"""LINE Messaging API webhook：接收使用者訊息，讓他們用自然語言自助訂閱/取消訂閱職缺通知。

部署在 Render 免費方案上（見 README「Render 部署」章節）。注意 Render 免費方案的
硬碟是暫時性的，這支程式不能寫任何檔案到本機——所有訂閱者資料都存在 Supabase。
"""

import hashlib
import hmac
import base64
import os
import sys
import traceback
from pathlib import Path

import requests
from flask import Flask, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import db  # noqa: E402
from common.area_codes import AREA_CODE_MAP  # noqa: E402
from webhook_service import gemini_parser  # noqa: E402

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

UNSUBSCRIBE_KEYWORDS = ["取消訂閱", "退訂", "unsubscribe", "stop"]

WELCOME_MESSAGE = (
    "歡迎使用找工作小幫手！請直接傳一句話告訴我你想找什麼樣的工作，例如：\n"
    "「我想找台北的前端工程師工作，年薪至少100萬」\n"
    "「我想找頂埔到忠孝敦化之間、步行五分鐘內的後端工程師，年薪至少100萬」\n\n"
    "之後我每小時都會幫你檢查一次新職缺（預設一次最多通知最新 5 筆，其餘留到下次繼續通知，"
    "預設也會一併通知全遠端職缺），傳訊息通知你。想改成不同的頻率、筆數或不要遠端職缺，直接傳"
    "「改成每 3 小時通知我」「一次給我10筆就好」「不要遠端職缺」之類的句子即可（筆數最多 20）。\n"
    "想取消訂閱，隨時傳「取消訂閱」即可。"
)
UNCLEAR_MESSAGE = (
    "不好意思，我沒有讀懂你的需求 🙏 可以換個方式描述嗎？例如：\n"
    "「我想找新北的後端工程師工作，年薪至少90萬」"
)
ERROR_MESSAGE = "系統暫時忙碌，請稍後再試一次 🙏"
NEED_KEYWORDS_MESSAGE = (
    "你還沒有訂閱職缺通知喔 🙏 請先告訴我想找的職稱，例如：\n"
    "「我想找頂埔到忠孝新生之間的後端工程師，也要全遠端職缺」"
)
UNSUBSCRIBE_MESSAGE = "已經幫你取消訂閱囉，之後不會再收到職缺通知。之後想重新開始，再傳一次你的需求給我即可。"

app = Flask(__name__)


@app.get("/")
def health_check():
    return "OK"


def _verify_signature(body: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET or not signature:
        return False
    digest = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def _reply(reply_token: str, text: str) -> None:
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {"replyToken": reply_token, "messages": [{"type": "text", "text": text}]}
    resp = requests.post(LINE_REPLY_URL, headers=headers, json=body, timeout=15)
    if resp.status_code != 200:
        print(f"[錯誤] LINE 回覆失敗：HTTP {resp.status_code} {resp.text}", file=sys.stderr)


def _area_line(parsed: dict) -> str:
    mrt_stations = parsed.get("mrt_stations")
    if mrt_stations:
        walk_km = parsed.get("max_walk_km")
        walk_bit = f"，步行 {walk_km} 公里內" if walk_km else ""
        return f"捷運 {mrt_stations[0]}↔{mrt_stations[-1]}{walk_bit}"
    return "、".join(parsed.get("areas") or []) or "不限"


def _area_codes(areas: list[str] | None) -> list[str] | None:
    codes = [AREA_CODE_MAP[a] for a in (areas or []) if a in AREA_CODE_MAP]
    return codes or None


def _confirmation_message(
    parsed: dict, notify_interval_hours: int, max_jobs_per_run: int, include_remote: bool
) -> str:
    keywords_line = "、".join(parsed["keywords"])
    area_line = _area_line(parsed)
    remote_line = "包含（僅完全遠端）" if include_remote else "不包含"
    salary = parsed["min_annual_salary"]
    salary_line = f"{salary:,}" if salary else "不限"
    return (
        "已經幫你更新訂閱條件 ✅\n"
        f"關鍵字：{keywords_line}\n"
        f"地區：{area_line}\n"
        f"是否含遠端職缺：{remote_line}\n"
        f"最低年薪：{salary_line}\n"
        f"通知頻率：每 {notify_interval_hours} 小時檢查一次\n"
        f"每次最多通知：{max_jobs_per_run} 筆\n\n"
        "其餘尚未通知過的職缺會留到下次繼續通知，不會漏掉。\n"
        "想修改條件，直接再傳一次新的需求就會覆蓋舊的設定（通知頻率、筆數、是否含遠端除外，沒提到"
        "就會沿用原本設定）；想單獨調整這幾項，傳「改成每 X 小時通知我」「一次給我 X 筆就好」"
        "「不要遠端職缺」；想取消訂閱，傳「取消訂閱」。"
    )


def _handle_text_message(user_id: str, reply_token: str, text: str) -> None:
    text = text.strip()

    if any(keyword.lower() in text.lower() for keyword in UNSUBSCRIBE_KEYWORDS):
        db.deactivate_subscriber(user_id)
        _reply(reply_token, UNSUBSCRIBE_MESSAGE)
        return

    try:
        parsed = gemini_parser.parse(text)
    except Exception:  # noqa: BLE001 - Gemini 暫時忙碌/呼叫失敗，跟「看不懂」是不同情況
        traceback.print_exc()
        # Gemini 掛掉（額度用完、金鑰失效、過載…）時，「城市＋職稱＋薪資」這種常見訊息
        # 改用規則式解析，不要讓使用者只能一直收到「系統忙碌」
        parsed = gemini_parser.fallback_parse(text)
        if parsed is None:
            _reply(reply_token, ERROR_MESSAGE)
            return
        print(f"[警告] Gemini 無法使用，改用規則式解析：{parsed}", file=sys.stderr)
    if parsed is None:
        _reply(reply_token, UNCLEAR_MESSAGE)
        return

    if parsed["intent"] == "unsubscribe":
        db.deactivate_subscriber(user_id)
        _reply(reply_token, UNSUBSCRIBE_MESSAGE)
        return

    interval = parsed.get("notify_interval_hours")
    max_jobs_per_run = parsed.get("max_jobs_per_run")
    include_remote = parsed.get("include_remote")
    areas = parsed.get("areas")
    min_annual_salary = parsed.get("min_annual_salary")
    mrt_stations = parsed.get("mrt_stations")
    max_walk_km = parsed.get("max_walk_km")

    # 訊息裡沒有任何職稱關鍵字，代表使用者只是想單獨調整設定，不要動到既有的職缺關鍵字
    if not parsed["keywords"]:
        changes = []
        if mrt_stations or areas:
            changes.append(f"地區：{_area_line(parsed)}")
        if min_annual_salary is not None:
            changes.append(f"最低年薪：{min_annual_salary:,}")
        if include_remote is not None:
            changes.append(f"是否含遠端職缺：{'包含（僅完全遠端）' if include_remote else '不包含'}")
        if interval is not None:
            changes.append(f"通知頻率：每 {interval} 小時檢查一次")
        if max_jobs_per_run is not None:
            changes.append(f"每次最多通知：{max_jobs_per_run} 筆")

        if not changes:
            _reply(reply_token, UNCLEAR_MESSAGE)
            return
        # 捷運範圍與城市是二擇一的地區條件：設定其中一種時要清掉另一種（傳空清單），
        # 否則殘留的舊城市會讓排程只搜尋那個城市，捷運範圍內其他城市的職缺就搜不到
        area_kwargs = {}
        if mrt_stations:
            area_kwargs = dict(
                mrt_stations=mrt_stations, max_walk_km=max_walk_km, area_codes=[], area_labels=[]
            )
        elif areas:
            area_kwargs = dict(area_codes=_area_codes(areas), area_labels=areas, mrt_stations=[])
        updated = db.update_settings(
            user_id,
            notify_interval_hours=interval,
            max_jobs_per_run=max_jobs_per_run,
            include_remote=include_remote,
            min_annual_salary=min_annual_salary,
            **area_kwargs,
        )
        if updated:
            _reply(reply_token, "已經幫你更新設定 ✅\n" + "\n".join(changes) + "\n\n其他訂閱條件維持不變。")
        else:
            _reply(reply_token, NEED_KEYWORDS_MESSAGE)
        return

    existing = db.get_subscriber(user_id)
    final_interval = interval or (existing["notify_interval_hours"] if existing else 1)
    final_max_jobs = max_jobs_per_run or (existing["max_jobs_per_run"] if existing else 5)
    if include_remote is not None:
        final_include_remote = include_remote
    else:
        final_include_remote = existing["include_remote"] if existing else True

    db.upsert_subscriber(
        line_user_id=user_id,
        keywords=parsed["keywords"],
        area_codes=_area_codes(areas),
        area_labels=areas,
        min_annual_salary=min_annual_salary,
        mrt_stations=mrt_stations,
        max_walk_km=max_walk_km,
        include_remote=include_remote,
        notify_interval_hours=interval,
        max_jobs_per_run=max_jobs_per_run,
    )
    _reply(
        reply_token,
        _confirmation_message(parsed, final_interval, final_max_jobs, final_include_remote),
    )


@app.post("/webhook")
def webhook():
    body = request.get_data()
    signature = request.headers.get("X-Line-Signature", "")
    if not _verify_signature(body, signature):
        return "invalid signature", 400

    payload = request.get_json(silent=True) or {}
    for event in payload.get("events", []):
        event_type = event.get("type")
        try:
            if event_type == "follow":
                _reply(event["replyToken"], WELCOME_MESSAGE)
            elif event_type == "unfollow":
                db.deactivate_subscriber(event["source"]["userId"])
            elif event_type == "message" and event.get("message", {}).get("type") == "text":
                _handle_text_message(
                    user_id=event["source"]["userId"],
                    reply_token=event["replyToken"],
                    text=event["message"]["text"],
                )
        except Exception as exc:  # noqa: BLE001 - 單一事件出錯不能影響其他事件，也不能讓 LINE 重試風暴
            print(f"[錯誤] 處理事件失敗：{exc}", file=sys.stderr)
            traceback.print_exc()
            reply_token = event.get("replyToken")
            if reply_token:
                _reply(reply_token, ERROR_MESSAGE)

    return "OK", 200


if __name__ == "__main__":
    app.run(debug=True)

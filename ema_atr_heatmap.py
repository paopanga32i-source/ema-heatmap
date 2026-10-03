"""
EMA ATR Heatmap — 2 กลุ่ม (หุ้นใหญ่ / หุ้นกลางสภาพคล่องสูง)
วัดว่าราคาปิดอยู่ห่างจาก EMA เร็วกี่เท่าของ ATR + กรองหุ้นที่ราคาเปิดสูงกว่า EMA ช้า

วิธีรัน:
    pip install streamlit yfinance pandas
    streamlit run ema_atr_heatmap.py
"""

from datetime import timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="EMA ATR Heatmap", page_icon="🔥", layout="wide")

# ---------------- รายชื่อเริ่มต้น ----------------
# หุ้นใหญ่: market cap > 30,000 ล้านบาท (ประมาณจาก SET50 — ควรเช็กกับข้อมูล SET อีกครั้ง)
BIG_TH = (
    "ADVANC AOT AWC BBL BDMS BEM BH BJC BTS CBG CCET CENTEL COM7 CPALL "
    "CPAXT CPF CPN CRC DELTA EGCO GPSC GULF HMPRO IVL KBANK KKP KTB KTC LH "
    "MINT MTC OR OSP PTT PTTEP PTTGC RATCH SCB SCC SCGP TCAP TIDLOR TISCO "
    "TLI TOP TRUE TTB WHA"
)
# หุ้นกลาง: market cap ราว 5,000–30,000 ล้านบาท ที่คาดว่าซื้อขายคล่อง (ยังไม่ได้ยืนยัน)
MID_TH = (
    "AAV AMATA AP AURA BAM BCH BCP BLA CHG CK CKP DOHOME ERW GFPT GLOBAL "
    "HANA ICHI JMART JMT KCE MEGA MOSHI PLANB PR9 PSL QH RCL SAWAD SIRI "
    "SISB SJWD SNNP SPALI STA STGT TASCO THANI TKN TOA TQM VGI WHAUP"
)
BIG_US = "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX COST JPM LLY PLTR"
MID_US = "ELF DUOL ONON CAVA HIMS AFRM SOFI RKLB IONQ CELH"

# ระดับสีตามระยะห่าง (เท่าของ ATR)
LEVELS = [
    (float("-inf"), 0, "#3F3F3F", "#FFFFFF", "ต่ำกว่า EMA เร็ว (ห้ามเล่น)"),
    (0, 1, "#E3F4E6", "#1E4D2B", "0 – 1"),
    (1, 1.5, "#B9E4C2", "#1E4D2B", "1 – 1.5"),
    (1.5, 2, "#86CF98", "#123D20", "1.5 – 2"),
    (2, 2.5, "#4FAE6A", "#FFFFFF", "2 – 2.5"),
    (2.5, 3, "#2B8A47", "#FFFFFF", "2.5 – 3"),
    (3, float("inf"), "#145A2C", "#FFFFFF", "มากกว่า 3"),
]


def level_of(v):
    if pd.isna(v):
        return "#FFFFFF", "#AAAAAA"
    for lo, hi, bg, fg, _ in LEVELS:
        if lo <= v < hi:
            return bg, fg
    return LEVELS[-1][2], LEVELS[-1][3]


def parse(raw, suffix):
    syms = [s.strip().upper() for s in raw.replace(",", " ").split() if s.strip()]
    seen, out = set(), []
    for s in syms:
        t = s if (not suffix or s.endswith(suffix)) else s + suffix
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------- ดึงข้อมูล ----------------
@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers: tuple, today: str):
    end = pd.Timestamp(today) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=400)
    data = yf.download(
        list(tickers), start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
        interval="1d", group_by="ticker", auto_adjust=True, threads=False, progress=False,
    )
    out = {}
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if len(df) >= 60:
                out[t] = df
        except KeyError:
            pass
    return out


def compute(df, ema_fast, ema_slow, atr_len):
    d = df.copy()
    d["EMA_F"] = d["Close"].ewm(span=ema_fast, adjust=False).mean()
    d["EMA_S"] = d["Close"].ewm(span=ema_slow, adjust=False).mean()
    prev_close = d["Close"].shift(1)
    tr = pd.concat(
        [d["High"] - d["Low"], (d["High"] - prev_close).abs(), (d["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / atr_len, adjust=False).mean()  # Wilder ATR
    d["DIST"] = (d["Close"] - d["EMA_F"]) / d["ATR"]
    d["OPEN_ABOVE_SLOW"] = d["Open"] > d["EMA_S"]
    d["VALUE_M"] = d["Close"] * d["Volume"] / 1e6  # มูลค่าซื้อขาย (ล้าน)
    return d


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("ตั้งค่า")
    market = st.radio("ตลาด", ["หุ้นไทย", "หุ้น US"], horizontal=True)
    is_th = market == "หุ้นไทย"
    unit = "ล้านบาท" if is_th else "ล้าน USD"

    ema_fast = st.number_input("EMA เร็ว (วัดระยะห่าง)", 5, 100, 20)
    ema_slow = st.number_input("EMA ช้า (กรองราคาเปิด)", 10, 300, 50)
    atr_len = st.number_input("ATR", 5, 50, 14)
    days = st.slider("จำนวนวันย้อนหลังใน heatmap", 5, 30, 10)
    only_pass = st.checkbox("แสดงเฉพาะหุ้นที่ราคาเปิดวันล่าสุด > EMA ช้า", True)

    st.subheader("หุ้นใหญ่")
    big_min = st.number_input(f"มูลค่าซื้อขายเฉลี่ย 20 วัน ขั้นต่ำ ({unit})",
                              0, 10000, 100 if is_th else 500, key="big_min")
    with st.expander("แก้รายชื่อหุ้นใหญ่"):
        big_raw = st.text_area("รายชื่อ", BIG_TH if is_th else BIG_US, height=160,
                               key=f"big_{market}")

    st.subheader("หุ้นกลางสภาพคล่องสูง")
    mid_min = st.number_input(f"มูลค่าซื้อขายเฉลี่ย 20 วัน ขั้นต่ำ ({unit})",
                              0, 10000, 50 if is_th else 100, key="mid_min")
    mid_days = st.slider("ต้องซื้อขายเกินขั้นต่ำอย่างน้อยกี่วันใน 20 วัน", 0, 20, 15)
    with st.expander("แก้รายชื่อหุ้นกลาง"):
        mid_raw = st.text_area("รายชื่อ", MID_TH if is_th else MID_US, height=160,
                               key=f"mid_{market}")

    if st.button("ดึงข้อมูลใหม่"):
        st.cache_data.clear()

suffix = ".BK" if is_th else ""
big = parse(big_raw, suffix)
mid = [t for t in parse(mid_raw, suffix) if t not in big]  # กันซ้ำกลุ่ม
all_tickers = tuple(big + mid)

st.title(f"ระยะห่างจาก EMA{ema_fast} เป็นหน่วย ATR")
st.caption(
    f"ค่า = (ราคาปิด − EMA{ema_fast}) ÷ ATR{atr_len}  |  ตัวกรอง: ราคาเปิดสูงกว่า EMA{ema_slow}  |  "
    "ข้อมูลจาก Yahoo Finance อาจดีเลย์"
)

if not all_tickers:
    st.info("ใส่รายชื่อหุ้นในแถบด้านซ้ายเพื่อเริ่มสแกน")
    st.stop()

with st.spinner(f"กำลังดึงราคา {len(all_tickers)} ตัว..."):
    today_bkk = pd.Timestamp.now(tz="Asia/Bangkok").strftime("%Y-%m-%d")
    prices = load_prices(all_tickers, today_bkk)

short = lambda t: t[: -len(suffix)] if suffix and t.endswith(suffix) else t

missing = [t for t in all_tickers if t not in prices]
if missing:
    st.warning("ดึงข้อมูลไม่ได้: " + ", ".join(short(m) for m in missing))

# ---------------- เช็กความสดของข้อมูล ----------------
if not prices:
    st.error("ไม่มีข้อมูลให้แสดง ลองเช็กชื่อหุ้นหรือกดดึงข้อมูลใหม่")
    st.stop()

last_dates = pd.Series({t: pd.Timestamp(df.index[-1]).date() for t, df in prices.items()})
newest = last_dates.max()
lag_days = (pd.Timestamp.now(tz="Asia/Bangkok").date() - newest).days
st.caption(f"ข้อมูลล่าสุดถึงวันที่ {newest.strftime('%d/%m/%Y')}")
if lag_days > 4:
    st.error(f"ข้อมูลล่าสุดเก่ากว่าวันนี้ {lag_days} วัน ลองกด 'ดึงข้อมูลใหม่'")
stale = last_dates[last_dates < newest - timedelta(days=3)]
if len(stale):
    st.warning("ตัดออกเพราะข้อมูลไม่อัปเดต: " + ", ".join(
        f"{short(t)} (ถึง {d.strftime('%d/%m')})" for t, d in stale.items()))
    prices = {t: df for t, df in prices.items() if t not in stale.index}

computed = {t: compute(df, ema_fast, ema_slow, atr_len) for t, df in prices.items()}

# ---------------- Style ----------------
st.markdown(
    """
<style>
.hm-wrap{overflow-x:auto}
.hm{border-collapse:separate;border-spacing:2px;font-size:13px;font-variant-numeric:tabular-nums}
.hm th{font-weight:500;color:#666;padding:4px 6px;white-space:nowrap}
.hm th.sym{text-align:left;font-weight:600;position:sticky;left:0;z-index:1;
  background:#FFFFFF;color:#111111}
@media (prefers-color-scheme: dark){
  .hm th.sym{background:#0E1117;color:#FAFAFA}
}
.hm td{min-width:44px;text-align:center;padding:6px 4px;border-radius:3px}
</style>
""",
    unsafe_allow_html=True,
)
legend = "".join(
    f'<span style="background:{bg};color:{fg};padding:4px 10px;border-radius:4px;'
    f'margin:0 6px 6px 0;display:inline-block;font-size:13px">{label}</span>'
    for _, _, bg, fg, label in LEVELS
)


def build_group(tickers, min_value, min_days):
    """คืนตารางสรุปของกลุ่ม + หุ้นที่ตกเกณฑ์สภาพคล่อง"""
    rows, low_liq = [], []
    for t in tickers:
        if t not in computed:
            continue
        d = computed[t]
        last = d.iloc[-1]
        v20 = d["VALUE_M"].tail(20)
        avg_val = v20.mean()
        days_ok = int((v20 >= min_value).sum())
        if avg_val < min_value or days_ok < min_days:
            low_liq.append(short(t))
            continue
        rows.append({
            "หุ้น": short(t),
            "ราคาปิด": round(last["Close"], 2),
            f"EMA{ema_fast}": round(last["EMA_F"], 2),
            f"EMA{ema_slow}": round(last["EMA_S"], 2),
            "ATR": round(last["ATR"], 2),
            "ATR%": round(last["ATR"] / last["Close"] * 100, 2),
            "ระยะ (ATR)": round(last["DIST"], 2),
            f"มูลค่าเฉลี่ย 20 วัน ({unit})": round(avg_val, 1),
            "เปิด > EMA ช้า": bool(last["OPEN_ABOVE_SLOW"]),
            "_t": t,
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("ระยะ (ATR)", ascending=False)
        if only_pass:
            table = table[table["เปิด > EMA ช้า"]]
    return table, low_liq


def render_group(table, low_liq, scanned, key):
    c1, c2, c3 = st.columns(3)
    c1.metric("สแกน", scanned)
    c2.metric("ผ่านตัวกรอง", len(table))
    c3.metric("ยืดเกิน 3 ATR", int((table["ระยะ (ATR)"] >= 3).sum()) if len(table) else 0)
    if low_liq:
        st.caption("ตัดออกเพราะสภาพคล่องไม่ถึงเกณฑ์: " + ", ".join(low_liq))
    st.markdown(legend, unsafe_allow_html=True)

    if table.empty:
        st.info("กลุ่มนี้ไม่มีหุ้นผ่านเงื่อนไขวันนี้ ลองปิดตัวกรองราคาเปิดหรือลดเกณฑ์สภาพคล่อง")
        return

    hist = {r["หุ้น"]: computed[r["_t"]].tail(days) for _, r in table.iterrows()}
    all_dates = sorted(set().union(*[set(h.index) for h in hist.values()]))
    dates = all_dates[-days:]
    head = "".join(f"<th>{pd.Timestamp(x).strftime('%d/%m')}</th>" for x in dates)
    body = ""
    for name, h in hist.items():
        h = h.reindex(dates)
        cells = ""
        for _, r in h.iterrows():
            if pd.isna(r["DIST"]):
                cells += '<td style="color:#AAA">–</td>'
                continue
            bg, fg = level_of(r["DIST"])
            mark = "" if r["OPEN_ABOVE_SLOW"] else "opacity:0.35;"
            cells += f'<td style="background:{bg};color:{fg};{mark}">{r["DIST"]:.1f}</td>'
        body += f"<tr><th class='sym'>{name}</th>{cells}</tr>"
    st.markdown(
        f'<div class="hm-wrap"><table class="hm"><tr><th></th>{head}</tr>{body}</table></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"ช่องจาง = วันนั้นราคาเปิดไม่สูงกว่า EMA{ema_slow}  |  เรียงจากระยะห่างมากไปน้อยของวันล่าสุด")

    st.subheader("รายละเอียดวันล่าสุด")
    show = table.drop(columns=["_t"])
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.download_button(
        "ดาวน์โหลด CSV", show.to_csv(index=False).encode("utf-8-sig"),
        f"ema_atr_{key}.csv", "text/csv", key=f"dl_{key}",
    )


big_table, big_low = build_group(big, big_min, 0)
mid_table, mid_low = build_group(mid, mid_min, mid_days)

tab_big, tab_mid = st.tabs([
    f"หุ้นใหญ่ ({len(big_table)})",
    f"หุ้นกลางสภาพคล่องสูง ({len(mid_table)})",
])
with tab_big:
    render_group(big_table, big_low, len(big), "big")
with tab_mid:
    st.caption("หุ้นกลุ่มนี้เหวี่ยงแรงกว่า ควรใช้ขนาด position เล็กกว่า และเช็กมาตรการ Cash Balance ก่อนเข้า")
    render_group(mid_table, mid_low, len(mid), "mid")

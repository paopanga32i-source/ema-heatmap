"""
EMA20 ATR Heatmap
วัดว่าราคาปิดอยู่ห่างจาก EMA20 กี่เท่าของ ATR + กรองหุ้นที่ราคาเปิดสูงกว่า EMA50

วิธีรัน:
    pip install streamlit yfinance pandas
    streamlit run ema_atr_heatmap.py
"""

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="EMA20 ATR Heatmap", page_icon="🔥", layout="wide")

# ---------------- ค่าเริ่มต้น ----------------
DEFAULT_TH = (
    "ADVANC AOT AWC BANPU BBL BDMS BEM BH BJC BTS CBG CCET CENTEL COM7 CPALL "
    "CPAXT CPF CPN CRC DELTA EGCO GPSC GULF HMPRO IVL KBANK KKP KTB KTC LH "
    "MINT MTC OR OSP PTT PTTEP PTTGC RATCH SCB SCC SCGP TCAP TIDLOR TISCO "
    "TLI TOP TRUE TTB WHA"
)
DEFAULT_US = "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX COST JPM LLY PLTR"

# ระดับสีตามระยะห่าง (เท่าของ ATR)
LEVELS = [
    (float("-inf"), 0, "#5B7DB1", "#FFFFFF", "ต่ำกว่า EMA20"),
    (0, 1, "#E9ECEF", "#333333", "0 – 1"),
    (1, 1.5, "#B7E4C7", "#1B4332", "1 – 1.5"),
    (1.5, 2, "#52B788", "#FFFFFF", "1.5 – 2"),
    (2, 2.5, "#FFD166", "#5C3D00", "2 – 2.5"),
    (2.5, 3, "#F4A261", "#4A1F00", "2.5 – 3"),
    (3, float("inf"), "#D62828", "#FFFFFF", "มากกว่า 3"),
]


def level_of(v):
    if pd.isna(v):
        return "#FFFFFF", "#AAAAAA"
    for lo, hi, bg, fg, _ in LEVELS:
        if lo <= v < hi:
            return bg, fg
    return LEVELS[-1][2], LEVELS[-1][3]


# ---------------- ดึงข้อมูล ----------------
@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers: tuple):
    data = yf.download(
        list(tickers), period="1y", interval="1d",
        group_by="ticker", auto_adjust=True, threads=True, progress=False,
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
    return d


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("ตั้งค่า")
    market = st.radio("ตลาด", ["หุ้นไทย", "หุ้น US"], horizontal=True)
    default_list = DEFAULT_TH if market == "หุ้นไทย" else DEFAULT_US
    raw = st.text_area("รายชื่อหุ้น (เว้นวรรคหรือขึ้นบรรทัดใหม่)", default_list, height=180)
    ema_fast = st.number_input("EMA เร็ว", 5, 100, 20)
    ema_slow = st.number_input("EMA ช้า (ใช้กรองราคาเปิด)", 10, 300, 50)
    atr_len = st.number_input("ATR", 5, 50, 14)
    days = st.slider("จำนวนวันย้อนหลังใน heatmap", 5, 30, 10)
    only_pass = st.checkbox("แสดงเฉพาะหุ้นที่ราคาเปิดวันล่าสุด > EMA ช้า", True)
    if st.button("ดึงข้อมูลใหม่"):
        st.cache_data.clear()

symbols = [s.strip().upper() for s in raw.replace(",", " ").split() if s.strip()]
suffix = ".BK" if market == "หุ้นไทย" else ""
tickers = tuple(s if s.endswith(suffix) else s + suffix for s in symbols)

st.title("ระยะห่างจาก EMA20 เป็นหน่วย ATR")
st.caption(
    f"ค่า = (ราคาปิด − EMA{ema_fast}) ÷ ATR{atr_len}  |  "
    f"ตัวกรอง: ราคาเปิดสูงกว่า EMA{ema_slow}  |  ข้อมูลจาก Yahoo Finance อาจดีเลย์"
)

if not tickers:
    st.info("ใส่รายชื่อหุ้นในแถบด้านซ้ายเพื่อเริ่มสแกน")
    st.stop()

with st.spinner(f"กำลังดึงราคา {len(tickers)} ตัว..."):
    prices = load_prices(tickers)

missing = [t for t in tickers if t not in prices]
if missing:
    st.warning("ดึงข้อมูลไม่ได้: " + ", ".join(m.replace(suffix, "") for m in missing))

rows, hist = [], {}
for t, df in prices.items():
    d = compute(df, ema_fast, ema_slow, atr_len)
    last = d.iloc[-1]
    name = t.replace(suffix, "") if suffix else t
    rows.append({
        "หุ้น": name,
        "ราคาปิด": round(last["Close"], 2),
        f"EMA{ema_fast}": round(last["EMA_F"], 2),
        f"EMA{ema_slow}": round(last["EMA_S"], 2),
        "ATR": round(last["ATR"], 2),
        "ระยะ (ATR)": round(last["DIST"], 2),
        "เปิด > EMA ช้า": bool(last["OPEN_ABOVE_SLOW"]),
    })
    hist[name] = d.tail(days)

if not rows:
    st.error("ไม่มีข้อมูลให้แสดง ลองเช็กชื่อหุ้นหรือกดดึงข้อมูลใหม่")
    st.stop()

table = pd.DataFrame(rows).sort_values("ระยะ (ATR)", ascending=False)
if only_pass:
    table = table[table["เปิด > EMA ช้า"]]

c1, c2, c3 = st.columns(3)
c1.metric("สแกนทั้งหมด", len(rows))
c2.metric("ผ่านตัวกรอง", int(table["เปิด > EMA ช้า"].sum()))
c3.metric("ยืดเกิน 3 ATR", int((table["ระยะ (ATR)"] >= 3).sum()))

# ---------------- Legend ----------------
legend = "".join(
    f'<span style="background:{bg};color:{fg};padding:4px 10px;border-radius:4px;'
    f'margin:0 6px 6px 0;display:inline-block;font-size:13px">{label}</span>'
    for _, _, bg, fg, label in LEVELS
)
st.markdown(legend, unsafe_allow_html=True)

if table.empty:
    st.info("วันนี้ไม่มีหุ้นที่ราคาเปิดสูงกว่า EMA ช้า ลองปิดตัวกรองหรือเพิ่มรายชื่อหุ้น")
    st.stop()

# ---------------- Heatmap ----------------
order = table["หุ้น"].tolist()
dates = hist[order[0]].index
head = "".join(f"<th>{pd.Timestamp(x).strftime('%d/%m')}</th>" for x in dates)
body = ""
for name in order:
    h = hist[name]
    cells = ""
    for _, r in h.iterrows():
        bg, fg = level_of(r["DIST"])
        mark = "" if r["OPEN_ABOVE_SLOW"] else "opacity:0.35;"
        cells += f'<td style="background:{bg};color:{fg};{mark}">{r["DIST"]:.1f}</td>'
    body += f"<tr><th class='sym'>{name}</th>{cells}</tr>"

st.markdown(
    """
<style>
.hm-wrap{overflow-x:auto}
.hm{border-collapse:separate;border-spacing:2px;font-size:13px;font-variant-numeric:tabular-nums}
.hm th{font-weight:500;color:#666;padding:4px 6px;white-space:nowrap}
.hm th.sym{text-align:left;color:inherit;font-weight:600;position:sticky;left:0;background:var(--background-color,#fff)}
.hm td{min-width:44px;text-align:center;padding:6px 4px;border-radius:3px}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="hm-wrap"><table class="hm"><tr><th></th>{head}</tr>{body}</table></div>',
    unsafe_allow_html=True,
)
st.caption(f"ช่องจาง = วันนั้นราคาเปิดไม่สูงกว่า EMA{ema_slow}  |  เรียงจากระยะห่างมากไปน้อยของวันล่าสุด")

# ---------------- ตารางรายละเอียด ----------------
st.subheader("รายละเอียดวันล่าสุด")
st.dataframe(table, hide_index=True, use_container_width=True)
st.download_button(
    "ดาวน์โหลด CSV", table.to_csv(index=False).encode("utf-8-sig"),
    "ema_atr_scan.csv", "text/csv",
)

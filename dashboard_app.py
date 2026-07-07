"""Dashboard RV Requin — rode com: streamlit run dashboard_app.py"""

from datetime import datetime
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Dashboard RV Requin", page_icon="📊", layout="wide")

PERIOD_INTERVAL_MAP = {
    "1d": "1m",
    "5d": "15m",
    "1mo": "1d",
    "6mo": "1d",
    "1y": "1wk",
    "5y": "1mo",
}

PERIOD_LABELS = {
    "1d": "1 dia",
    "5d": "5 dias",
    "1mo": "1 mês",
    "6mo": "6 meses",
    "1y": "1 ano",
    "5y": "5 anos",
}

CHART_TYPES = ["Candle", "Linha", "Barra"]

MACRO_INDICATORS = [
    {"label": "Selic (meta)", "code": 432, "unit": "% a.a.", "focus": "Selic", "freq": "diária"},
    {"label": "IPCA (12 meses)", "code": 13522, "unit": "% a.a.", "focus": "IPCA", "freq": "mensal"},
    {"label": "Dívida Bruta / PIB", "code": 13762, "unit": "% do PIB", "focus": None, "freq": "mensal"},
    {"label": "Câmbio (USD/BRL)", "code": 1, "unit": "R$", "focus": "Câmbio", "freq": "diária"},
    {"label": "PIB (variação real anual)", "code": 7326, "unit": "%", "focus": "PIB Total", "freq": "anual"},
]

# O endpoint /dados/ultimos/{n} do BCB aceita no máximo 20 valores e o
# endpoint de intervalo de datas aceita no máximo ~10 anos para séries
# diárias, então cada opção define sua própria janela de busca.
MACRO_HISTORY_OPTIONS = {
    "Variação do PIB (anual)": {"type": "sgs", "code": 7326, "unit": "%", "years_back": 25, "n_tail": 20},
    "IPCA mensal": {"type": "sgs", "code": 433, "unit": "%", "years_back": 8, "n_tail": 36},
    "IPCA acumulado 12 meses": {"type": "sgs", "code": 13522, "unit": "% a.a.", "years_back": 8, "n_tail": 36},
    "IPCA acumulado no ano (YTD)": {"type": "ipca_ytd", "unit": "%", "n_years": 5},
    "Selic (nível)": {"type": "sgs", "code": 432, "unit": "% a.a.", "years_back": 3, "n_tail": 365},
    "Dívida pública / PIB": {"type": "sgs", "code": 13762, "unit": "% do PIB", "years_back": 8, "n_tail": 36},
}

# Paleta institucional
NAVY_DARK = "#0b1f3a"
NAVY = "#0f2747"
GOLD = "#c9a227"
BLUE_ACCENT = "#4fc3f7"
UP_COLOR = "#2e7d32"
DOWN_COLOR = "#c62828"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def fetch_history(ticker: str, period: str) -> pd.DataFrame:
    interval = PERIOD_INTERVAL_MAP.get(period, "1d")
    return yf.Ticker(ticker).history(period=period, interval=interval)


@st.cache_data(ttl=5)
def fetch_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).get_info()

    ebitda = info.get("ebitda")
    total_debt = info.get("totalDebt")
    total_cash = info.get("totalCash")
    net_debt_ebitda = None
    if ebitda and total_debt is not None and total_cash is not None:
        net_debt_ebitda = (total_debt - total_cash) / ebitda

    return {
        "name": info.get("shortName", ticker),
        "currency": info.get("currency", ""),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose"),
        "pe": info.get("trailingPE"),
        "dividend_yield": info.get("dividendYield"),
        "roe": info.get("returnOnEquity"),
        "price_to_book": info.get("priceToBook"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "net_debt_ebitda": net_debt_ebitda,
    }


@st.cache_data(ttl=3600)
def fetch_bcb_series(code: int, n: int = 60) -> pd.DataFrame:
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/{n}?formato=json"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    # A API não garante ordem cronológica em todas as séries.
    return df.sort_values("data").reset_index(drop=True)


@st.cache_data(ttl=3600)
def fetch_bcb_history(code: int, years_back: int, n_tail: int) -> pd.DataFrame:
    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=years_back)
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
        f"?formato=json&dataInicial={start:%d/%m/%Y}&dataFinal={end:%d/%m/%Y}"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    # A API não garante ordem cronológica em todas as séries.
    return df.sort_values("data").tail(n_tail).reset_index(drop=True)


@st.cache_data(ttl=3600)
def fetch_ipca_ytd(n_years: int = 5) -> pd.DataFrame:
    monthly = fetch_bcb_history(433, years_back=n_years + 1, n_tail=(n_years + 1) * 12).sort_values("data").copy()
    monthly["year"] = monthly["data"].dt.year
    monthly["factor"] = 1 + monthly["valor"] / 100
    monthly["ytd_factor"] = monthly.groupby("year")["factor"].cumprod()
    monthly["valor"] = (monthly["ytd_factor"] - 1) * 100
    return monthly[["data", "valor"]]


@st.cache_data(ttl=3600)
def fetch_focus_expectation(indicator: str, year: int | None = None):
    year = year or datetime.now().year
    base = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
    filter_str = f"Indicador eq '{indicator}' and DataReferencia eq '{year}'"
    url = f"{base}?$top=1&$filter={quote(filter_str, safe=chr(39))}&$orderby=Data%20desc&$format=json"
    url = url.replace(" ", "%20")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    rows = resp.json().get("value", [])
    if not rows:
        return None
    return float(rows[0]["Mediana"])


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_currency(value, currency=""):
    if value is None:
        return "N/D"
    return f"{currency} {value:,.2f}"


def fmt_ratio(value):
    if value is None:
        return "N/D"
    return f"{value:.2f}"


def fmt_multiple(value):
    if value is None:
        return "N/D"
    return f"{value:.2f}x"


def fmt_pct_auto(value):
    # yfinance às vezes retorna dividend yield/ROE como fração (0.045) e
    # às vezes já como número percentual (4.5), dependendo do ticker/versão.
    if value is None:
        return "N/D"
    pct = value if abs(value) > 1 else value * 100
    return f"{pct:.2f}%"


def price_delta(price, previous_close):
    if price is None or not previous_close:
        return None
    change = price - previous_close
    change_pct = (price / previous_close - 1) * 100
    return f"{change:+.2f} ({change_pct:+.2f}%)"


# ---------------------------------------------------------------------------
# Shared UI pieces
# ---------------------------------------------------------------------------

def inject_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, {NAVY_DARK} 0%, #0d264a 100%);
        }}
        [data-testid="stSidebar"] {{
            background-color: #081527;
            border-right: 1px solid rgba(201,162,39,0.25);
        }}
        [data-testid="stSidebar"] * {{
            color: #e8edf5;
        }}
        .requin-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 28px;
            margin-bottom: 20px;
            border-radius: 10px;
            background: linear-gradient(90deg, #0f2747 0%, #163460 60%, #0f2747 100%);
            border: 1px solid rgba(201,162,39,0.35);
        }}
        .requin-header h1 {{
            font-family: Georgia, 'Times New Roman', serif;
            letter-spacing: 0.04em;
            color: #f4f6fa;
            margin: 0;
            font-size: 30px;
        }}
        .requin-header span.tag {{
            color: {GOLD};
            font-weight: 700;
        }}
        .requin-header p {{
            margin: 4px 0 0 0;
            color: #9fb1cc;
            font-size: 13px;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}
        [data-testid="stMetric"] {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(201,162,39,0.18);
            border-radius: 10px;
            padding: 10px 14px;
        }}
        [data-testid="stMetricLabel"] {{
            color: #9fb1cc !important;
        }}
        h2, h3, h4 {{
            font-family: Georgia, 'Times New Roman', serif;
            color: #f4f6fa !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="requin-header">
            <div>
                <h1>Dashboard <span class="tag">RV</span> Requin</h1>
                <p>Renda Variável · Fundos Imobiliários · Macroeconomia</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_price_chart(hist: pd.DataFrame, title: str, currency: str, chart_type: str) -> go.Figure:
    if chart_type == "Candle":
        data = [
            go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                increasing_line_color=UP_COLOR,
                decreasing_line_color=DOWN_COLOR,
                name=title,
            )
        ]
    elif chart_type == "Barra":
        data = [go.Bar(x=hist.index, y=hist["Close"], marker_color=GOLD, name=title)]
    else:
        data = [go.Scatter(x=hist.index, y=hist["Close"], mode="lines", line=dict(color=BLUE_ACCENT, width=2), name=title)]

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        xaxis_title="Data/Hora",
        yaxis_title=f"Preço ({currency})",
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0d1b2e",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_watchlist(session_key: str, title: str, extra_fields=None):
    st.subheader(title)

    with st.form(f"form_{session_key}", clear_on_submit=True):
        c1, c2 = st.columns([4, 1])
        new_ticker = c1.text_input(
            "Adicionar ticker",
            key=f"input_{session_key}",
            label_visibility="collapsed",
            placeholder="Ex.: VALE3.SA",
        )
        add_clicked = c2.form_submit_button("Adicionar", width="stretch")

    if add_clicked and new_ticker.strip():
        candidate = new_ticker.strip().upper()
        if candidate not in st.session_state[session_key]:
            st.session_state[session_key].append(candidate)

    watchlist = st.session_state[session_key]
    if not watchlist:
        st.caption("Nenhum ativo na lista. Adicione um ticker acima.")
        return

    n_cols = 4
    for i in range(0, len(watchlist), n_cols):
        row_tickers = watchlist[i : i + n_cols]
        cols = st.columns(n_cols)
        for col, w_ticker in zip(cols, row_tickers):
            with col:
                with st.container(border=True):
                    try:
                        fund = fetch_fundamentals(w_ticker)
                        st.metric(
                            w_ticker,
                            fmt_currency(fund["price"], fund["currency"]),
                            delta=price_delta(fund["price"], fund["previous_close"]),
                        )
                        for label, key, formatter in extra_fields or []:
                            st.caption(f"{label}: {formatter(fund[key])}")
                    except Exception:
                        st.metric(w_ticker, "N/D")
                    if st.button("Remover", key=f"remove_{session_key}_{w_ticker}", width="stretch"):
                        st.session_state[session_key].remove(w_ticker)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def render_acoes(ticker, period, auto_refresh, refresh_seconds):
    @st.fragment(run_every=refresh_seconds if auto_refresh else None)
    def _render():
        if not ticker:
            st.warning("Informe um ticker.")
            return

        try:
            hist = fetch_history(ticker, period)
            fund = fetch_fundamentals(ticker)
        except Exception as exc:
            st.error(f"Erro ao buscar {ticker}: {exc}")
            return

        if hist.empty:
            st.error(f"Sem dados para {ticker} no período selecionado.")
            return

        row1 = st.columns(3)
        row1[0].metric("Preço atual", fmt_currency(fund["price"], fund["currency"]), delta=price_delta(fund["price"], fund["previous_close"]))
        row1[1].metric("P/L", fmt_ratio(fund["pe"]))
        row1[2].metric("Dividend Yield", fmt_pct_auto(fund["dividend_yield"]))

        row2 = st.columns(3)
        row2[0].metric("ROE", fmt_pct_auto(fund["roe"]))
        row2[1].metric("EV/EBITDA", fmt_multiple(fund["ev_ebitda"]))
        row2[2].metric("Dívida Líq./EBITDA", fmt_multiple(fund["net_debt_ebitda"]))

        chart_type = st.radio("Tipo de gráfico", CHART_TYPES, horizontal=True, key="acoes_chart_type")
        fig = build_price_chart(hist, f"{fund['name']} ({ticker})", fund["currency"], chart_type)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Atualizado em {datetime.now().strftime('%H:%M:%S')}")

        st.divider()
        render_watchlist(
            "watchlist_acoes",
            "👀 Outras ações",
            extra_fields=[
                ("DY", "dividend_yield", fmt_pct_auto),
                ("P/L", "pe", fmt_ratio),
                ("ROE", "roe", fmt_pct_auto),
                ("EV/EBITDA", "ev_ebitda", fmt_multiple),
                ("Dív.Líq./EBITDA", "net_debt_ebitda", fmt_multiple),
            ],
        )

    _render()


def render_fiis(ticker, period, auto_refresh, refresh_seconds):
    @st.fragment(run_every=refresh_seconds if auto_refresh else None)
    def _render():
        if not ticker:
            st.warning("Informe um ticker.")
            return

        try:
            hist = fetch_history(ticker, period)
            fund = fetch_fundamentals(ticker)
        except Exception as exc:
            st.error(f"Erro ao buscar {ticker}: {exc}")
            return

        if hist.empty:
            st.error(f"Sem dados para {ticker} no período selecionado.")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Preço atual", fmt_currency(fund["price"], fund["currency"]), delta=price_delta(fund["price"], fund["previous_close"]))
        col2.metric("Dividend Yield", fmt_pct_auto(fund["dividend_yield"]))
        col3.metric("P/VP", fmt_ratio(fund["price_to_book"]))

        chart_type = st.radio("Tipo de gráfico", CHART_TYPES, horizontal=True, key="fii_chart_type")
        fig = build_price_chart(hist, f"{fund['name']} ({ticker})", fund["currency"], chart_type)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Atualizado em {datetime.now().strftime('%H:%M:%S')}")

        st.divider()
        render_watchlist(
            "watchlist_fiis",
            "👀 Outros FIIs",
            extra_fields=[
                ("DY", "dividend_yield", fmt_pct_auto),
                ("P/VP", "price_to_book", fmt_ratio),
            ],
        )

    _render()


def render_macro():
    st.subheader("🌎 Panorama Macroeconômico — Brasil")
    st.caption(
        "Fonte: séries temporais do Banco Central (SGS) e expectativas de mercado do Boletim Focus. "
        "A expectativa mostrada é a mediana das projeções para o ano corrente."
    )

    cols = st.columns(len(MACRO_INDICATORS))
    for col, ind in zip(cols, MACRO_INDICATORS):
        with col:
            try:
                hist = fetch_bcb_series(ind["code"], n=5)
                latest = hist.iloc[-1]
                st.metric(ind["label"], f"{latest['valor']:.2f} {ind['unit']}")
                st.caption(f"Ref.: {latest['data']:%d/%m/%Y} · série {ind['freq']}")
            except Exception as exc:
                st.metric(ind["label"], "N/D")
                st.caption(f"Erro ao buscar série: {exc}")

            if ind["focus"]:
                try:
                    expectation = fetch_focus_expectation(ind["focus"])
                    if expectation is not None:
                        st.caption(f"Expectativa {datetime.now().year}: **{expectation:.2f} {ind['unit']}**")
                    else:
                        st.caption("Expectativa: N/D")
                except Exception:
                    st.caption("Expectativa: N/D")

    st.divider()
    st.markdown("#### Histórico interativo")
    selected = st.selectbox(
        "Selecione o indicador para visualizar o histórico",
        options=list(MACRO_HISTORY_OPTIONS.keys()),
        key="macro_history_select",
    )
    spec = MACRO_HISTORY_OPTIONS[selected]

    try:
        if spec["type"] == "sgs":
            hist = fetch_bcb_history(spec["code"], years_back=spec["years_back"], n_tail=spec["n_tail"])
        else:
            hist = fetch_ipca_ytd(n_years=spec["n_years"])

        fig = go.Figure(go.Scatter(x=hist["data"], y=hist["valor"], mode="lines", line=dict(color=GOLD, width=2)))
        fig.update_layout(
            title=f"{selected} ({spec['unit']})",
            template="plotly_dark",
            height=420,
            margin=dict(l=30, r=20, t=50, b=30),
            plot_bgcolor="#0d1b2e",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.error(f"Erro ao buscar série: {exc}")


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

st.session_state.setdefault("watchlist_acoes", ["VALE3.SA", "ITUB4.SA", "BBDC4.SA"])
st.session_state.setdefault("watchlist_fiis", ["HGLG11.SA", "KNRI11.SA", "XPLG11.SA"])

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

inject_theme()
render_header()

with st.sidebar:
    st.header("Navegação")
    section = st.radio("Seção", ["📈 Ações", "🏢 FIIs", "🌎 Macroeconomia"], key="section")
    st.divider()

    if section == "📈 Ações":
        st.subheader("Controles — Ações")
        acoes_ticker = st.text_input("Ticker", value="PETR4.SA", key="acoes_ticker").strip().upper()
        acoes_period = st.selectbox(
            "Período", options=list(PERIOD_INTERVAL_MAP.keys()), format_func=lambda p: PERIOD_LABELS[p], key="acoes_period"
        )
        acoes_auto_refresh = st.checkbox("Auto-atualizar", value=False, key="acoes_auto_refresh")
        acoes_refresh_seconds = st.slider(
            "A cada (s)", min_value=5, max_value=120, value=30, step=5, disabled=not acoes_auto_refresh, key="acoes_refresh_seconds"
        )
        st.caption("Ações da B3 usam sufixo .SA (ex.: PETR4.SA, VALE3.SA). Ações dos EUA usam o ticker puro (ex.: AAPL).")

    elif section == "🏢 FIIs":
        st.subheader("Controles — FIIs")
        fii_ticker = st.text_input("Ticker do FII", value="MXRF11.SA", key="fii_ticker").strip().upper()
        fii_period = st.selectbox(
            "Período", options=list(PERIOD_INTERVAL_MAP.keys()), format_func=lambda p: PERIOD_LABELS[p], key="fii_period"
        )
        fii_auto_refresh = st.checkbox("Auto-atualizar", value=False, key="fii_auto_refresh")
        fii_refresh_seconds = st.slider(
            "A cada (s)", min_value=5, max_value=120, value=30, step=5, disabled=not fii_auto_refresh, key="fii_refresh_seconds"
        )
        st.caption("Fundos imobiliários da B3 usam sufixo .SA (ex.: MXRF11.SA, HGLG11.SA).")

    else:
        st.subheader("Macroeconomia")
        st.caption("Selic, IPCA, dívida/PIB, câmbio e variação do PIB, com expectativas do Boletim Focus.")

st.caption(
    "Dados de ações/FIIs via Yahoo Finance (yfinance), normalmente com atraso de ~15 minutos — "
    "não é cotação institucional em tempo real."
)

if section == "📈 Ações":
    render_acoes(acoes_ticker, acoes_period, acoes_auto_refresh, acoes_refresh_seconds)
elif section == "🏢 FIIs":
    render_fiis(fii_ticker, fii_period, fii_auto_refresh, fii_refresh_seconds)
else:
    render_macro()

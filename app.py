"""Strategy Lab: an options strategy lab and a credit structuring lab.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Strategy Lab", page_icon=":material/insights:", layout="wide")

# Keep sidebar and input values when switching pages (Streamlit forgets the state of
# widgets that are not on screen unless it is re-saved on every run).
PERSIST = [
    # options lab
    "spot", "rate", "div", "atm", "smile_shape", "skew", "smile", "sticky", "mult_choice", "mult_custom", "lots",
    "strategy", "days", "budget", "basis", "stress_rng", "hedge_mult", "grid_rng", "grid_vols", "pay_vol",
    # credit lab
    "cs_preset", "cr_pd", "cr_rho", "cr_lgd", "cr_T", "cr_r", "cr_mode", "cr_n", "cr_sims", "cr_lumpy", "as_mezz", "as_senior",
    "wf_bal", "wf_wac", "wf_term", "wf_amort", "wf_cpr", "wf_cdr", "wf_sev", "wf_lag", "wf_fee",
    "srt_ead", "srt_pd", "srt_lgd", "srt_m", "srt_ac", "srt_n", "srt_mt", "srt_sts", "srt_cap", "srt_coe",
]
for _k in PERSIST:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

# Slightly smaller metric values so long numbers ("Unlimited", two breakevens) fit their columns.
st.markdown("<style>[data-testid='stMetricValue'] {font-size: 1.7rem;}</style>", unsafe_allow_html=True)

pages = [
    st.Page("views/strategy_lab.py", title="Options strategy lab", icon=":material/show_chart:", default=True),
    st.Page("views/credit_lab.py", title="Credit structuring", icon=":material/layers:"),
]
st.navigation(pages).run()

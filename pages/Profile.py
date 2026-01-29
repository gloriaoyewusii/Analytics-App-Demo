# pages/Profile.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Profile", layout="wide")
DATA_DIR = Path("data")


# -----------------------
# Helpers
# -----------------------
def require_file(filename: str) -> Path:
    path = DATA_DIR / filename
    if not path.exists():
        st.error(f"Missing file: {path}. Put **{filename}** inside your **data/** folder.")
        st.stop()
    return path


def get_query_bvn() -> str | None:
    bvn = st.query_params.get("bvn")
    # Some Streamlit versions return list instead of str
    if isinstance(bvn, list):
        bvn = bvn[0] if bvn else None
    if not bvn:
        return None
    return str(bvn).strip()


def fmt_naira(x):
    if pd.isna(x):
        return ""
    try:
        return f"₦{float(x):,.0f}"
    except Exception:
        return ""


def fmt_ratio(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""


# -----------------------
# Loaders
# -----------------------
@st.cache_data
def load_people():
    return pd.read_csv(require_file("nigeria_people_dataset.csv"), dtype={"bvn": str})


@st.cache_data
def load_demographics():
    return pd.read_csv(require_file("people_demographics_dataset.csv"), dtype={"bvn": str})


@st.cache_data
def load_accounts():
    # keep account_id as string if present
    df = pd.read_csv(require_file("accounts_dataset.csv"), dtype={"bvn": str, "account_id": str})
    return df


@st.cache_data
def load_transactions():
    df = pd.read_csv(
        require_file("transactions_dataset.csv"),
        dtype={"bvn": str, "account_id": str, "transaction_id": str},
    )
    df["txn_datetime"] = pd.to_datetime(df.get("txn_datetime"), errors="coerce")
    # Make sure amount is numeric
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


@st.cache_data
def load_monthly():
    df = pd.read_csv(require_file("spend_metrics_monthly_dataset.csv"), dtype={"bvn": str})
    # numeric columns used for charts/math
    for c in ["total_inflows", "total_outflows", "spending_ratio", "net_cashflow", "state_avg_spending_ratio"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data
def load_employment():
    return pd.read_csv(require_file("employment_history_dataset.csv"), dtype={"bvn": str, "employer_id": str})


@st.cache_data
def load_employers():
    return pd.read_csv(require_file("employers_dataset.csv"), dtype={"employer_id": str})


@st.cache_data
def load_risk():
    df = pd.read_csv(require_file("risk_flags_dataset.csv"), dtype={"bvn": str})
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    return df


# -----------------------
# Load data
# -----------------------
people = load_people()
demo = load_demographics()
accounts = load_accounts()
txns = load_transactions()
monthly = load_monthly()
employment = load_employment()
employers = load_employers()
risk = load_risk()

st.title("Individual Profile")

# -----------------------
# BVN from query param
# -----------------------
bvn = get_query_bvn()
if not bvn:
    st.warning("No BVN selected. Go back to Individuals and open a profile.")
    if st.button("← Back to Individuals"):
        st.switch_page("pages/Individuals.py")
    st.stop()

# -----------------------
# Person records
# -----------------------
p_df = people.loc[people["bvn"] == bvn]
d_df = demo.loc[demo["bvn"] == bvn]
r_df = risk.loc[risk["bvn"] == bvn]

if p_df.empty:
    st.error("BVN not found in nigeria_people_dataset.csv")
    if st.button("← Back to Individuals"):
        st.switch_page("pages/Individuals.py")
    st.stop()

p = p_df.iloc[0]
d = d_df.iloc[0] if not d_df.empty else None
r = r_df.iloc[0] if not r_df.empty else None

# Required columns (per your confirmation)
# If these are missing, give a clear error.
for col in ["state_of_residence", "local_government_area", "source_of_income"]:
    if col not in people.columns:
        st.error(f"Missing required column in people dataset: **{col}**")
        st.stop()

state_res = str(p.get("state_of_residence", "")).strip()
lga_res = str(p.get("local_government_area", "")).strip()
income_src = str(p.get("source_of_income", "")).strip()

# Safe name (only shown here in profile)
if d is not None:
    first = str(d.get("first_name", "")).strip()
    last = str(d.get("last_name", "")).strip()
    name = (f"{first} {last}").strip() or "Unknown Name"
    emp_status = str(d.get("employment_status", "")).strip()
else:
    name = "Unknown Name"
    emp_status = ""

# -----------------------
# Header card
# -----------------------
left, right = st.columns([2, 1], vertical_alignment="top")

with left:
    st.subheader(name)
    st.caption(f"BVN: {bvn}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("State (Residence)", state_res or "—")
    c2.metric("LGA (Residence)", lga_res or "—")
    c3.metric("Source of Income", income_src or "—")
    c4.metric("Employment Status", emp_status or "—")

with right:
    if r is not None:
        st.metric("Risk Status", str(r.get("status", "")))
        rs = r.get("risk_score", None)
        st.metric("Risk Score", f"{float(rs):.1f}" if pd.notna(rs) else "N/A")
        st.caption(str(r.get("flag_reason", "")))
    else:
        st.info("No risk flag record for this BVN.")

st.divider()

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["Financial Overview", "Transactions", "Accounts", "Employment"])


# -----------------------
# TAB 1: Financial Overview (NO line charts)
# -----------------------
with tab1:
    st.subheader("Financial Overview (Simple Visuals)")

    m = monthly[monthly["bvn"] == bvn].copy()
    if m.empty:
        st.warning("No monthly metrics found for this BVN.")
    else:
        m = m.sort_values("year_month")

        # Big totals (easy)
        total_inflows = float(m["total_inflows"].sum()) if "total_inflows" in m.columns else 0.0
        total_outflows = float(m["total_outflows"].sum()) if "total_outflows" in m.columns else 0.0
        avg_ratio = float(m["spending_ratio"].dropna().mean()) if "spending_ratio" in m.columns else float("nan")

        k1, k2, k3 = st.columns(3)
        k1.metric("Total Inflows", fmt_naira(total_inflows))
        k2.metric("Total Outflows", fmt_naira(total_outflows))
        k3.metric("Avg Spending Ratio", fmt_ratio(avg_ratio) if pd.notna(avg_ratio) else "N/A")

        st.divider()

        # A) Inflows vs Outflows (bar)
        st.markdown("### Inflows vs Outflows (Year-to-date)")
        totals = pd.DataFrame(
            {"Type": ["Inflows", "Outflows"], "Amount": [total_inflows, total_outflows]}
        )
        fig = px.bar(totals, x="Type", y="Amount", text="Amount")
        fig.update_traces(texttemplate="₦%{text:,.0f}", textposition="outside")
        fig.update_layout(yaxis_title="Amount (₦)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # B) Benchmarks (Individual vs LGA avg vs State avg) - SAFE merge
        st.markdown("### Benchmarks (Individual vs LGA vs State)")

        monthly_plus = monthly.merge(
            people[["bvn", "state_of_residence", "local_government_area"]]
            .rename(columns={"state_of_residence": "res_state", "local_government_area": "res_lga"}),
            on="bvn",
            how="left",
        )

        # same months as person
        person_months = set(m["year_month"].astype(str).tolist())
        bench = monthly_plus[monthly_plus["year_month"].astype(str).isin(person_months)].copy()

        # normalize for matching
        bench["res_state"] = bench["res_state"].astype(str).str.strip()
        bench["res_lga"] = bench["res_lga"].astype(str).str.strip()

        state_df = bench[bench["res_state"] == state_res].copy()
        lga_df = bench[(bench["res_state"] == state_res) & (bench["res_lga"] == lga_res)].copy()

        def agg_group(df: pd.DataFrame, label: str):
            if df.empty:
                return {"Group": label, "Avg Outflows": 0.0, "Avg Spending Ratio": 0.0}
            per_person_outflows = df.groupby("bvn")["total_outflows"].sum()
            return {
                "Group": label,
                "Avg Outflows": float(per_person_outflows.mean()),
                "Avg Spending Ratio": float(df["spending_ratio"].mean()),
            }

        bench_table = pd.DataFrame(
            [
                {
                    "Group": "Individual",
                    "Avg Outflows": float(m["total_outflows"].sum()),
                    "Avg Spending Ratio": float(m["spending_ratio"].mean()),
                },
                agg_group(lga_df, "LGA Average"),
                agg_group(state_df, "State Average"),
            ]
        )

        # Outflows bar
        fig_out = px.bar(bench_table, x="Group", y="Avg Outflows", text="Avg Outflows")
        fig_out.update_traces(texttemplate="₦%{text:,.0f}", textposition="outside")
        fig_out.update_layout(yaxis_title="Outflows (₦)", xaxis_title="")
        st.plotly_chart(fig_out, use_container_width=True)

       

        st.divider()

        # C) Spending breakdown donut (categories)
        st.markdown("### Spending Category Breakdown")

        t_person = txns[txns["bvn"] == bvn].copy()
        t_person = t_person.dropna(subset=["txn_datetime"])

        outflows = t_person[t_person["direction"].astype(str).str.lower() == "outflow"].copy()
        if outflows.empty:
            st.info("No outflow transactions available for category breakdown.")
        else:
            by_cat = (
                outflows.groupby("category", as_index=False)["amount"]
                .sum()
                .sort_values("amount", ascending=False)
            )

            top_n = 8
            top = by_cat.head(top_n).copy()
            other_sum = float(by_cat["amount"].iloc[top_n:].sum())
            if other_sum > 0:
                top = pd.concat([top, pd.DataFrame([{"category": "Other", "amount": other_sum}])], ignore_index=True)

            fig_cat = px.pie(top, names="category", values="amount", hole=0.45)
            fig_cat.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_cat, use_container_width=True)

        # D) Channel mix donut
        st.markdown("### Spending Channel")

        if outflows.empty:
            st.info("No outflow transactions available for channel mix.")
        else:
            by_channel = (
                outflows.groupby("channel", as_index=False)["amount"]
                .sum()
                .sort_values("amount", ascending=False)
            )

            top_n = 6
            top = by_channel.head(top_n).copy()
            other_sum = float(by_channel["amount"].iloc[top_n:].sum())
            if other_sum > 0:
                top = pd.concat([top, pd.DataFrame([{"channel": "Other", "amount": other_sum}])], ignore_index=True)

            fig_ch = px.pie(top, names="channel", values="amount", hole=0.45)
            fig_ch.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_ch, use_container_width=True)

        st.divider()

        # Monthly summary table (pretty headers + ₦ formatting)
        st.markdown("### Monthly Summary Table")

        cols = [
            "year_month",
            "total_inflows",
            "total_outflows",
            "spending_ratio",
            "net_cashflow",
            "top_spend_category",
            "state_avg_spending_ratio",
        ]
        cols = [c for c in cols if c in m.columns]

        show_m = m[cols].copy().rename(
            columns={
                "year_month": "Year-Month",
                "total_inflows": "Total Inflows (₦)",
                "total_outflows": "Total Outflows (₦)",
                "spending_ratio": "Spending Ratio",
                "net_cashflow": "Net Cashflow (₦)",
                "top_spend_category": "Top Spend Category",
                "state_avg_spending_ratio": "State Avg Spending Ratio",
            }
        )

        st.dataframe(
            show_m.style.format(
                {
                    "Total Inflows (₦)": fmt_naira,
                    "Total Outflows (₦)": fmt_naira,
                    "Net Cashflow (₦)": fmt_naira,
                    "Spending Ratio": fmt_ratio,
                    "State Avg Spending Ratio": fmt_ratio,
                }
            ),
            use_container_width=True,
        )


# -----------------------
# TAB 2: Transactions
# -----------------------
with tab2:
    st.subheader("Transactions")

    t = txns[txns["bvn"] == bvn].copy()
    if t.empty:
        st.warning("No transactions found for this BVN.")
    else:
        t = t.dropna(subset=["txn_datetime"]).sort_values("txn_datetime", ascending=False)

        # Date filters
        min_dt = t["txn_datetime"].min()
        max_dt = t["txn_datetime"].max()

        cA, cB, cC = st.columns([1, 1, 1])
        with cA:
            start = st.date_input("From", value=min_dt.date())
        with cB:
            end = st.date_input("To", value=max_dt.date())
        with cC:
            direction = st.selectbox("Direction", ["All", "inflow", "outflow"])

        mask = (t["txn_datetime"].dt.date >= start) & (t["txn_datetime"].dt.date <= end)
        t = t[mask]

        if direction != "All":
            t = t[t["direction"].astype(str).str.lower() == direction]

        categories = ["All"] + sorted(t["category"].dropna().unique().tolist())
        cat = st.selectbox("Category", categories)
        if cat != "All":
            t = t[t["category"] == cat]

        t = t.sort_values("txn_datetime", ascending=False)

        st.caption(f"Showing **{len(t):,}** transactions (display limited to 300 rows).")

        show_t = (
            t[
                [
                    "txn_datetime",
                    "direction",
                    "amount",
                    "category",
                    "channel",
                    "merchant_name",
                    "narration",
                    "account_id",
                ]
            ]
            .head(300)
            .copy()
            .rename(
                columns={
                    "txn_datetime": "Date/Time",
                    "direction": "Direction",
                    "amount": "Amount (₦)",
                    "category": "Category",
                    "channel": "Channel",
                    "merchant_name": "Merchant",
                    "narration": "Narration",
                    "account_id": "Account ID",
                }
            )
        )

        st.dataframe(
            show_t.style.format({"Amount (₦)": fmt_naira}),
            use_container_width=True,
            height=520,
        )


# -----------------------
# TAB 3: Accounts
# -----------------------
with tab3:
    st.subheader("Accounts")

    a = accounts[accounts["bvn"] == bvn].copy()
    if a.empty:
        st.warning("No accounts found.")
    else:
        cols = [c for c in ["bank_name", "masked_account_number", "account_type", "currency", "opened_date", "status"] if c in a.columns]
        show_a = a[cols].copy().rename(
            columns={
                "bank_name": "Bank",
                "masked_account_number": "Account (Masked)",
                "account_type": "Account Type",
                "currency": "Currency",
                "opened_date": "Opened Date",
                "status": "Account Status",
            }
        )
        st.dataframe(show_a, use_container_width=True)


# -----------------------
# TAB 4: Employment
# -----------------------
with tab4:
    st.subheader("Employment History")

    e = employment[employment["bvn"] == bvn].copy()
    if e.empty:
        st.info("No employment records found (possible in demo).")
    else:
        e = e.merge(employers, on="employer_id", how="left")

        if "start_date" in e.columns:
            e["start_date"] = pd.to_datetime(e["start_date"], errors="coerce").dt.date
        if "end_date" in e.columns:
            e["end_date"] = pd.to_datetime(e["end_date"].replace("", pd.NA), errors="coerce").dt.date

        sort_cols = [c for c in ["is_current", "start_date"] if c in e.columns]
        if sort_cols:
            e = e.sort_values(sort_cols, ascending=[False] * len(sort_cols))

        cols = [c for c in ["employer_name", "industry", "job_title", "employment_type", "start_date", "end_date", "is_current", "work_state", "work_lga"] if c in e.columns]
        show_e = e[cols].copy().rename(
            columns={
                "employer_name": "Employer",
                "industry": "Industry",
                "job_title": "Job Title",
                "employment_type": "Employment Type",
                "start_date": "Start Date",
                "end_date": "End Date",
                "is_current": "Current?",
                "work_state": "Work State",
                "work_lga": "Work LGA",
            }
        )

        st.dataframe(show_e, use_container_width=True)


# -----------------------
# Back button
# -----------------------
st.divider()
if st.button("← Back to Individuals"):
    st.switch_page("pages/Individuals.py")

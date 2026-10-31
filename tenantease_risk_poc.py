"""
TenantEase rent-risk PoC (Cynbit AI/ML intern task)

    streamlit run tenantease_risk_poc.py     -> interactive demo
    python tenantease_risk_poc.py            -> prints results in the terminal

Needs: pandas numpy scikit-learn streamlit   (anthropic is optional)

All data here is synthetic. Cynbit's real TenantEase data isn't public, so this
shows the pipeline and the idea. The numbers are not a claim about real tenants.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

NUMERIC = ["rent", "tenure_months", "deposit_months", "id_verified", "festive_month",
           "reminders_last_month", "autopay", "late_count_6m", "avg_days_late"]
CATS = ["pay_method", "occupation"]
NAMES = ["Aarav", "Riya", "Mohit", "Priya", "Kunal", "Sneha", "Vikram", "Anjali", "Rahul",
         "Pooja", "Deepak", "Neha", "Suresh", "Kavita", "Amit", "Meena", "Rohan", "Isha"]


# ---------- data ----------
def make_data(n=4000, seed=42):
    r = np.random.default_rng(seed)
    sig = lambda x: 1 / (1 + np.exp(-x))
    d = r.normal(size=n)  # hidden "how reliable is this person", never shown to the model

    df = pd.DataFrame({
        "tenant_id": [f"T-{i:04d}" for i in range(1, n + 1)],
        "name": r.choice(NAMES, n),
        "rent": r.choice([4000, 6000, 8000, 12000, 18000, 25000], n, p=[.15, .25, .25, .2, .1, .05]),
        "tenure_months": r.integers(1, 61, n),
        "occupation": r.choice(["salaried", "self_employed", "student", "other"], n, p=[.5, .25, .15, .1]),
        "pay_method": r.choice(["upi", "bank", "cash", "cheque"], n, p=[.45, .2, .25, .1]),
        "deposit_months": r.choice([1, 2, 3], n, p=[.3, .5, .2]),
        "id_verified": r.choice([0, 1], n, p=[.2, .8]),
        "festive_month": r.choice([0, 1], n, p=[.8, .2]),
        "reminders_last_month": r.integers(0, 4, n),
        "due_day": r.choice([1, 5, 10], n),
    })
    df["autopay"] = (df.pay_method.isin(["upi", "bank"]) & (r.random(n) < .4)).astype(int)
    df["late_count_6m"] = r.binomial(6, sig(-1.3 + 0.9 * d))
    df["avg_days_late"] = np.where(df.late_count_6m > 0,
                                   r.gamma(2, 2.0, n) * (1 + 0.2 * df.late_count_6m), 0).round(1)

    z = (-2.0 + 1.3 * d + 0.9 * (df.pay_method == "cash") - 1.3 * df.autopay
         + 0.8 * (df.occupation == "self_employed") + 1.0 * (df.occupation == "student")
         + 0.8 * df.festive_month + 0.5 * (df.rent >= 18000) - 0.5 * df.id_verified
         + 0.4 * (df.tenure_months < 6) - 0.15 * (df.deposit_months - 1))
    df["late_next"] = (r.random(n) < sig(z)).astype(int)
    return df


def encode(df):
    return pd.get_dummies(df[NUMERIC + CATS], columns=CATS).astype(float)


def group_of(col):
    return next((c for c in CATS if col.startswith(c + "_")), col)


# ---------- model ----------
def build(n=4000):
    df = make_data(n)
    X, y = encode(df), df.late_next
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=4,
                                           random_state=1).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    naive = df.loc[Xte.index, "late_count_6m"].values  # "just look at past lateness" rule

    k = int(len(yte) * 0.2)
    def top_k(score):
        hit = yte.values[np.argsort(-score, kind="stable")[:k]]
        return hit.mean(), hit.sum() / yte.sum()

    prec, rec = top_k(p)
    nprec, nrec = top_k(naive)
    metrics = pd.DataFrame({
        "Past-lateness rule": [roc_auc_score(yte, naive), average_precision_score(yte, naive), nprec, nrec],
        "Gradient boosting": [roc_auc_score(yte, p), average_precision_score(yte, p), prec, rec],
    }, index=["ROC-AUC", "PR-AUC", "Precision @ top 20%", "Recall @ top 20%"]).round(3)

    return dict(model=model, med=Xtr.median(), X=Xte, metrics=metrics, base_rate=float(y.mean()),
                portfolio=df.loc[Xte.index].assign(risk=p))


def band(p):
    return "high" if p >= 0.5 else "medium" if p >= 0.3 else "low"


LABELS = {
    "late_count_6m": lambda r: f"paid late in {r.late_count_6m} of the last 6 months",
    "avg_days_late": lambda r: f"averages {r.avg_days_late} days late when late",
    "pay_method": lambda r: f"pays by {r.pay_method}",
    "autopay": lambda r: "autopay is on" if r.autopay else "no autopay",
    "occupation": lambda r: f"occupation: {r.occupation.replace('_', ' ')}",
    "tenure_months": lambda r: f"only {r.tenure_months} months in the unit" if r.tenure_months < 12
                               else f"{r.tenure_months} months in the unit",
    "festive_month": lambda r: "festive month" if r.festive_month else "regular month",
    "id_verified": lambda r: "ID verified" if r.id_verified else "ID not verified",
    "rent": lambda r: f"rent is Rs {r.rent:,}",
    "deposit_months": lambda r: f"{r.deposit_months}-month deposit",
    "reminders_last_month": lambda r: f"{r.reminders_last_month} reminders needed last month",
}


def reasons(A, idx, k=3):
    """Local explanation: reset one feature group to the typical value and see how far the risk drops."""
    x = A["X"].loc[idx]
    row = A["portfolio"].loc[idx]
    groups = {}
    for c in x.index:
        groups.setdefault(group_of(c), []).append(c)
    variants = pd.DataFrame([x] * len(groups))
    for i, cols in enumerate(groups.values()):
        variants.iloc[i, [variants.columns.get_loc(c) for c in cols]] = A["med"][cols].values
    drops = row.risk - A["model"].predict_proba(variants)[:, 1]
    top = sorted(zip(groups, drops), key=lambda t: -t[1])[:k]
    return [LABELS.get(g, lambda r: g)(row) for g, d in top if d > 0.005]


# ---------- reminders ----------
def reminder(row, b, lang="English"):
    name, amt, day = row["name"], f"Rs {int(row['rent']):,}", int(row["due_day"])
    lead = {"high": 5, "medium": 3, "low": 1}[b]
    if lang == "English":
        if b == "low":
            return f"Hi {name}, a quick reminder that this month's rent of {amt} is due tomorrow (day {day}). Thanks!"
        return (f"Hi {name}, rent of {amt} is due on day {day}. Sending this {lead} days early so it's easy to plan. "
                f"If something is making it tricky this month, just reply and we'll sort it out together.")
    if b == "low":
        return f"Namaste {name} ji, kal {amt} ka rent due hai (tareekh {day}). Dhanyavaad!"
    return (f"Namaste {name} ji, {amt} ka rent {day} tareekh ko due hai. {lead} din pehle yaad dila rahe hain "
            f"taaki aap aaraam se plan kar sakein. Koi dikkat ho toh bata dijiye.")


def polish(text):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic
    resp = anthropic.Anthropic(api_key=key).messages.create(
        model=os.getenv("LLM_MODEL", "claude-haiku-5-5"), max_tokens=200,
        messages=[{"role": "user", "content": "Rewrite this rent reminder so it sounds warm and natural, "
                   "under 60 words. Keep the amount and day exactly as written. Reply with the message only.\n\n" + text}])
    return resp.content[0].text


# ---------- UI ----------
def ui():
    import streamlit as st
    st.set_page_config(page_title="TenantEase rent-risk PoC", layout="wide")
    A = st.cache_resource(build)()
    pf = A["portfolio"].copy()
    pf["band"] = pf.risk.map(band)
    pf = pf.sort_values("risk", ascending=False)

    st.title("TenantEase: who might pay late next month?")
    st.caption("Proof of concept on synthetic data. Scores the holdout tenants the model never saw in training.")
    t1, t2, t3 = st.tabs(["Portfolio", "One tenant", "How good is it?"])

    with t1:
        c = st.columns(4)
        c[0].metric("Tenants scored", len(pf))
        c[1].metric("High risk", int((pf.band == "high").sum()))
        c[2].metric("Medium risk", int((pf.band == "medium").sum()))
        c[3].metric("Rent that may slip (expected)", f"Rs {int((pf.rent * pf.risk).sum()):,}")
        st.bar_chart(pf.band.value_counts().reindex(["high", "medium", "low"]).fillna(0))
        top = pf.head(15)
        st.dataframe(pd.DataFrame({
            "Tenant": top.tenant_id, "Name": top["name"], "Rent": top.rent,
            "Late risk": top.risk.round(2),
            "Why": [", ".join(reasons(A, i)) for i in top.index]}), hide_index=True, use_container_width=True)

    with t2:
        pick = st.selectbox("Tenant (riskiest first)", pf.head(100).tenant_id)
        idx = pf.index[pf.tenant_id == pick][0]
        row = pf.loc[idx]
        st.metric("Chance of paying late", f"{row.risk:.0%}", row.band + " risk", delta_color="off")
        st.write("**What's driving it:** " + (", ".join(reasons(A, idx)) or "nothing stands out"))
        lang = st.radio("Reminder language", ["English", "Hinglish"], horizontal=True)
        msg = reminder(row, row.band, lang)
        st.info(msg)
        if st.button("Rewrite with an LLM"):
            out = polish(msg)
            st.success(out) if out else st.warning("Set ANTHROPIC_API_KEY to turn this on.")

    with t3:
        st.dataframe(A["metrics"], use_container_width=True)
        st.caption(f"Base late rate in the data: {A['base_rate']:.0%}. 'Precision @ top 20%' means: if a landlord "
                   "only chased the 20% we flag, how many of those would really have been late.")


def cli():
    A = build()
    print("Base late rate:", round(A["base_rate"], 3))
    print(A["metrics"], "\n")
    pf = A["portfolio"].sort_values("risk", ascending=False).head(5)
    for i, r in pf.iterrows():
        print(f"{r.tenant_id} {r['name']:8s} risk={r.risk:.2f} ({band(r.risk)}) <- {', '.join(reasons(A, i))}")
    print("\nSample message:", reminder(pf.iloc[0], band(pf.iloc[0].risk)))


def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _in_streamlit():
    ui()
elif __name__ == "__main__":
    cli()

"""Streamlit dashboard for the RL Trading Lab simulator."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Streamlit executes this root-level script directly; expose the local src package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_trading_lab.training import evaluate_held_out, evaluate_policy, train


st.set_page_config(page_title="RL Trading Lab", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="Training Q-learning agent and evaluating held-out paths...")
def train_and_evaluate(seed: int, episodes: int, regime: str, test_paths: int) -> dict[str, object]:
    """Train once, evaluate a display path and aggregate unseen evaluation paths."""
    agent, _ = train(episodes=episodes, seed=seed, regime=regime)  # type: ignore[arg-type]
    display, prices, policy_values, buy_and_hold_values = evaluate_policy(
        agent, seed=seed + 10_000, regime=regime  # type: ignore[arg-type]
    )
    held_out = evaluate_held_out(
        episodes=episodes,
        training_seed=seed,
        test_seeds=range(seed + 20_000, seed + 20_000 + test_paths),
        regime=regime,  # type: ignore[arg-type]
    )
    return {
        "prices": prices,
        "policy_values": policy_values,
        "buy_and_hold_values": buy_and_hold_values,
        "display": display,
        "held_out": held_out,
    }


st.title("RL Trading Lab")
st.caption("Educational Q-learning experiments on synthetic price paths — not investment advice.")

with st.sidebar:
    st.header("Experiment settings")
    seed = st.number_input("Training random seed", min_value=0, value=7, step=1)
    episodes = st.slider("Training episodes", min_value=50, max_value=2_000, value=300, step=50)
    regime = st.selectbox(
        "Synthetic market regime",
        options=("gbm", "trend", "mean_reverting"),
        help="GBM has no learnable next-day signal; trend and mean_reverting include return autocorrelation.",
    )
    test_paths = st.slider("Held-out evaluation paths", min_value=5, max_value=30, value=20, step=5)
    train_clicked = st.button("Train and evaluate", type="primary", use_container_width=True)

if train_clicked:
    run = train_and_evaluate(int(seed), episodes, regime, test_paths)
    display = run["display"]
    held_out = run["held_out"]

    st.subheader("One held-out price path")
    left, right = st.columns(2)
    left.metric("Learned policy", f"${display.policy_value:,.2f}")
    right.metric("Buy and hold", f"${display.buy_and_hold_value:,.2f}")

    price_chart = go.Figure(
        go.Scatter(y=run["prices"], mode="lines", name="Synthetic price", line={"color": "#2E7D32"})
    )
    price_chart.update_layout(title="Held-out synthetic price path", xaxis_title="Trading day", yaxis_title="Price")
    st.plotly_chart(price_chart, use_container_width=True)

    value_chart = go.Figure()
    value_chart.add_scatter(y=run["policy_values"], mode="lines", name="Learned policy")
    value_chart.add_scatter(y=run["buy_and_hold_values"], mode="lines", name="Buy and hold")
    value_chart.update_layout(
        title="Held-out equity curve", xaxis_title="Trading day", yaxis_title="Portfolio value ($)"
    )
    st.plotly_chart(value_chart, use_container_width=True)

    st.subheader(f"Held-out summary across {test_paths} paths")
    first, second, third = st.columns(3)
    first.metric("Policy mean return", f"{held_out['policy_return_mean']:.2%}")
    second.metric("Buy-and-hold mean return", f"{held_out['buy_and_hold_return_mean']:.2%}")
    third.metric("Mean excess return", f"{held_out['excess_return_mean']:.2%}")
    st.caption(
        f"Excess-return spread (population standard deviation): {held_out['excess_return_std']:.2%}. "
        "Results are synthetic and are not evidence of real-market profitability."
    )
else:
    st.info("Choose a regime and training settings, then select **Train and evaluate**.")

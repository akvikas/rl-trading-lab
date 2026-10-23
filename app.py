"""Streamlit Pro Dashboard for the RL Trading Lab Simulator.

Features:
- Dark Command Center UI with contrasting neon aesthetics
- Interactive Plotly Candlestick / Price Chart with exact BUY/SELL execution markers
- Complete Trade-by-Trade Execution Ledger with CSV Export
- Real-Time Profit & Performance Analytics (Net P&L, Alpha, Win Rate, Profit Factor, Drawdown)
- "Trader's Path to Success" Strategic Playbook and Hyperparameter Diagnostics
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Expose local package directory
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_trading_lab.environment import MarketRegime
from rl_trading_lab.training import (
    DetailedPolicyRun,
    evaluate_held_out,
    evaluate_policy_detailed,
    train,
)

# Page Configuration
st.set_page_config(
    page_title="RL Trading Lab — Cyber Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Dark UI & High-Contrast Cyber Styling
st.markdown(
    """
    <style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0a0e17;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metrics and Card Containers */
    [data-testid="stMetric"] {
        background-color: #131b2a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    [data-testid="stMetricValue"] {
        color: #00f0ff !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Accent Card Boxes */
    .cyber-card {
        background-color: #131b2a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.35);
    }
    .cyber-card-highlight {
        border-left: 4px solid #00f0ff;
    }
    .cyber-card-success {
        border-left: 4px solid #10b981;
    }
    .cyber-card-warning {
        border-left: 4px solid #f59e0b;
    }

    /* Badges */
    .badge-buy {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-sell {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.78rem;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00f0ff 0%, #3b82f6 100%) !important;
        color: #0a0e17 !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 240, 255, 0.4) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Training Q-learning agent and analyzing execution paths...")
def train_and_evaluate_simulation(
    seed: int, episodes: int, regime: MarketRegime, test_paths: int
) -> dict[str, object]:
    """Train once, generate detailed trade ledger for primary path, and evaluate held-out generalization."""
    agent, _ = train(episodes=episodes, seed=seed, regime=regime)
    detailed_run = evaluate_policy_detailed(agent, seed=seed + 10_000, regime=regime)
    held_out = evaluate_held_out(
        episodes=episodes,
        training_seed=seed,
        test_seeds=range(seed + 20_000, seed + 20_000 + test_paths),
        regime=regime,
    )
    return {
        "detailed": detailed_run,
        "held_out": held_out,
    }


# ==========================================
# Sidebar Controls & Strategy Presets
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ Strategy Command Center")
    st.caption("Reinforcement Learning Quantitative Trading Lab")

    st.markdown("---")
    preset = st.selectbox(
        "🎯 Strategy Preset",
        options=[
            "Conservative Trend Hunter",
            "Mean-Reversion Swing Scalper",
            "Capital Preservation Baseline",
            "Custom Strategy",
        ],
        index=0,
    )

    if preset == "Conservative Trend Hunter":
        default_regime = "trend"
        default_episodes = 600
        default_seed = 42
    elif preset == "Mean-Reversion Swing Scalper":
        default_regime = "mean_reverting"
        default_episodes = 700
        default_seed = 99
    elif preset == "Capital Preservation Baseline":
        default_regime = "gbm"
        default_episodes = 400
        default_seed = 7
    else:
        default_regime = "trend"
        default_episodes = 500
        default_seed = 42

    regime_choice: MarketRegime = st.selectbox(
        "Market Regime Dynamics",
        options=["trend", "mean_reverting", "gbm"],
        index=["trend", "mean_reverting", "gbm"].index(default_regime),
        help=(
            "trend: Positive return autocorrelation (momentum continues).\n"
            "mean_reverting: Negative return autocorrelation (dips bounce, rallies fade).\n"
            "gbm: Pure Geometric Brownian Motion random walk (zero next-day predictability)."
        ),
    )

    episodes = st.slider(
        "Training Episodes",
        min_value=50,
        max_value=2000,
        value=default_episodes,
        step=50,
        help="Number of epochs to train the tabular Q-learning agent with decaying epsilon exploration.",
    )

    seed = st.number_input(
        "Master Random Seed",
        min_value=0,
        max_value=999999,
        value=default_seed,
        step=1,
    )

    test_paths = st.slider(
        "Held-Out Generalization Paths",
        min_value=5,
        max_value=50,
        value=25,
        step=5,
        help="Unseen random market paths evaluated out-of-sample to test for policy generalization.",
    )

    transaction_fee_bps = st.slider(
        "Commission Friction (Basis Points)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        help="10 bps = 0.10% transaction cost applied to each buy and sell.",
    )

    st.markdown("---")
    run_sim = st.button("🚀 Train & Execute Strategy", type="primary", use_container_width=True)


# ==========================================
# Main Dashboard Header
# ==========================================
st.title("⚡ RL Trading Lab — Algorithmic Execution")
st.markdown(
    "**High-Contrast Quantitative Trading Simulator** · Tabular Q-Learning vs. Buy & Hold Benchmark with Full Trade Execution Transparency."
)

if run_sim or "sim_result" in st.session_state:
    if run_sim:
        with st.spinner("🧠 Optimizing Q-table and executing policy on unseen market path..."):
            sim_data = train_and_evaluate_simulation(int(seed), episodes, regime_choice, test_paths)
            st.session_state["sim_result"] = sim_data
            st.session_state["regime_choice"] = regime_choice

    sim_data = st.session_state["sim_result"]
    regime_choice = st.session_state.get("regime_choice", "trend")
    d: DetailedPolicyRun = sim_data["detailed"]
    held_out = sim_data["held_out"]

    # ==========================================
    # Real-Time Profit Tracking (KPI Row)
    # ==========================================
    st.markdown("### 📊 Performance & Profit Tracker")

    initial_cash = 10_000.0
    net_pnl = d.evaluation.policy_value - initial_cash
    net_pnl_pct = (net_pnl / initial_cash) * 100.0
    bnh_pnl = d.evaluation.buy_and_hold_value - initial_cash
    bnh_pnl_pct = (bnh_pnl / initial_cash) * 100.0
    alpha = net_pnl_pct - bnh_pnl_pct

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.metric(
            label="Portfolio Value",
            value=f"${d.evaluation.policy_value:,.2f}",
            delta=f"{net_pnl:+$0.2f} ({net_pnl_pct:+.2f}%)",
        )

    with kpi2:
        st.metric(
            label="Benchmark Alpha",
            value=f"{alpha:+.2f}%",
            delta="vs Buy & Hold" if alpha >= 0 else "Underperforming",
            delta_color="normal" if alpha >= 0 else "inverse",
        )

    with kpi3:
        st.metric(
            label="Trade Win Rate",
            value=f"{d.win_rate:.1f}%",
            delta=f"{d.winning_trades} Wins / {d.losing_trades} Losses",
        )

    with kpi4:
        pf_display = f"{d.profit_factor:.2f}" if d.profit_factor < 90 else "∞"
        st.metric(
            label="Profit Factor",
            value=pf_display,
            delta="Gross Win/Loss Ratio",
        )

    with kpi5:
        st.metric(
            label="Max Drawdown",
            value=f"-{d.max_drawdown:.2f}%",
            delta=f"Fees Paid: ${d.total_fees_paid:,.2f}",
            delta_color="inverse",
        )

    # ==========================================
    # High-Contrast Price Chart with Trade Markers
    # ==========================================
    st.markdown("### 📈 Price Trajectory & Execution Markers")
    st.caption("Cyan line indicates asset price. Green triangles (▲) denote BUY executions; Red triangles (▼) denote SELL liquidations.")

    price_fig = go.Figure()

    # Asset Price Path (Cyber Cyan)
    days = list(range(len(d.prices)))
    price_fig.add_trace(
        go.Scatter(
            x=days,
            y=d.prices,
            mode="lines",
            name="Asset Price ($)",
            line=dict(color="#00F0FF", width=2.2),
            hovertemplate="Day %{x}<br>Price: $%{y:.2f}<extra></extra>",
        )
    )

    # Buy Markers (Neon Emerald Green ▲)
    if d.buy_steps:
        price_fig.add_trace(
            go.Scatter(
                x=d.buy_steps,
                y=d.buy_prices,
                mode="markers",
                name="BUY Execution",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#10B981",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                hovertemplate="<b>BUY ORDER</b><br>Day: %{x}<br>Fill Price: $%{y:.2f}<extra></extra>",
            )
        )

    # Sell Markers (Neon Crimson Red ▼)
    if d.sell_steps:
        price_fig.add_trace(
            go.Scatter(
                x=d.sell_steps,
                y=d.sell_prices,
                mode="markers",
                name="SELL Execution",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#EF4444",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                hovertemplate="<b>SELL ORDER</b><br>Day: %{x}<br>Fill Price: $%{y:.2f}<extra></extra>",
            )
        )

    price_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131B2A",
        plot_bgcolor="#0A0E17",
        margin=dict(l=40, r=20, t=30, b=40),
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(19, 27, 42, 0.8)",
            bordercolor="#1E293B",
            borderwidth=1,
        ),
        xaxis=dict(
            title="Trading Day",
            gridcolor="#1E293B",
            showgrid=True,
            zerolinecolor="#334155",
        ),
        yaxis=dict(
            title="Asset Price ($)",
            gridcolor="#1E293B",
            showgrid=True,
            zerolinecolor="#334155",
        ),
    )
    st.plotly_chart(price_fig, use_container_width=True)

    # ==========================================
    # Cumulative Profit & Equity Curve Comparison
    # ==========================================
    st.markdown("### 💰 Equity Growth & Benchmark Spread")

    equity_fig = go.Figure()

    # Policy Equity Curve (Neon Green)
    equity_fig.add_trace(
        go.Scatter(
            x=days,
            y=d.policy_values,
            mode="lines",
            name="RL Policy Portfolio",
            line=dict(color="#10B981", width=3),
            hovertemplate="Day %{x}<br>Policy Value: $%{y:,.2f}<extra></extra>",
        )
    )

    # Buy & Hold Curve (Purple / Indigo)
    equity_fig.add_trace(
        go.Scatter(
            x=days,
            y=d.buy_and_hold_values,
            mode="lines",
            name="Buy & Hold Benchmark",
            line=dict(color="#8B5CF6", width=2, dash="dash"),
            hovertemplate="Day %{x}<br>Benchmark Value: $%{y:,.2f}<extra></extra>",
        )
    )

    equity_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131B2A",
        plot_bgcolor="#0A0E17",
        margin=dict(l=40, r=20, t=30, b=40),
        height=380,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(19, 27, 42, 0.8)",
            bordercolor="#1E293B",
            borderwidth=1,
        ),
        xaxis=dict(
            title="Trading Day",
            gridcolor="#1E293B",
            showgrid=True,
        ),
        yaxis=dict(
            title="Portfolio Value ($)",
            gridcolor="#1E293B",
            showgrid=True,
        ),
    )
    st.plotly_chart(equity_fig, use_container_width=True)

    # ==========================================
    # Detailed Trade Execution Ledger Table
    # ==========================================
    st.markdown("### 📜 Trade Execution Ledger")
    st.caption("Complete granular audit trail of every entry and exit signal triggered by the model.")

    if d.trades:
        trade_rows = []
        for t in d.trades:
            trade_rows.append(
                {
                    "Trade #": t.trade_id,
                    "Day": t.step,
                    "Action": "🟢 BUY" if t.action == "BUY" else "🔴 SELL",
                    "Price ($)": f"${t.price:,.2f}",
                    "Shares": t.shares,
                    "Value ($)": f"${t.cost_or_proceed:,.2f}",
                    "Fee ($)": f"${t.fee:,.2f}",
                    "Realized P&L ($)": (
                        f"{t.realized_pnl:+$0.2f}" if t.realized_pnl is not None else "—"
                    ),
                    "ROI (%)": (
                        f"{t.realized_pnl_pct:+.2f}%" if t.realized_pnl_pct is not None else "—"
                    ),
                    "Cash Balance": f"${t.cash_after:,.2f}",
                    "Portfolio Value": f"${t.portfolio_value:,.2f}",
                    "Market State": t.trend_state,
                }
            )

        df_trades = pd.DataFrame(trade_rows)
        st.dataframe(
            df_trades,
            use_container_width=True,
            hide_index=True,
        )

        # Download CSV Button
        csv_buffer = io.StringIO()
        df_trades.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Full Trade Ledger (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"rl_trade_ledger_seed_{seed}_{regime_choice}.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "🛡️ **No trades executed on this path.** The Q-learning agent determined that trading under current transaction friction would produce negative expected value, prioritizing 100% capital preservation."
        )

    # ==========================================
    # "The Trader's Path to Success" Playbook
    # ==========================================
    st.markdown("---")
    st.markdown("### 🏆 The Trader's Path to Success & Strategic Playbook")
    st.caption("Actionable quantitative insights to guide model profitability and risk management.")

    col_play1, col_play2 = st.columns(2)

    with col_play1:
        st.markdown(
            f"""
            <div class="cyber-card cyber-card-highlight">
                <h4 style="color: #00F0FF; margin-top:0;">🗺️ Market Regime Strategy Analysis</h4>
                <p><strong>Active Regime:</strong> <code>{regime_choice.upper()}</code></p>
                <p>
            """,
            unsafe_allow_html=True,
        )

        if regime_choice == "trend":
            st.markdown(
                r"""
                - **Mechanics:** Positive return autocorrelation ($\rho = +0.35$). When prices trend upward, the probability of continuation exceeds 50%.
                - **Success Path:** Train with **600+ episodes** and an $\epsilon$-decay rate of 0.995. The agent learns to execute BUY on trend breakouts and ride the wave until momentum shifts, minimizing churn while capturing the trend tail.
                - **Risk Guardrail:** Avoid excessive transaction fees. Notice how holding through short retracements yields higher alpha than high-frequency flipping.
                """
            )
        elif regime_choice == "mean_reverting":
            st.markdown(
                r"""
                - **Mechanics:** Negative return autocorrelation ($\rho = -0.35$). Sharp downward impulses represent statistical discounts that regularly revert toward the mean.
                - **Success Path:** The agent profits by buying when the discrete state detects a *Downtrend* signal ($change < -0.2\%$), and selling into *Uptrend* surges ($change > +0.2\%$).
                - **Risk Guardrail:** Ensure discount factor $\gamma \ge 0.95$ so the agent values the future reversal rather than reacting to short-term drawdown.
                """
            )
        else:
            st.markdown(
                r"""
                - **Mechanics:** Pure Geometric Brownian Motion ($\rho = 0.00$). Past price changes contain **zero** predictive information about future returns.
                - **Success Path (Capital Preservation):** In a market with zero edge and positive transaction fees (10 bps), any trade has an expected value of $-Fee$. The optimal mathematical policy is **patience and capital preservation**.
                - **Key Takeaway:** Avoiding negative-EV trades is the foundation of institutional risk management.
                """
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_play2:
        st.markdown(
            """
            <div class="cyber-card cyber-card-success">
                <h4 style="color: #10B981; margin-top:0;">⚡ Optimization Recommendations</h4>
                <p>Follow these steps to improve strategy performance:</p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            rf"""
            1. **Friction-Adjusted Alpha:** Total commissions paid were **${d.total_fees_paid:,.2f}**. {"Friction is well-managed." if d.total_fees_paid < 100 else "Consider increasing episode count to prune low-conviction trades."}
            2. **Win/Loss Asymmetry:** Gross profits: **${d.gross_profits:,.2f}** vs Gross losses: **${d.gross_losses:,.2f}**. A Profit Factor above **1.50** indicates strong edge over transaction drag.
            3. **Exploration Calibration:** If the model fails to exploit clear trends, increase training episodes to allow $\epsilon$-greedy exploration to decay below 0.05.
            4. **Out-of-Sample Discipline:** Never trust a single random path. Use the aggregate multi-path evaluation below to verify true expected value across diverse market seeds.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # Multi-Path Out-of-Sample Generalization
    # ==========================================
    st.markdown("### 🌐 Held-Out Generalization Summary")
    st.caption(f"Evaluated across {test_paths} distinct, previously unseen market seeds to guard against overfitting and data leakage.")

    sum1, sum2, sum3 = st.columns(3)
    sum1.metric("Policy Mean Return", f"{held_out['policy_return_mean']:.2%}")
    sum2.metric("Buy & Hold Mean Return", f"{held_out['buy_and_hold_return_mean']:.2%}")
    sum3.metric(
        "Mean Excess Alpha",
        f"{held_out['excess_return_mean']:.2%}",
        delta=f"Std Dev: ±{held_out['excess_return_std']:.2%}",
    )

else:
    st.info(
        "👋 **Welcome to the RL Trading Lab Cyber Command Center.**\n\n"
        "Select your strategy preset and parameters in the sidebar, then click **'🚀 Train & Execute Strategy'** to launch the simulation."
    )

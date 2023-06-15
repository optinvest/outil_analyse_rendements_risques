import streamlit as st
import plotly.express as px
from pypfopt.risk_models import cov_to_corr
from utils import download_financial_data, get_risk_return_dicts, generate_random_ptfs, construct_ptf_df, construct_ptf_plot_df, period_dict
import quantstats as qs

# Paramètres
risk_model = "exp_covariance"
returns_model = "ema_historical"
n_samples = 50
start_date = "2023-01-01"
end_date = None # "2023-06-01"
seed = 0

input_actions = st.text_input('Entrer les tickers séparés par des virgules ,', placeholder="par exemple: AAPL, AMZN, MSFT, TSLA, XOM")
with st.sidebar:
    period = st.radio('Choisir une période de temps:', list(period_dict.keys())[2:], index=4)
    period = period_dict[period]

btn_run = st.button("Lancer l'analyse")

# UI
col1, col2, col3 = st.columns(3)

if btn_run:
    actions = [ele.strip() for ele in input_actions.split(",")]
    actions = list(set([ele for ele in actions if len(ele)]))

    # pour le moment seul l'option 1 est possible pour la sélection d'une durée
    data = download_financial_data(actions=actions, period=period)
    returns = data.pct_change().dropna() # nous calculons la rentabilité des titres
    returns_model_dict, risk_model_dict = get_risk_return_dicts(data)

    mu = returns_model_dict[returns_model] 
    S = risk_model_dict[risk_model]


    w, rets, stds, sharpes = generate_random_ptfs(n_samples, actions, mu, S, seed=seed)

    renta_df = (mu * 100).to_frame("Rentabilité (annualisée) %").sort_values(by=["Rentabilité (annualisée) %"], ascending=False)
    corr_df = (cov_to_corr(S) * 100 ).style.format('{:,.2f}')
    sharpe_df = qs.stats.sharpe(returns).to_frame("Sharpe ratio").sort_values(by=["Sharpe ratio"], ascending=False)
    sortino_df = qs.stats.adjusted_sortino(returns).to_frame("Sortino ratio").sort_values(by=["Sortino ratio"], ascending=False)


    ptf_df = construct_ptf_df(data, actions, w, rets, stds, sharpes, mu, S)
    ptf_plot_df = construct_ptf_plot_df(ptf_df, actions)

    fig = px.scatter(data_frame=ptf_plot_df,
                    x="Volatilité (%)",
                    y="Rentabilité (%)",
                    color="Sharpe",
                    symbol="Type",
                    color_continuous_scale='Bluered_r',
                    hover_data=ptf_plot_df.columns)
    fig.update_layout(coloraxis_colorbar_x=-0.20)

    with col1:
        st.write("**Rentabilité individuelle**")
        st.write(renta_df)

    st.write("**Corrélations entre les titres financiers**")
    st.write(corr_df)

    with col2:
        st.write("**Ratios de Sharpe**")
        st.write(sharpe_df)

    with col3:
        st.write("**Ratios de Sortino**")
        st.write(sortino_df)

    st.write("## Graphique dynamique avec génération de portefeuilles aléatoires")
    st.plotly_chart(fig, use_container_width=False)

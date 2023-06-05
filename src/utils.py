import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return, ema_historical_return
from pypfopt.risk_models import CovarianceShrinkage, semicovariance, exp_cov
import quantstats as qs
import yfinance as yf
import streamlit as st

period_dict = {"1 jour": "1d", 
               "5 jours": "5d", 
               "1 mois": "1mo", 
               "3 mois": "3mo", 
               "6 mois": "6mo", 
               "1 an": "1y", 
               "2 ans": "2y", 
               "5 ans": "5y", 
               "10 ans": "10y", 
               "ytd": "ytd", 
               "max": "max"}

@st.cache_data
def download_financial_data(actions, period, start_date=None, end_date=None):
    if end_date is None:
        print("Téléchargement des données historiques selon la période définie en 3.1")
        # Téléchargement des prix historiques si le calendrier n'a pas été utilisé
        data = yf.download(tickers=actions,     # liste des tickers
                           auto_adjust=True,    # important : ajustement des prix selon splits et dividendes
                           period=period,       # jusqu'à quand devons-nous remonter dans le temps (1y: 1 année)
                           interval="1d",       # interval de trading
                           ignore_tz=True,      # ignorer la timezone
                           group_by='ticker',
                           prepost=False)
    else:
        print("Téléchargement des données historiques selon les dates définies en 3.2")
        data = yf.download(tickers=actions,     # liste des tickers
                           auto_adjust=True,    # important : ajustement des prix selon splits et dividendes
                           start=start_date,
                           end=end_date,
                           ignore_tz=True,      # ignorer la timezone
                           group_by='ticker',
                           prepost=False)
    data = data.filter(like="Close") # nous sélectionnons uniquement les colonnes Close (prix de clôture)
    cols = [col[0] for col in data.columns]
    data.columns = cols
    return data


def get_risk_return_dicts(data):
    returns_model_dict = {"mean_historical": mean_historical_return(data), 
                          "ema_historical": ema_historical_return(data)}

    risk_model_dict = {"covariance": CovarianceShrinkage(data).ledoit_wolf(),
                       "semi_covariance": semicovariance(data), 
                       "exp_covariance":  exp_cov(data)}
    return returns_model_dict, risk_model_dict


def generate_random_ptfs(n_samples, actions, mu, S, seed=0):
    np.random.seed(seed)
    w = np.random.dirichlet(np.ones(len(actions)), n_samples)
    rets = w.dot(mu.values)
    stds = np.sqrt(np.diag(w @ S @ w.T))
    sharpes = rets / stds
    return w, rets, stds, sharpes


def construct_ptf_df(data, actions, w, rets, stds, sharpes, mu, S):
    ptf_df = pd.DataFrame({"Return": rets, "Volatility": stds, "Sharpe": sharpes, "Type": "Allocation Aléatoire"}).join(pd.DataFrame(w, columns=actions))
    ptf_indiv_df = pd.DataFrame({"Return": mu.values, "Volatility": np.sqrt(np.diag(S)), "Sharpe": mu.values / np.sqrt(np.diag(S))}).join(pd.DataFrame(np.eye(mu.shape[0]), columns=mu.index))
    ptf_indiv_df["Type"] = mu.index
    ptf_equi_df = pd.DataFrame({"Return": mu @ np.ones(mu.shape[0]) / mu.shape[0], 
                                "Volatility": np.sqrt(qs.stats.volatility((data * np.ones(mu.shape[0]) / mu.shape[0]).sum(axis=1)))}, index=[0])
    ptf_equi_df["Sharpe"] = ptf_equi_df["Return"] / ptf_equi_df["Volatility"]
    ptf_equi_df = ptf_equi_df.join(pd.DataFrame([np.ones(mu.shape[0]) / mu.shape[0]], columns=mu.index))
    ptf_equi_df["Type"] = "Equi-réparti"
    ptf_df = pd.concat([ptf_df, ptf_indiv_df, ptf_equi_df], axis=0).reset_index(drop=True)
    return ptf_df


def construct_ptf_plot_df(ptf_df, actions, round=2):
    ptf_plot_df = ptf_df.copy()
    ptf_plot_df = ptf_plot_df.rename(columns={"Return": "Rentabilité", "Volatility": "Volatilité"})
    ptf_plot_df = ptf_plot_df.rename(columns={k: f"%{k}" for k in actions})
    ptf_plot_df.loc[:, ptf_plot_df.filter(like="%").columns] = (ptf_plot_df.filter(like="%") * 100).round(round)

    ptf_plot_df["Rentabilité (%)"] = (ptf_plot_df["Rentabilité"] * 100).round(round)
    ptf_plot_df["Volatilité (%)"] = (ptf_plot_df["Volatilité"] * 100).round(round)
    return ptf_plot_df

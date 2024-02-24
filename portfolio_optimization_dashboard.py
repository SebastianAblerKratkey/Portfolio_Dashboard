import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.stats import norm
import os
import datetime
import statsmodels.api as sm
import base64
from io import StringIO, BytesIO

# this module is utilized to prevent the annotations in the plot from overlapping
from adjustText import adjust_text

# Get Yahoo Finance Data
import yfinance as yf

# Library for Website creation
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def convert_date_index(df):
    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)
    # Extract the month and year from the datetime
    df.index = df.index.strftime("%b %Y")
    return df

def visualize_performance(prices, list_of_names):
    benchmarking_data_ = prices / prices.iloc[-1]
    benchmarking_data = benchmarking_data_[::-1]*100
    individual_prices_list = []
    for n in list_of_names:
        individual_prices_list.append(benchmarking_data[n])
    plt.figure(figsize=(15, 10))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:,.0f}'.format))
    plt.fill_between(benchmarking_data.index, benchmarking_data.max(axis=1), benchmarking_data.min(axis=1),
                     color='grey', alpha=0.17, label="Range of all assets")
    color_list = ['deepskyblue', 'steelblue', 'mediumslateblue', 'cornflowerblue', 'lightsteelblue', 
                  'mediumslateblue', 'lightblue']
    
    for p, c in zip(individual_prices_list, color_list):
        plt.plot(p.index, p, color=c, label = p.name)
        plt.scatter(p.tail(1).index.max(), p[p.tail(1).index.max()], color=c)
        plt.text(len(benchmarking_data), p[p.tail(1).index.max()], '{:,.2f}'.format(p[p.tail(1).index.max()]),color=c, size=12)

    plt.gca().xaxis.set_major_locator(MaxNLocator())
    #plt.gca().set_xlim(left=benchmarking_data.head(1).index.max(), right=benchmarking_data.tail(1).index.max())
    plt.gca().set_xlim(left=benchmarking_data.head(1).index.max(), right=len(benchmarking_data)*1.1)
    plt.grid('on', ls="--")
    plt.ylabel(f"Performance (indexed: {benchmarking_data.head(1).index.max()} = 100)", fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

def visualize_summary(summary):
    fontsize=8
    plt.rc('font', size=fontsize)    
    fig, (ax1, ax2) = plt.subplots(1, 2, clip_on=False)
    ax1.grid('on', ls="--")
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(False)
    ax2.grid('on', ls="--")
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(False)
    ax3 =  ax2.twiny()
    ax4 = ax1.twiny()
    ax1.xaxis.set_major_locator(MaxNLocator(nbins="auto"))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins="auto"))
    ax3.xaxis.set_major_locator(MaxNLocator(nbins="auto"))
    ax4.xaxis.set_major_locator(MaxNLocator(prune='upper', nbins="auto"))
    x_dim = max(max(summary["mean return"]), max(summary['standard deviation'])) * 1.1
    height_of_fig = len(summary)*0.1
    ax1.set_position([0, 0, 0.35, height_of_fig])
    ax2.set_position([0.35, 0, 0.35, height_of_fig])
    ax1.set_xlim(left=0, right=x_dim)
    ax2.set_xlim(left=0, right=x_dim)
    ax3.set_xlim(left=0, right=x_dim)
    ax4.set_xlim(left=-x_dim, right=0)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
    ax3.xaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
    ax4.xaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
    ax1.invert_xaxis()
    ax2.tick_params(left = False, bottom=False)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    summary_sorted = summary.copy()
    summary_sorted["r/std"] = summary["mean return"] / summary['standard deviation']
    summary_sorted.sort_values("r/std", inplace=True)
    bar_width = 0.6  # Set a fixed width for the horizontal bars
    for index, row in summary_sorted.iterrows():
        ax1.barh(index, row['standard deviation'], height=bar_width, color="steelblue")
        ax2.barh(index,  row['mean return'], height=bar_width, color="deepskyblue")
        if row['mean return'] < 0:
            if abs(row['mean return']) > abs(row['standard deviation']):
                ax1.barh(index, abs(row['mean return']), height=bar_width, color="deepskyblue")
                ax1.barh(index, row['standard deviation'], height=bar_width, color="steelblue")
            if abs(row['mean return']) <= abs(row['standard deviation']):
                ax1.barh(index, row['standard deviation'], height=bar_width, color="steelblue")
                ax1.barh(index, abs(row['mean return']), height=bar_width, color="deepskyblue")
    ax1_patch = mpatches.Patch(color='deepskyblue', label='Mean return')
    ax1.legend(handles=[ax1_patch], fontsize=fontsize, frameon=False, loc='center', ncol=2, bbox_to_anchor=(1, 1+0.8/len(summary)))
    ax2_patch = mpatches.Patch(color='steelblue', label='Volatility')
    ax2.legend(handles=[ax2_patch], fontsize=fontsize, frameon=False, loc='center', ncol=2, bbox_to_anchor=(0, -0.8/len(summary)))
    plt.show()


def visualize_correlation(corr):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["deepskyblue", "mediumslateblue", "slategrey"])
    mask = np.triu(corr, k=1)
    plt.figure(figsize=(12, 7))
    sns.heatmap(corr, annot=True, cmap=cmap, mask=mask, linewidths=5,
                annot_kws={'color':'white'})
    plt.show()

def portfolio_std(weights):
    portfolio_std = np.sum(weights * np.sum(weights * annualized_cov_returns, axis=1)) ** 0.5
    return portfolio_std

def portfolio_return(weights, returns):
    portfolio_return = np.sum(weights * returns)
    return portfolio_return

def negative_portfolio_SR(weights, rf, returns):
    return_p = portfolio_return(weights, returns)
    std_p = portfolio_std(weights)
    negative_sharpe_ratio = -1*(return_p - rf) / std_p
    return negative_sharpe_ratio

def negative_portfolio_utility(weights, returns):
    return_p = portfolio_return(weights, returns)
    std_p = portfolio_std(weights)
    negative_portfolio_utility = -1*(return_p - 0.5*A*std_p**2)
    return negative_portfolio_utility

def create_KPI_report(name, weights, rf, returns):
    KPIs = pd.DataFrame(index=[name])
    KPIs["portfolio return"] = portfolio_return(weights, returns)
    KPIs["protfolio std"] = portfolio_std(weights)
    KPIs["sharpe ratio"] = (KPIs["portfolio return"]- rf) / KPIs["protfolio std"]
    KPIs["utility"] = KPIs["portfolio return"] - 0.5*A*KPIs["protfolio std"]**2
    return KPIs

def create_portfolio_visual(name, summary, KPIs):
    plt.figure(figsize=(8, 8))   
    plt.pie(summary["weight"], wedgeprops=dict(width=0.45), 
            colors=['deepskyblue', 'steelblue', 'lightblue', 'lightsteelblue', 'cornflowerblue',
                    'mediumslateblue','thistle', 'dodgerblue', 'slategrey'],
            autopct='%.2f%%',pctdistance=0.8, startangle=90,labels=summary.index)
    plt.annotate(name, xy=(0,0), fontsize=30, va="center", ha="center")
    plt.annotate("E(r): {}%".format(float((KPIs["portfolio return"]*100).round(decimals=2))), 
                 xy=(-0.07,-0.18), fontsize=10, va="center", ha="right")
    plt.annotate("Vola: {}%".format(float((KPIs["protfolio std"]*100).round(decimals=2))), 
                 xy=(+0.07,-0.18), fontsize=10, va="center", ha="left")
    plt.show()

def create_mvf_cal_visual():
    #plot minimum varriance frontier and CAL
    color1 = 'cornflowerblue'
    color2 = 'darkmagenta'

    plt.figure(figsize=(15, 10))

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:,.2%}'.format))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:,.2%}'.format))
    plt.gca().set_xlim(left=0)
    plt.gca().set_xlim(right=max(max(summary["standard deviation"]),float(KPIs_ocp["protfolio std"]))*1.05)

    plt.scatter(summary["standard deviation"], summary["mean return"], color=color1)

    # capital allocation line

    # between std = 0 and std = std_orp_l
    std_cal_1 = np.arange(0, float(KPIs_orp_l["protfolio std"]), step)
    return_cal_1 = rf_l + float(KPIs_orp_l["sharpe ratio"])*std_cal_1
    plt.plot(std_cal_1 ,return_cal_1, color=color1, label='Capital allocation line')

    # between std_orp_l and std_orp_b -> follows minimum varriance frontier
    mvf_plot_slice = mvf_plot_data[(mvf_plot_data["return"] >= float(KPIs_orp_l["portfolio return"])) & 
                               (mvf_plot_data["return"] <= float(KPIs_orp_b["portfolio return"]))]
    std_cal_2 = mvf_plot_slice["std"]
    return_cal_2 = mvf_plot_slice["return"]
    plt.plot(std_cal_2,return_cal_2,color=color1)

    # after std_orp_b
    endpoint_cal = plt.gca().get_xlim()[1] 
    std_cal_3 = np.arange(float(KPIs_orp_b["protfolio std"]), endpoint_cal, step)
    return_cal_3 = rf_b + float(KPIs_orp_b["sharpe ratio"])*std_cal_3
    plt.plot(std_cal_3 ,return_cal_3, color=color1)

    # minimum varriance frontier
    plt.plot(mvf_plot_data["std"], mvf_plot_data["return"], color=color1, linestyle='--',
         label='Minimum varriance frontier')

    plt.scatter(KPIs_mvp["protfolio std"], KPIs_mvp["portfolio return"], color=color2)
    plt.scatter(KPIs_orp["protfolio std"], KPIs_orp["portfolio return"], color=color2)
    plt.scatter(KPIs_ocp["protfolio std"], KPIs_ocp["portfolio return"], color=color2)

    plt.legend(fontsize=12)
    plt.xlabel("Volatility", fontsize=12)
    plt.ylabel("Mean return", fontsize=12)
    plt.grid('on', ls="--")

    # labeling
    x_offset = plt.gca().get_xlim()[1]*0.01
    for i in summary_p.index:
        plt.annotate(i,(summary_p["protfolio std"][i], summary_p["portfolio return"][i]),
                 (summary_p["protfolio std"][i]-x_offset, summary_p["portfolio return"][i]), 
                 color=color2, fontsize=12, ha='right')   

    labels = []
    for i in summary.index:
        labels.append(plt.text(summary["standard deviation"][i], summary["mean return"][i], i, size=8))
    adjust_text(labels) 
    
    plt.show()

def currency_formatter_alt_EUR_decimal_seperator(x, currency="EUR"):
         if currency == 'EUR':
            return f'{currency} {x:,.2f}'.replace(",", "X").replace(".", ",").replace("X", ".")
         elif currency == 'USD':
            return f'{currency} {x:,.2f}'
         
def currency_formatter(x, currency="EUR"):
    return f'{currency} {x:,.2f}'

def currency_formatter_signs(x, currency="EUR"):
         if currency == 'EUR':
            return f'€ {x:,.2f}'
         elif currency == 'USD':
            return f'$ {x:,.2f}'

def visualize_simulaiton(sim_avg, deposits, currency='EUR'):
    """
    Plots the average simulated performance over time.
    
    Parameters:
    sim_avg (DataFrame): A DataFrame containing the average simulated performance.
    currency (str): The currency in which to display the performance data.
    
    Returns:
    None
    """
    def currency_formatter(x, pos):
         if currency == 'EUR':
            return f'€ {x:,.2f}'
         elif currency == 'USD':
            return f'$ {x:,.2f}'
    
    plt.figure(figsize=(15, 5))
    
    # Set the y-axis formatter
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Set the tick locations and labels
    plt.xticks(sim_avg.index)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
    
    # Plot the bars
    if sim_avg.iloc[-1] > deposits[-1]:
        plt.bar(sim_avg.index, sim_avg, color='deepskyblue', label="Capital", align='center')
        plt.bar(sim_avg.index, deposits, color='steelblue', label="Money invested", align='center')
    if sim_avg.iloc[-1] < deposits[-1]:
        plt.bar(sim_avg.index, deposits, color='steelblue', label="Money invested", align='center')
        plt.bar(sim_avg.index, sim_avg, color='deepskyblue', label="Capital", align='center')

    # Set the x-axis limits based on the minimum and maximum values in the index
    plt.gca().set_xlim(left=sim_avg.index.min()-0.8)
    plt.gca().set_xlim(right=sim_avg.index.max()+0.8)
    
    # Rotate xticks if needed
    if len(sim_avg.index) > 22:
        plt.xticks(rotation=45)
    if len(sim_avg.index) > 60:
        plt.xticks(rotation=90)
    if len(sim_avg.index) > 75:
        plt.gca().tick_params(axis='x', labelsize=8)
        
    plt.gca().set_axisbelow(True)
    plt.grid('on', ls="--")
    plt.gca().xaxis.grid(False)
    plt.legend(fontsize=12)
    
    plt.show()

def generate_excel_download_link(df):
    towrite = BytesIO()
    df.to_excel(towrite, index=False, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="template.xlsx">Excel template'
    return st.markdown(href, unsafe_allow_html=True)

def maximum_drawdowns(price_df):
    """
    Calculate the maximum drawdowns of a dataframe of asset prices.

    Parameters:
    price_df (pd.DataFrame): A pandas DataFrame containing asset prices.
    (date index must be sorted ascending)

    Returns:
    pd.Series: Series of asset names and corresponding maximum drawdowns.
    """
    price_df_sorted = price_df.sort_index(ascending=True)
    max_price_df = price_df_sorted.rolling(window=len(price_df_sorted),min_periods=1).max()
    dd_price_df = price_df_sorted / max_price_df -1
    max_dd_series = dd_price_df.min()

    return max_dd_series

def get_monthly_closing_prices(price_df_daily): 
    price_df_monthly = price_df_daily.loc[price_df_daily.groupby(price_df_daily.index.to_period('M')).apply(lambda x: x.index.max())]
    return price_df_monthly


option = st.sidebar.selectbox("What do you want to see?", ("Past performance", "Custom portfolio","Return correlation", "MVP, ORP and OCP", "Minimum varriance frontier and Capital allocation line", "CAPM", "Savings plan simulation", "Data"))

st.header("Portfolio Analysis")

tickers = []
input_tickers = st.text_input("Enter the [Yahoo Finace](https://finance.yahoo.com) tickers of the assets you are interested in (seperated by comma). Make sure to select at least two.")
# [link](https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py)
if input_tickers:
    tickers  = input_tickers.split(",")
    tickers = [x.strip() for x in tickers]

# generate dataframe for Excel template
template_df = pd.DataFrame(columns=["Yahoo finance ticker",	"Asset class", "Number of shares", "TER"], index=None)


custom_p = st.file_uploader("You can also upload a custom portfolio by filling out and uploading the Excel template below.", type="xlsx")
generate_excel_download_link(template_df)

if custom_p:
    custom_p_df = pd.read_excel(custom_p)
    tickers_template = list(custom_p_df["Yahoo finance ticker"])
    tickers += tickers_template

download_sucess = False
if input_tickers or custom_p:
    tickers = [x.upper() for x in tickers]
    price_df = yf.download(tickers, period='max')["Adj Close"]
    price_df = price_df.dropna()

    eurusd = yf.download(["EURUSD=X","EUR=X"], period='max', interval='1mo')["Adj Close"]
    
    if len(price_df) < 1:
        st.error("(Some) assets could not be found.")
    elif len(tickers) == 1:
        st.error("Make sure to select at least two assets.")
    else:
        st.success("Assets added!")
        download_sucess = True

st.write("Disclaimer: This is not financial advice.")


currency = st.selectbox("Select a currency.", ["EUR", "USD"])

rf_l = st.slider("What is your risk-free interest rate for lending money?",
                min_value=0.0, max_value=10.0, step=0.01, format='%.2f%%')
rf_l = rf_l/100

rf_b = st.slider("What is your risk-free interest rate for borrowing money?",
                min_value=0.0, max_value=10.0, step=0.01, format='%.2f%%')
rf_b = rf_b/100

A = st.slider("Adjust your risk aversion parameter (higher = more risk averse, lower = less risk averse).",
                min_value=0.1, max_value=15.0, step=0.1, value=10.0)




if download_sucess:

    # Currency conversion and long name dictionary creation
    curr_conv_tabl = pd.DataFrame(index=price_df.index, columns=price_df.columns)
    curr_conv_tabl.fillna(1, inplace=True)

    long_name_dict = {}
    for col in curr_conv_tabl.columns:
        try:
            long_name_dict[col] = yf.Ticker(col).info["longName"]
            curr = yf.Ticker(col).info["currency"]
            if curr != currency:
                if currency == "EUR":
                    curr_conv_tabl[col] = curr_conv_tabl[col] * eurusd["EUR=X"]
                elif currency == "USD":
                    curr_conv_tabl[col] = curr_conv_tabl[col] * eurusd["EURUSD=X"] 
        except:
            pass
    
    #Paste down last avalible EUR/USD rate (Dec 2003) for datapoints before that date
    col_names = list(curr_conv_tabl.columns)
    last_exch_rates = curr_conv_tabl.dropna().iloc[0,:].tolist()
    last_exch_rate_dict = dict(zip(col_names, last_exch_rates))
    curr_conv_tabl.fillna(last_exch_rate_dict, inplace=True)

    price_df = price_df * curr_conv_tabl
    
    daily_adjusted_closing_prices = price_df

    #Calculate YTD returns
    now = price_df.index[-1]
    end_prev_y = price_df.index[price_df.index.year<now.year][-1]
    ytd_returns = price_df.loc[now] / price_df.loc[end_prev_y] - 1

    start_date = daily_adjusted_closing_prices.index.min().date()
    end_date = daily_adjusted_closing_prices.index.max().date()
    
    date_range = st.slider("Define the timeframe to be considered for the analysis.", value=[start_date, end_date], min_value=start_date, max_value=end_date, format ='DD MM YYYY')
    start_date_from_index = daily_adjusted_closing_prices.index[(daily_adjusted_closing_prices.index.month==date_range[0].month) & (daily_adjusted_closing_prices.index.year==date_range[0].year)].min().date()
    end_date_from_index = daily_adjusted_closing_prices.index[(daily_adjusted_closing_prices.index.month==date_range[1].month) & (daily_adjusted_closing_prices.index.year==date_range[1].year)].max().date()
    daily_adjusted_closing_prices = daily_adjusted_closing_prices.loc[start_date_from_index:end_date_from_index]

    montly_adjusted_closing_prices = get_monthly_closing_prices(price_df_daily=daily_adjusted_closing_prices)

     # calculate maximum drawdown
    max_dds = maximum_drawdowns(price_df=montly_adjusted_closing_prices)

    montly_adjusted_closing_prices = convert_date_index(montly_adjusted_closing_prices)
    monthly_log_returns = np.log(montly_adjusted_closing_prices / montly_adjusted_closing_prices.shift(1))

    annualized_mean_returns = monthly_log_returns.mean() * 12
    annualized_std_returns = monthly_log_returns.std() * 12**0.5
    annualized_cov_returns = monthly_log_returns.cov() * 12
    corr_returns = monthly_log_returns.corr()

    summary = pd.DataFrame()

    summary["mean return"] = annualized_mean_returns
    summary["standard deviation"] = annualized_std_returns
    summary["weight"] = 1/len(summary)
    
    mvp_summary = summary.copy()
    mvp = minimize(portfolio_std, x0=mvp_summary["weight"].values,  
               bounds=Bounds(0,1), 
               constraints={'type': 'eq','fun' : lambda weights: np.sum(weights) - 1})
    mvp_summary["weight"] = mvp.x
    mvp_summary_abrev = mvp_summary[mvp_summary["weight"] > 0.00001]

    KPIs_mvp = create_KPI_report("MVP",
                             mvp_summary["weight"],
                             rf_l,
                             mvp_summary["mean return"])

    orp_l_summary = summary.copy()
    orp_l = minimize(negative_portfolio_SR, x0=orp_l_summary["weight"].values, 
                 args=(rf_l, summary["mean return"] ), 
                 bounds=Bounds(0,1), 
                 constraints={'type': 'eq','fun' : lambda weights: np.sum(weights) - 1})
    orp_l_summary["weight"] = orp_l.x
    orp_l_summary_abrev = orp_l_summary[orp_l_summary["weight"] > 0.00001]

    KPIs_orp_l = create_KPI_report("ORP",
                             orp_l_summary["weight"],
                             rf_l,
                             orp_l_summary["mean return"])

    orp_b_summary = summary.copy()
    orp_b = minimize(negative_portfolio_SR, x0=orp_b_summary["weight"].values, 
                 args=(rf_b, summary["mean return"]), 
                 bounds=Bounds(0,1), 
                 constraints={'type': 'eq','fun' : lambda weights: np.sum(weights) - 1})
    orp_b_summary["weight"] = orp_b.x
    orp_b_summary_abrev = orp_b_summary[orp_b_summary["weight"] > 0.00001]

    KPIs_orp_b = create_KPI_report("ORP",
                             orp_b_summary["weight"],
                             rf_b,
                             orp_b_summary["mean return"])

    orp_indiff_summary = summary.copy()
    orp_indiff = minimize(negative_portfolio_utility, x0=orp_indiff_summary["weight"].values, 
                      args=(summary["mean return"]), 
                      bounds=Bounds(0,1), 
                      constraints={'type': 'eq','fun' : lambda weights: np.sum(weights) - 1})
    orp_indiff_summary["weight"] = orp_indiff.x
    orp_indiff_summary_abrev = orp_indiff_summary[orp_indiff_summary["weight"] > 0.00001]

    KPIs_orp_indiff = create_KPI_report("ORP",
                             orp_indiff_summary["weight"],
                             rf_l,
                             orp_indiff_summary["mean return"])
    
    # OCP is defined
    weight_risky_l = (float(KPIs_orp_l["portfolio return"]) - rf_l) / (A * float(KPIs_orp_l["protfolio std"])**2)
    weight_risky_b = (float(KPIs_orp_b["portfolio return"]) - rf_b) / (A * float(KPIs_orp_b["protfolio std"])**2)

    if weight_risky_l <= 1:
        orp_summary = orp_l_summary
        weight_orp = weight_risky_l
        return_orp = float(KPIs_orp_l["portfolio return"])
        std_orp = float(KPIs_orp_l["protfolio std"])
        KPIs_orp = KPIs_orp_l
        weight_rf = 1 - weight_orp
        rf = rf_l
        rf_summary = pd.DataFrame(index=["risk-free lending"])
    elif weight_risky_b > 1:
        orp_summary = orp_b_summary
        weight_orp = weight_risky_b
        return_orp = float(KPIs_orp_b["portfolio return"])
        std_orp = float(KPIs_orp_b["protfolio std"])
        KPIs_orp = KPIs_orp_b
        weight_rf = 1 - weight_orp
        rf = rf_b
        rf_summary = pd.DataFrame(index=["risk-free borrowing"]) 
    else:
        orp_summary = orp_indiff_summary
        weight_orp = 1
        KPIs_orp = KPIs_orp_indiff
        return_orp = float(KPIs_orp_indiff["portfolio return"])
        std_orp = float(KPIs_orp_indiff["protfolio std"])
        weight_rf = 0
        rf = 0
        rf_summary = pd.DataFrame()

    rf_summary["mean return"] = rf    
    rf_summary["standard deviation"] = 0
    rf_summary["weight"] = weight_rf    

    orp_summary_abrev = orp_summary[(orp_summary["weight"] > 0.00001)]

    temp = orp_summary
    temp["weight"] = orp_summary["weight"] * weight_orp
    ocp_summary = pd.concat([temp, rf_summary])
    ocp_summary_abrev = ocp_summary[(ocp_summary["weight"] > 0.00001) | (ocp_summary["weight"] < -0.00001)]


    KPIs_ocp = pd.DataFrame(index=["OCP"])
    KPIs_ocp["portfolio return"] = weight_orp*return_orp + weight_rf*rf
    KPIs_ocp["protfolio std"] = weight_orp*std_orp
    KPIs_ocp["sharpe ratio"] = (KPIs_ocp["portfolio return"] - rf) / KPIs_ocp["protfolio std"]
    KPIs_ocp["utility"] = KPIs_ocp["portfolio return"] - 0.5*A*KPIs_ocp["protfolio std"]**2

    summary_p = pd.concat([KPIs_mvp, KPIs_orp, KPIs_ocp])    

    if summary_p["portfolio return"]["OCP"] == summary_p["portfolio return"]["ORP"]:
        summary_p.rename(index={'OCP':'OCP = ORP'}, inplace=True)
        summary_p.drop("ORP", inplace=True)

    # Get data for minimum varriance frontier plot
    step = 0.001
    acheivable_returns = np.arange(summary["mean return"].min(), summary["mean return"].max()+ step, step)

    min_var_list = []
    for r in acheivable_returns:
        min_var = minimize(portfolio_std, x0=summary["weight"].values, 
                       bounds=Bounds(0,1), 
                       constraints=({'type': 'eq','fun' : lambda weights: np.sum(weights) - 1},
                                    {'type': 'eq','fun' : lambda weights: np.sum(weights*summary["mean return"]) - r}))
        min_var_list.append(min_var.fun)

    mvf_plot_data = pd.DataFrame()
    mvf_plot_data["return"] = acheivable_returns
    mvf_plot_data["std"] = min_var_list

    # Custom portfolio dataframe and metrics
    if custom_p:
            custom_p_df = custom_p_df.join(price_df.loc[now, tickers_template].rename("Current price"), on="Yahoo finance ticker")
            custom_p_df["Current value"] = custom_p_df["Number of shares"] * custom_p_df["Current price"]
            custom_p_df["weight"] = custom_p_df["Current value"] / sum(custom_p_df["Current value"])
            custom_p_df.set_index(keys="Yahoo finance ticker", inplace=True)
            custom_p_df.index.name = None
            custom_p_df["Full name"] = pd.Series(long_name_dict)
            custom_p_df["YTD return"] = ytd_returns
            #st.dataframe(custom_p_df)
            asset_class_df = custom_p_df.groupby('Asset class').weight.sum().to_frame()
            #st.dataframe(asset_class_df)

            custom_p_summary = summary.loc[tickers_template].copy()
            custom_p_summary["weight"] = custom_p_df["weight"]

            KPIs_custom_p = create_KPI_report("Custom portfolio",
                             custom_p_summary["weight"],
                             rf_l,
                             custom_p_summary["mean return"])

            custom_p_summary["Full name"] = custom_p_df["Full name"]
            custom_p_summary_long_name = custom_p_summary.set_index(keys="Full name")
            custom_p_summary_long_name.index.name = None
            r_custom_p = float(KPIs_custom_p["portfolio return"])
            std_custom_p = float(KPIs_custom_p["protfolio std"])
            weight_avg_TER = sum(custom_p_df["TER"] * custom_p_df["weight"])
            ytd_return_custom_p = sum(custom_p_df["YTD return"] * custom_p_df["weight"])
            custom_p_worth = sum(custom_p_df["Current value"])


    st.subheader(option)
    if option == "Past performance":
        display_summary = pd.DataFrame()
        display_summary["Full name"] = pd.Series(long_name_dict)
        display_summary["Mean return p.a."] = summary["mean return"].map('{:.2%}'.format)
        display_summary["Volatility p.a."] = summary["standard deviation"].map('{:.2%}'.format)
        display_summary["Sharpe ratio"] = (summary["mean return"]- rf_l) / summary['standard deviation']
        display_summary["YTD return"] = ytd_returns.map('{:.2%}'.format)
        display_summary["Maximum drawdown"] = max_dds.map('{:.2%}'.format)
        #display_summary["Mean return p.a."] = display_summary["Mean return p.a."].map('{:.2%}'.format)
        #display_summary["Volatility p.a."] = display_summary["Volatility p.a."].map('{:.2%}'.format)
        #display_summary["YTD return"] = display_summary["YTD return"].map('{:.2%}'.format)
        display_summary.sort_values("Sharpe ratio", inplace=True, ascending=False)
        st.dataframe(display_summary)

        visualize_summary(summary)
        st.pyplot()

        tickers_chosen = st.multiselect("Select the assets you want to compare:", tickers)
        visualize_performance(montly_adjusted_closing_prices, tickers_chosen)
        st.pyplot()

    if option == "Custom portfolio":
        if custom_p: 
            
            colmn_1, colmn_2, colmn_3, colmn_4 = st.columns([0.5, 0.5, 0.5, 0.5]) 
            colmn_1.metric("Expected return p.a.", f"{r_custom_p:.2%}")
            colmn_2.metric("Volatility p.a.", f"{std_custom_p:.2%}")
            colmn_3.metric("YTD return", f"{ytd_return_custom_p:.2%}")
            colmn_4.metric("Weighted average TER", f"{weight_avg_TER:.2%}")

            display_option = st.selectbox("Select a view setting for the portfolio visualization.", ["Individual asset view", "Asset class view"])
            if display_option == "Individual asset view":
                create_portfolio_visual(f'{currency_formatter_signs(custom_p_worth, currency=currency)}', custom_p_summary_long_name, KPIs_custom_p)
                st.pyplot()
            elif display_option == "Asset class view": 
                create_portfolio_visual(f'{currency_formatter_signs(custom_p_worth, currency=currency)}', asset_class_df, KPIs_custom_p)
                st.pyplot()

            
        else:
            st.write("Upload a filled out Excel template to view your custom portfolio.")    
        

    if option == "Return correlation":
        visualize_correlation(corr_returns)
        st.pyplot()
    
    if option == "MVP, ORP and OCP":
        # Show MVP
        st.write("""
                 In the **minimum variance portfolio (MVP)**, assets are allocated in such a way as to achieve the lowest possible 
                 portfolio variance, taking into account each asset's historical standard deviation of returns (volatility) and 
                 the correlation between the assets' returns. In general, combining assets with a low to negative return correlation 
                 will result in a lower portfolio variance. This is because price movements in one asset are dampened by weaker 
                 movements in other assets, or movements in the opposite direction. For the selected assets, the MVP can be achieved 
                 via the following allocation:
                """)
        create_portfolio_visual("MVP", mvp_summary_abrev, KPIs_mvp)
        st.pyplot()
        
        # Show ORP
        st.write("""
                 The **optimal risky portfolio (ORP)** is achieved by combining the (risky) assets such that the Sharpe ratio 
                 — i.e., the risk-return trade-off — is maximized. The Sharpe ratio defines the slope of the capital allocation 
                 line (CAL). Therefore, the ORP is the point of tangency between the CAL and the minimum variance frontier. 
                 For this reason, the ORP is sometimes referred to as the " tangency portfolio." In theory, the ORP is 
                 always chosen by rational investors as the risky portion of their investment. Therefore, the ORP should 
                 be independent of your risk aversion parameter. However, if your risk-free lending rate differs from your 
                 risk-free borrowing rate, two different Sharpe ratios arise, resulting in two potential ORPs. One for when 
                 you lend out money as part of your investment, and one for when you borrow money to increase your stake in 
                 the ORP. Depending on your risk appetite, you may also be indifferent between borrowing and lending. In 
                 this case, the ORP is determined by maximizing your utility function given a 100% stake in the risky assets. 
                 For the selected assets, the ORP can be obtained via the following allocation:
                """)
        create_portfolio_visual("ORP", orp_summary_abrev, KPIs_orp)
        st.pyplot()
        

        # Show OCP (not finished)
        st.write("""
                The **optimal complete portfolio (OCP)** is created by combining the ORP with a risk-free instrument. Depending on 
                your risk appetite you will either lend out money at the risk-free rate — i.e., invest less than 100% in the ORP 
                 — or borrow money at the risk-free rate to increase your stake in the ORP beyond 100%. Since the risk-free borrowing 
                 rate, you are facing will most likely be different (higher) from your risk-free lending rate, there is a range of 
                 risk aversion parameters for which you will be indifferent between lending and borrowing. Here the CAL will follow 
                 the minimum variance frontier and your OCP will be the same as your ORP.
                 """)
        if weight_rf > 0:
            st.write("For the selected assets and risk aversion parameter, the OCP is obtained by combining the ORP with risk-free lending:")
            create_portfolio_visual("OCP", ocp_summary_abrev, KPIs_ocp)
            st.pyplot()
        elif weight_rf == 0:
            st.write("For the selected assets and risk aversion parameter, you are indifferent between lending and borrwing. Therefore, the OCP is equal to the ORP.")
        else:
            st.write(f"""
            For the selected assets, the OCP can be obtained by investing {weight_orp:.2%} of 
            your money in the ORP i.e., borrowing the missing {(weight_orp-1):.2%} at the risk-free rate. 
            In this case the OCP would be expected to generate a return of {float(KPIs_ocp['portfolio return']):.2%} 
            p.a. at a standard deviation of {float(KPIs_ocp['protfolio std']):.2%}.
            """)


    if option == "Minimum varriance frontier and Capital allocation line":
        create_mvf_cal_visual()
        st.pyplot()

    if option == "CAPM":
        
        market_proxy_input = st.text_input("As per default, the S&P 500 Index (^GSPC) is used as a proxy for the market portfolio. If you consider another index more suitable for your analysis, you can enter its [Yahoo Finace](https://finance.yahoo.com) ticker below (E.g. STOXX Europe 600: ^STOXX, Dax-Performance-Index: ^GDAXI, FTSE 100 Index: ^FTSE)")
        riskfree_proxy_input = st.text_input("As per default, 10-year U.S. Treasury yields (^TNX) are used as a proxy for the risk-free rate. You may enter the ticker of a different proxy below (make sure the proxy is quoted in yields, not prices; e.g. 13-week U.S. Treasury yields: ^IRX, 5-year U.S. Treasury yields: ^FVX, 30-year U.S. Treasury yields: ^TYX)")

        if market_proxy_input:
            market_proxy = market_proxy_input
        else:
            market_proxy = "^GSPC"
        if riskfree_proxy_input:
            riskfree_proxy = riskfree_proxy_input
        else:
            riskfree_proxy = "^TNX"

        proxys_M_rf = [market_proxy, riskfree_proxy]

        CAPM_data = yf.download(proxys_M_rf, period='max')["Adj Close"]
        CAPM_data.dropna(inplace=True) 
        CAPM_data = get_monthly_closing_prices(price_df_daily=CAPM_data)

        download_sucess2 = False
        if len(CAPM_data) < 1:
            st.error("Asset could not be found.")
        else:
            if market_proxy_input or riskfree_proxy_input:
                st.success("Proxy updated!")
            download_sucess2 = True

        if download_sucess2:
            CAPM_quotes = pd.DataFrame()
            CAPM_quotes[["Market", "risk-free"]] = CAPM_data[proxys_M_rf]
            CAPM_quotes["risk-free"] = CAPM_quotes["risk-free"]/100
            CAPM_quotes = convert_date_index(CAPM_quotes)
            CAPM_returns = pd.DataFrame()
            CAPM_returns["Market"] = np.log(CAPM_quotes["Market"] / CAPM_quotes["Market"].shift(1))
            CAPM_returns["risk-free"] = CAPM_quotes["risk-free"]*(1/12)
            CAPM_returns["MRP"] = CAPM_returns["Market"] - CAPM_returns["risk-free"]

            for asset in monthly_log_returns.columns:
                CAPM_returns[f"{asset}-rf"] = monthly_log_returns[asset] - CAPM_returns["risk-free"]
            
            CAPM_returns.dropna(inplace=True)

            mean_rf = CAPM_returns["risk-free"].mean()*12
            mean_MRP = CAPM_returns["MRP"].mean()*12
            
            CAPM_summary = pd.DataFrame()
            for asset in monthly_log_returns.columns:
                Y=CAPM_returns[f"{asset}-rf"]
                X= sm.add_constant(CAPM_returns["MRP"])
                regress = sm.OLS(Y, X)
                result = regress.fit()
                CAPM_summary.loc[asset,"Mean rf"] = mean_rf
                CAPM_summary.loc[asset,"Beta"] = float(result.params[1])
                CAPM_summary.loc[asset,"Mean MRP"] = mean_MRP
                CAPM_summary.loc[asset,"Fair return"] = CAPM_summary.loc[asset,"Mean rf"] + CAPM_summary.loc[asset,"Beta"]*CAPM_summary.loc[asset,"Mean MRP"]
                CAPM_summary.loc[asset,"Alpha"] = float(result.params[0]*12)
                CAPM_summary.loc[asset,"Expected return"] = CAPM_summary.loc[asset,"Fair return"] + CAPM_summary.loc[asset,"Alpha"]

            display_CAPM_summary = pd.DataFrame()
            display_CAPM_summary["Mean rf"] = CAPM_summary["Mean rf"].map('{:.2%}'.format)
            display_CAPM_summary["Beta"] = CAPM_summary["Beta"].map('{:.2f}'.format)
            display_CAPM_summary["Mean MRP"] = CAPM_summary["Mean MRP"].map('{:.2%}'.format)
            display_CAPM_summary["Fair Return"] = CAPM_summary["Fair return"].map('{:.2%}'.format)
            display_CAPM_summary["Alpha"] = CAPM_summary["Alpha"].map('{:.2%}'.format)
            display_CAPM_summary["Expected Return"] = CAPM_summary["Expected return"].map('{:.2%}'.format)
            st.dataframe(display_CAPM_summary)

            #Visualize SML
            step = 0.001

            color1 = 'cornflowerblue'
            color2 = 'darkmagenta'

            plt.figure(figsize=(15, 10))

            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:,.2%}'.format))
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:,.2}'.format))
            plt.gca().set_xlim(left=min(min(CAPM_summary["Beta"])*1.5,0))
            plt.gca().set_xlim(right=max(CAPM_summary["Beta"])*1.5)

            plt.scatter(CAPM_summary["Beta"], CAPM_summary["Expected return"], color=color2, label="Expected return")

            for i in summary.index:
                alpha_y = np.arange(min(CAPM_summary["Fair return"][i],CAPM_summary["Expected return"][i]), 
                                    max(CAPM_summary["Fair return"][i],CAPM_summary["Expected return"][i]), step)
                alpha_x = np.ones((len(alpha_y),1))*CAPM_summary["Beta"][i]
                if i == summary.index[0]:
                    plt.plot(alpha_x ,alpha_y, color=color2, linestyle='--', label="Alpha")
                else:
                    plt.plot(alpha_x ,alpha_y, color=color2, linestyle='--')

            plt.scatter(CAPM_summary["Beta"], CAPM_summary["Fair return"], color=color1, label="Fair return")

            # SML
            endpoint_sml = plt.gca().get_xlim()[1] 

            betas_sml = np.arange(min(min(CAPM_summary["Beta"])*1.5,0), endpoint_sml, step)
            return_sml = mean_rf + betas_sml*mean_MRP
            plt.plot(betas_sml ,return_sml, color=color1, label='Security market line')


            plt.legend(fontsize=12)
            plt.xlabel("Beta", fontsize=12)
            plt.ylabel("Return", fontsize=12)
            plt.grid('on', ls="--")

            labels = []
            for i in summary.index:
                labels.append(plt.text(CAPM_summary["Beta"][i], CAPM_summary["Expected return"][i], i, size=8))
            adjust_text(labels) 
            
            plt.show()
            st.pyplot()





    if option == "Savings plan simulation":
        num_trials = 10000

        r_ocp = float(KPIs_ocp["portfolio return"])
        std_ocp = float(KPIs_ocp["protfolio std"])

        r_orp = float(KPIs_orp["portfolio return"])
        std_orp = float(KPIs_orp["protfolio std"])

        r_mvp = float(KPIs_mvp["portfolio return"])
        std_mvp = float(KPIs_mvp["protfolio std"])

       
        if custom_p:
            portfolios = ["Custom portfolio","OCP", "ORP", "MVP"]
            val_today_assume = custom_p_worth
        else:
            portfolios = ["OCP", "ORP", "MVP"]
            val_today_assume = 1000.00
        
        val_today = st.number_input("What is your starting capital i.e. the ammount of money you can invest today?",min_value=1.0, value=val_today_assume)

        #Ask user to enter amount of money they want to save each year
        additional_investment_per_month = st.number_input("How much money do you want to save each month?",min_value=0.0, value=100.0)
        additional_investment_per_year = additional_investment_per_month*12




        selected_p = st.selectbox("What portfolio do you want to invest in?", portfolios)


        col1, col2 = st.columns([1,3])
        
        # Here the "abrev" version of the summary must be used. Just the sumamry is not correct anymore at this point for some reason.
        if selected_p == "Custom portfolio":
            col1.metric("Expected return p.a.", f"{r_custom_p:.2%}")
            col2.metric("Volatility p.a.", f"{std_custom_p:.2%}")
            sim_summary = custom_p_summary.copy()
            sim_summary["mean return"] = custom_p_summary["mean return"] / 12
            sim_summary["standard deviation"] = custom_p_summary["standard deviation"] / (12**0.5)
        
        elif selected_p == "MVP":
            col1.metric("Expected return p.a.", f"{r_mvp:.2%}")
            col2.metric("Volatility p.a.", f"{std_mvp:.2%}")
            sim_summary = mvp_summary_abrev.copy()
            sim_summary["mean return"] = mvp_summary_abrev["mean return"] / 12
            sim_summary["standard deviation"] = mvp_summary_abrev["standard deviation"] / (12**0.5)

        elif selected_p == "ORP":
            col1.metric("Expected return p.a.", f"{r_orp:.2%}")
            col2.metric("Volatility p.a.", f"{std_orp:.2%}")
            sim_summary = orp_summary_abrev.copy()
            sim_summary["mean return"] = orp_summary_abrev["mean return"] / 12
            sim_summary["standard deviation"] = orp_summary_abrev["standard deviation"] / (12**0.5)

        elif selected_p == "OCP":
            col1.metric("Expected return p.a.", f"{r_ocp:.2%}")
            col2.metric("Volatility p.a.", f"{std_ocp:.2%}")
            sim_summary = ocp_summary_abrev.copy()
            sim_summary["mean return"] = ocp_summary_abrev["mean return"] / 12
            sim_summary["standard deviation"] = ocp_summary_abrev["standard deviation"] / (12**0.5)
        

        num_years = st.slider("For how many years do you want to save?",
                              min_value=1, max_value=100, step=1, value=20)
        num_months = 12*num_years

        p = st.slider("Define the percentage of simulation outcomes to be contained in a symmetrical bandwidth around the simulation mean:",
                      min_value=0.05, max_value=0.95, step=0.05, value=0.8)
        
        current_year = now = datetime.datetime.now().year

        
        total_additional_investments = num_years * additional_investment_per_year

        

        if st.button("Run simulation"):
            # Generate the random trials
            sim_list = []
            for index, row in sim_summary.iterrows():
                sim_list.append(np.random.normal(row["mean return"], row["standard deviation"], size=(num_trials, num_months)) * row["weight"])

            simulated_returns = np.array(sim_list).sum(axis=0)

            # Calculate the potential future values of the investment
            val_future = np.zeros((num_trials, num_months+1))
            val_future[:,0] = val_today

            for i in range(1, num_months+1):
                val_future[:,i] = val_future[:,i-1] * np.exp(simulated_returns[:,i-1]) + additional_investment_per_month
        
            simulated_performance = pd.DataFrame(val_future).transpose()
            simulated_performance_annual = simulated_performance[simulated_performance.index % 12 == 0]


            # Set the index of the DataFrame to a range of years representing the time horizon of the simulation
            future_years = np.arange(current_year, current_year + num_years + 1, 1)
            simulated_performance_annual.set_index(pd.Index(future_years), inplace=True)

            # Calculate the average simulated performance across all trials
            avg_simulated_performance = simulated_performance_annual.mean(axis=1)

            # Calculate the cumulative deposits
            if additional_investment_per_year > 0:
                cumulative_depositis = np.arange(val_today, 
                                    val_today+total_additional_investments+additional_investment_per_year,
                                    additional_investment_per_year)
            else:
                cumulative_depositis = np.array([val_today]*(num_years+1))
    
            visualize_simulaiton(avg_simulated_performance, cumulative_depositis, currency=currency)
            st.pyplot()

            expected_capital = float(avg_simulated_performance.iloc[-1])

            # Cash flow of the savings plan
            cash_flow = [-val_today]+[-additional_investment_per_month]*(num_months-1)+[avg_simulated_performance.iloc[-1]-additional_investment_per_month]
            # IRR / money weighted retun
            irr_monthly = npf.irr(cash_flow)
            irr_annual = (1 + irr_monthly)**12 - 1
            # Time weighted return
            avg_log_TWR = simulated_returns.mean(axis=0).sum() / num_years


            sim_text = (
                    f"Based on each asset's historic annual mean return and standard deviation and their respective weight in "
                    f"the {selected_p} the simulation predicts the investor's capital to reach **{currency_formatter(expected_capital, currency=currency)}** "
                    f"in {current_year+num_years} based on the above specified savings plan. "
                    f"This corresponds to an expected time-weighted return of {avg_log_TWR:.2%} p.a. "
                    f"and an expected money-weighted return (IRR) of {irr_annual:.2%} p.a."
                    )
            st.write(sim_text)
            sim_outcomes = simulated_performance.iloc[-1,:]
           
            percent_in_interval = 0.0

            moe = 1.0
            while percent_in_interval <= 1.0:
                in_interval = (expected_capital - moe < sim_outcomes) & (sim_outcomes < expected_capital + moe)
                percent_in_interval = in_interval.mean()
                if round(percent_in_interval,4) == p:
                    break
                if percent_in_interval > p:
                    moe = moe*0.9999
                elif percent_in_interval > p-0.05:
                    moe = moe*1.001
                else:
                    moe = moe*1.5
            
            p_above_mean = (expected_capital < sim_outcomes).mean()
            p_below_mean = (expected_capital > sim_outcomes).mean()

            lower_bound = expected_capital - moe
            upper_bound = expected_capital + moe

            if currency == "USD":
                st.markdown(
                            f"""
                            It is important to note that {currency_formatter(expected_capital, currency=currency)} is just the simulation mean and not a guaranteed outcome:
                            - {percent_in_interval:.0%} of simulation outcomes lie between {currency_formatter(lower_bound, currency=currency)} and {currency_formatter(upper_bound, currency=currency)}
                            - {p_below_mean:.0%} of simulation outcomes lie between {currency_formatter(expected_capital, currency=currency)} and {currency_formatter(sim_outcomes.min(), currency=currency)} (the lowest simulation outcome)
                            - {p_above_mean:.0%} of simulation outcomes lie between {currency_formatter(expected_capital, currency=currency)} and {currency_formatter(sim_outcomes.max(), currency=currency)} (the highest simulation outcome)
                            """
                            )
            else:
                st.markdown(
                            f"""
                            It is important to note that {currency_formatter(expected_capital, currency=currency)} is just the simulation mean and not a guaranteed outcome:
                            - {percent_in_interval:.0%} of simulation outcomes lie between {currency_formatter(lower_bound, currency=currency)} and {currency_formatter(upper_bound, currency=currency)}
                            - {p_below_mean:.0%} of simulation outcomes lie between {currency_formatter(expected_capital, currency=currency)} and {currency_formatter(sim_outcomes.min(), currency=currency)} (the lowest simulation outcome)
                            - {p_above_mean:.0%} of simulation outcomes lie between {currency_formatter(expected_capital, currency=currency)} and {currency_formatter(sim_outcomes.max(), currency=currency)} (the highest simulation outcome)
                            """
                            )



    if option == "Data":
        st.write("Monthly adjusted closing prices:")
        st.dataframe(montly_adjusted_closing_prices)
        st.write("Monthly log returns:")
        st.dataframe(monthly_log_returns)

import numpy as np
import numpy_financial as npf
import pandas as pd
from pandas_datareader.data import DataReader as dr
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib import gridspec
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
import mplfinance as mpf
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.optimize import brentq
from scipy.optimize import Bounds
from scipy.stats import norm
import os
#import datetime
from datetime import datetime, timedelta
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

def create_performance_index(price_df):
    returns = price_df.pct_change()
    growth = returns+1
    growth = growth.fillna(1) # set starting value for index
    index = growth.cumprod()
    index = index - 1 # deduct starting value to get the percentage change
    return index

def visualize_performance(prices, list_of_names):
    benchmarking_data = create_performance_index(prices)
   
    color_list = ['deepskyblue', 'steelblue', 'mediumslateblue', 'cornflowerblue', 'lightsteelblue', 
                    'mediumslateblue', 'lightblue']

    benchmarking_data_filtered = benchmarking_data.filter(list_of_names)
    if len(list_of_names) > 0:
        benchmarking_data_filtered.plot(figsize=(15, 10), color=color_list)
    else:
        plt.figure(figsize=(15, 10))

    
    plt.fill_between(benchmarking_data.index, benchmarking_data.max(axis=1), benchmarking_data.min(axis=1),
                        color='grey', alpha=0.17, label="Range")

    # Calculate the number of days to add
    num_days = (benchmarking_data_filtered.index.max() - benchmarking_data_filtered.index.min()).days
    days_to_add1 = num_days / 120
    #days_to_add2 = num_days / 12
    days_to_add2 = 0

    # Plot scatter points at the end of each line
    for col in benchmarking_data_filtered.columns:
        #plt.scatter(benchmarking_data_filtered.index[-1], benchmarking_data_filtered[col].iloc[-1], color=color_list[list_of_names.index(col)], zorder=5)
        #text lablel is offset by a number of days to the right
        plt.text(benchmarking_data_filtered.index[-1] + pd.Timedelta(days=days_to_add1), benchmarking_data_filtered[col].iloc[-1], str(round(benchmarking_data_filtered[col].iloc[-1]*100, 2))+"%",color=color_list[list_of_names.index(col)], size=12, verticalalignment='center')
        
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
    plt.gca().xaxis.set_major_locator(MaxNLocator())
    plt.gca().set_xlim(left=benchmarking_data.head(1).index.max())

    

    plt.xlim(right=benchmarking_data.index.max() + pd.Timedelta(days=days_to_add2))  # Extend x-axis limit by number of days
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format dates to show month and year
    plt.grid('on', ls="--")
    plt.ylabel(f"Performance (indexed: {benchmarking_data.head(1).index.max().strftime('%d.%m.%Y')} = 0%)", fontsize=12)
    plt.legend(fontsize=12)

    # Rotate x-axis labels to be horizontal
    plt.xticks(rotation=0, ha='center')

    # Remove x-axis label
    plt.gca().set_xlabel('')
    plt.minorticks_off()
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
    summary_sorted.sort_values("Sharpe ratio", inplace=True)
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

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
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
    plt.ylabel("Expected return", fontsize=12)
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

def run_and_display_monte_carlo_sim():
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
            f"Based on each asset's historic mean return and standard deviation and their respective weight in "
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


def simulate_leveraged_daily_compounded_annual_return(daily_return, 
                                                      daily_vola, 
                                                      leverage, 
                                                      reference_rate, 
                                                      expense_ratio, 
                                                      assumed_trading_days,
                                                      sim_runs):
    delta_t = 1/assumed_trading_days
    daily_leverage_cost = ((leverage-1)*reference_rate + expense_ratio)*delta_t
    
    # run monte carlo simmulation
    daily_return_sim = np.log(1 + leverage*(daily_return + daily_vola*np.random.normal(0, 1, size=(sim_runs, assumed_trading_days))) - daily_leverage_cost)

    daily_compounded_annual_returns = np.sum(daily_return_sim, axis=1)

    mean_daily_compounded_annual_return = daily_compounded_annual_returns.mean()
    std_daily_compounded_annual_return = daily_compounded_annual_returns.std()
    
    return mean_daily_compounded_annual_return, std_daily_compounded_annual_return

def create_leverage_sim_visual(results_df):
    # Create figure and axis objects
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # Plot mean return on primary y-axis
    ax1.scatter(results_df['Leverage'], results_df['Mean_Return'], label='Simulated return', color='cornflowerblue')
    ax1.set_xlabel('Leverage')
    ax1.set_ylabel('Daily compounded annual return')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))

    plt.grid('on', ls="--")
    # Create secondary y-axis for standard deviation
    ax2 = ax1.twinx()
    ax2.scatter(results_df['Leverage'], results_df['Std_Return'], label='Simulated volatility', color='darkmagenta')
    ax2.set_ylabel('Volatility of annual returns')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc=0)

    # Display the plot
    plt.show()

def create_binary_colormap_for_plt_charts(data_values, two_color_list):

    cmap = matplotlib.colors.ListedColormap(two_color_list)

    # scale data
    denominator = max(data_values) - min(data_values)
    scaled_data = [(datum-min(data_values))/denominator for datum in data_values]

    colors = []
    for decimal in scaled_data:
        colors.append(cmap(decimal))

    return colors

def create_colormap_for_plt_charts(data_values, color_list):
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color_list)

    # scale data
    denominator = max(data_values) - min(data_values)
    scaled_data = [(datum-min(data_values))/denominator for datum in data_values]

    colors = []
    for decimal in scaled_data:
        colors.append(cmap(decimal))

    return colors

def run_CAPM(price_df_monthly, CAPM_data, proxys_M_rf):
  CAPM_quotes = pd.DataFrame()
  CAPM_quotes[["Market", "risk-free"]] = CAPM_data[proxys_M_rf]
  CAPM_quotes["risk-free"] = CAPM_quotes["risk-free"]/100
  CAPM_quotes = CAPM_quotes.merge(price_df_monthly, left_index=True, right_index=True, how="inner")
  CAPM_returns = np.log(CAPM_quotes / CAPM_quotes.shift(1))
  CAPM_returns["risk-free"] = CAPM_quotes["risk-free"]*(1/12)
  CAPM_returns["MRP"] = CAPM_returns["Market"] - CAPM_returns["risk-free"]

  for asset in price_df_monthly.columns:
      CAPM_returns[f"{asset}-rf"] = CAPM_returns[asset] - CAPM_returns["risk-free"]

  CAPM_returns.dropna(inplace=True)

  mean_rf = CAPM_returns["risk-free"].mean()*12
  mean_MRP = CAPM_returns["MRP"].mean()*12

  CAPM_summary = pd.DataFrame()
  for asset in price_df_monthly.columns:
      Y=CAPM_returns[f"{asset}-rf"]
      X= sm.add_constant(CAPM_returns["MRP"])
      regress = sm.OLS(Y, X)
      result = regress.fit()
      CAPM_summary.loc[asset,"Mean rf"] = mean_rf
      CAPM_summary.loc[asset,"Beta"] = float(result.params[1])
      CAPM_summary.loc[asset,"Mean MRP"] = mean_MRP
      CAPM_summary.loc[asset,"Fair return"] = CAPM_summary.loc[asset,"Mean rf"] + CAPM_summary.loc[asset,"Beta"]*CAPM_summary.loc[asset,"Mean MRP"]
      CAPM_summary.loc[asset,"Alpha"] = float(result.params[0]*12)
      CAPM_summary.loc[asset,"Mean return"] = CAPM_summary.loc[asset,"Fair return"] + CAPM_summary.loc[asset,"Alpha"]
    
  return CAPM_summary, mean_rf, mean_MRP

#Technical Analysis functions
def calculate_macd(data, price="Close", days_fast=12, days_slow=26, days_signal=9):
    short_ema = data[price].ewm(span=days_fast, adjust=False).mean()
    long_ema = data[price].ewm(span=days_slow, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=days_signal, adjust=False).mean()
    macd_hist = macd - signal
    return short_ema, long_ema, macd, signal, macd_hist

def pandas_rsi(df: pd.DataFrame, window_length: int = 14, output: str = None, price: str = 'Close'):
    """
    An implementation of Wells Wilder's RSI calculation as outlined in
    his 1978 book "New Concepts in Technical Trading Systems" which makes
    use of the α-1 Wilder Smoothing Method of calculating the average
    gains and losses across trading periods and the Pandas library.

    @author: https://github.com/alphazwest
    Args:
        df: pandas.DataFrame - a Pandas Dataframe object
        window_length: int - the period over which the RSI is calculated. Default is 14
        output: str or None - optional output path to save data as CSV
        price: str - the column name from which the RSI values are calcuated. Default is 'Close'

    Returns:
        DataFrame object with columns as such, where xxx denotes an inconsequential
        name of the provided first column:
            ['xxx', 'diff', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs', 'rsi']
    """
    # Calculate Price Differences using the column specified as price.
    df['diff1'] = df[price].diff(1)

    # Calculate Avg. Gains/Losses
    df['gain'] = df['diff1'].clip(lower=0).round(2)
    df['loss'] = df['diff1'].clip(upper=0).abs().round(2)

    # Get initial Averages
    df['avg_gain'] = df['gain'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
    df['avg_loss'] = df['loss'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]

    # Calculate Average Gains
    for i, row in enumerate(df['avg_gain'].iloc[window_length+1:]):
        df['avg_gain'].iloc[i + window_length + 1] =\
            (df['avg_gain'].iloc[i + window_length] *
             (window_length - 1) +
             df['gain'].iloc[i + window_length + 1])\
            / window_length

    # Calculate Average Losses
    for i, row in enumerate(df['avg_loss'].iloc[window_length+1:]):
        df['avg_loss'].iloc[i + window_length + 1] =\
            (df['avg_loss'].iloc[i + window_length] *
             (window_length - 1) +
             df['loss'].iloc[i + window_length + 1])\
            / window_length

    # Calculate RS Values
    df['rs'] = df['avg_gain'] / df['avg_loss']

    # Calculate RSI
    df['rsi'] = 100 - (100 / (1.0 + df['rs']))

    # Save if specified
    if output is not None:
        df.to_csv(output)

    return df

def get_cumulatieve_investment_values(returns, payments):
  sum_cum_values = pd.Series(0, index=payments.index, dtype=float)

  for i in payments[payments > 0].index:
    returns_i = returns.loc[i:]
    growth_i = 1 + returns_i
    growth_i[0] = payments[i]
    cum_value_i = growth_i.cumprod()

    sum_cum_values = sum_cum_values.add(cum_value_i, fill_value=0)

  return sum_cum_values


def apply_investment_signal(returns, starting_value, signal):
  date_index = signal.index
  values = pd.Series(starting_value, index=date_index, dtype=float)

  for i in range(1, len(date_index)):
    if signal.iloc[i] == 1:
      values.iloc[i] = values.iloc[i-1] * (1+returns.iloc[i])
    else:
      values.iloc[i] = values.iloc[i-1]

  return values

#Options
def call_payoff(S0, K):
    """ Call option payoff

    :param S0: spot price
    :param K: strike price
    :return: Call option option payoff
    """
    return np.maximum(S0 - K, 0)

def call_PnL(S0, K, C):
    """ Call option profit or loss

    :param S0: spot price
    :param K: strike price
    :param C: call option premium
    :return: Call option profit or loss
    """
    return np.maximum(S0 - K, 0) - C

def N(z):
    """ Normal cumulative density function

    :param z: point at which cumulative density is calculated 
    :return: cumulative density under normal curve
    """
    return stats.norm.cdf(z)

def black_scholes_call_value(S0, K, rf, T, vol):
    """ Black-Scholes call option

    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :return: BS call option value
    """
    d1 = (1.0/(vol * np.sqrt(T))) * (np.log(S0/K) + (rf + 0.5 * vol**2.0) * T)
    d2 = d1 - (vol * np.sqrt(T))
    
    return N(d1) * S0 - N(d2) * K * np.exp(-rf * T)

def black_scholes_put_value(S0, K, rf, T, vol):
    """ Black-Scholes put option

    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :return: BS call option value
    """
    d1 = (1.0/(vol * np.sqrt(T))) * (np.log(S0/K) + (rf + 0.5 * vol**2.0) * T)
    d2 = d1 - (vol * np.sqrt(T))
    
    return N(-d2) * K * np.exp(-rf * T) - N(-d1) * S0

def phi(x):
    """ Phi helper function
    
    """
    return np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi))

def call_delta(S0, K, rf, T, vol):
    """ Black-Scholes call delta
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :return: call delta
    """
    d1 = (1.0/(vol * np.sqrt(T))) * (np.log(S0/K) + (rf + 0.5 * vol**2.0) * T)
    
    return N(d1)

def put_delta(S0, K, rf, T, vol):
    """ Black-Scholes put delta
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :return: put delta
    """
    d1 = (1.0/(vol * np.sqrt(T))) * (np.log(S0/K) + (rf + 0.5 * vol**2.0) * T)
    
    return N(d1) - 1.0

def gamma(S0, K, rf, T, vol):
    """ Black-Scholes gamma
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :return: gamma
    """
    d1 = (1.0/(vol * np.sqrt(T))) * (np.log(S0/K) + (rf + 0.5 * vol**2.0) * T)
    
    return phi(d1) / (S0 * vol * np.sqrt(T))

def vega(S0, K, rf, T, vol):
    """ Black-Scholes vega
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :return: vega
    """
    d1 = (1.0/(vol * np.sqrt(T))) * (np.log(S0/K) + (rf + 0.5 * vol**2.0) * T)
    
    return (S0 * phi(d1) * np.sqrt(T)) / 100.0

def call_theta(S0, K, rf, T, vol):
    """ Black-Scholes call theta
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :return: call theta
    """
    d1 = (1.0/(vol * np.sqrt(T))) * (np.log(S0/K) + (rf + 0.5 * vol**2.0) * T)
    d2 = d1 - (vol * np.sqrt(T))
    
    theta = -((S0 * phi(d1) * vol) / (2.0 * np.sqrt(T))) - (rf * K * np.exp(-rf * T) * N(d2))
    return theta / 365.0

def put_theta(S0, K, rf, T, vol):
    """ Black-Scholes put theta
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :return: put theta
    """
    d1 = (1.0/(vol * np.sqrt(T))) * (np.log(S0/K) + (rf + 0.5 * vol**2.0) * T)
    d2 = d1 - (vol * np.sqrt(T))
    
    theta = -((S0 * phi(d1) * vol) / (2.0 * np.sqrt(T))) + (rf * K * np.exp(-rf * T) * N(-d2))
    return theta / 365.0

def call_rho(S0, K, rf, T, vol):
    """ Black-Scholes call rho
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :return: call rho
    """
    d1 = (1.0/(vol * np.sqrt(T))) * (np.log(S0/K) + (rf + 0.5 * vol**2.0) * T)
    d2 = d1 - (vol * np.sqrt(T))
    
    rho = K * T * np.exp(-rf * T) * N(d2)
    return rho / 100.0

def put_rho(S0, K, rf, T, vol):
    """ Black-Scholes put rho
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :return: put rho
    """
    d1 = (1.0/(vol * np.sqrt(T))) * (np.log(S0/K) + (rf + 0.5 * vol**2.0) * T)
    d2 = d1 - (vol * np.sqrt(T))
    
    rho = -K * T * np.exp(-rf * T) * N(-d2)
    return rho / 100.0

def call_omega(S0, K, rf, T, vol):
    """ Black-Scholes call omega (leverage)
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :return: call omega
    """
    return call_delta(S0, K, rf, T, vol) * (S0 / black_scholes_call_value(S0, K, rf, T, vol))

def call_implied_volatility_objective_function(S0, K, rf, T, vol, call_option_market_price):
    """ Objective function which sets market and model prices to zero
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :param call_option_market_price: market observed option price
    :return: error between market and model price
    """
    return call_option_market_price - black_scholes_call_value(S0, K, rf, T, vol)

def call_implied_volatility(S0, K, rf, T, call_option_market_price, a=-2.0, b=2.0, xtol=1e-6):
    """ Call implied volatility function
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param call_option_market_price: market observed option price
    :param a: lower bound for brentq method
    :param b: upper gound for brentq method
    :param xtol: tolerance which is considered good enough
    :return: volatility to sets the difference between market and model price to zero
    
    """
    # avoid mirroring outer scope  
    _S, _K, _r, _t, _call_option_market_price = S0, K, rf, T, call_option_market_price
    
    # define a nested function that takes our target param as the input
    def fcn(vol):
        
        # returns the difference between market and model price at given volatility
        return call_implied_volatility_objective_function(_S, _K, _r, _t, vol, _call_option_market_price)
    
    # first we try to return the results from the brentq algorithm
    try:
        result = brentq(fcn, a=a, b=b, xtol=xtol)
        
        # if the results are *too* small, sent to np.nan so we can later interpolate
        return np.nan if result <= 1.0e-6 else result
    
    # if it fails then we return np.nan so we can later interpolate the results
    except ValueError:
        return np.nan

def put_implied_volatility_objective_function(S0, K, rf, T, vol, put_option_market_price):
    """ Objective function which sets market and model prices to zero
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    :param call_option_market_price: market observed option price
    :return: error between market and model price
    """
    return put_option_market_price - black_scholes_put_value(S0, K, rf, T, vol)

def put_implied_volatility(S0, K, rf, T, put_option_market_price, a=-2.0, b=2.0, xtol=1e-6):
    """ Put implied volatility function
    
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param call_option_market_price: market observed option price
    :param a: lower bound for brentq method
    :param b: upper gound for brentq method
    :param xtol: tolerance which is considered good enough
    :return: volatility to sets the difference between market and model price to zero
    
    """
    
    # avoid mirroring out scope  
    _S, _K, _r, _t, _put_option_market_price = S0, K, rf, T, put_option_market_price
    
    # define a nsted function that takes our target param as the input
    def fcn(vol):
        
        # returns the difference between market and model price at given volatility
        return put_implied_volatility_objective_function(_S, _K, _r, _t, vol, _put_option_market_price)
    
    # first we try to return the results from the brentq algorithm
    try:
        result = brentq(fcn, a=a, b=b, xtol=xtol)
        
        # if the results are *too* small, sent to np.nan so we can later interpolate
        return np.nan if result <= 1.0e-6 else result
    
    # if it fails then we return np.nan so we can later interpolate the results
    except ValueError:
        return np.nan

def profit_long_call_at_expiry(ST, S0, K, rf, T, vol):
    """ Long call profit at expiry
    
    :param ST: target price at expiry
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    """
    return np.maximum((ST - K),0) - black_scholes_call_value(S0, K, rf, T, vol)

def return_long_call_at_expiry(ST, S0, K, rf, T, vol, pa=True):
    """ Long call profit at expiry
    
    :param pa: return p.a. or total return
    :param ST: target price at expiry
    :param S0: spot price
    :param K: strike price
    :param rf: riskfree rate
    :param T: time to expiration
    :param vol: volatility
    """
    if pa:
        return np.log(np.maximum((ST - K),0) / black_scholes_call_value(S0, K, rf, T, vol)) * (1/T)
    else:
        return np.log(np.maximum((ST - K),0) / black_scholes_call_value(S0, K, rf, T, vol))


option = st.sidebar.selectbox("What do you want to see?", ("Past performance", "Custom portfolio","Return correlation", "Portfolio optimization (Markowitz model)", "CAPM", "Daily leverage simulation", "Technical Analysis", "Call Options", "Data"))

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

    eurusd = yf.download(["EURUSD=X","EUR=X"], period='max')["Adj Close"]
    
    if len(price_df) < 1:
        st.error("(Some) assets could not be found.")
    elif len(tickers) == 1:
        st.error("Make sure to select at least two assets.")
    else:
        st.success("Assets added!")
        download_sucess = True

st.write("Disclaimer: This is not financial advice.")


currency = st.selectbox("Select a currency.", ["EUR", "USD"])


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
    
    date_range = st.slider("Define the timeframe to be considered for the analysis.", value=[start_date, end_date], min_value=start_date, max_value=end_date, format ='DD.MM.YYYY')
    start_date_from_index = daily_adjusted_closing_prices.index[(daily_adjusted_closing_prices.index.day==date_range[0].day) & (daily_adjusted_closing_prices.index.month==date_range[0].month) & (daily_adjusted_closing_prices.index.year==date_range[0].year)].min().date()
    end_date_from_index = daily_adjusted_closing_prices.index[(daily_adjusted_closing_prices.index.day==date_range[1].day) & (daily_adjusted_closing_prices.index.month==date_range[1].month) & (daily_adjusted_closing_prices.index.year==date_range[1].year)].max().date()
    daily_adjusted_closing_prices = daily_adjusted_closing_prices.loc[start_date_from_index:end_date_from_index]

    montly_adjusted_closing_prices = get_monthly_closing_prices(price_df_daily=daily_adjusted_closing_prices)

    # get 3-month T-Bill data for Sharpe ratio calculation
    UST_3_mo = dr("TB3MS", 'fred',  start=now - timedelta(days=65))
    UST_3_mo.dropna(inplace=True)
    UST_3_mo = float(UST_3_mo.iloc[-1])/100

    # get SOFR data
    SOFR_90_day = dr("SOFR90DAYAVG", 'fred',  start=now - timedelta(days=10))
    SOFR_90_day.dropna(inplace=True)
    SOFR_90_day = float(SOFR_90_day.iloc[-1])

     # calculate maximum drawdown
    max_dds = maximum_drawdowns(price_df=daily_adjusted_closing_prices)

    #montly_adjusted_closing_prices = convert_date_index(montly_adjusted_closing_prices)
    monthly_log_returns = np.log(montly_adjusted_closing_prices / montly_adjusted_closing_prices.shift(1))

    annualized_mean_returns = monthly_log_returns.mean() * 12
    annualized_std_returns = monthly_log_returns.std() * 12**0.5
    annualized_cov_returns = monthly_log_returns.cov() * 12
    corr_returns = monthly_log_returns.corr()

    summary = pd.DataFrame()

    summary["mean return"] = annualized_mean_returns
    summary["standard deviation"] = annualized_std_returns
    summary["Sharpe ratio"] = (summary["mean return"]- UST_3_mo) / summary['standard deviation']
    summary["weight"] = 1/len(summary)

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
            A=10
            KPIs_custom_p = create_KPI_report("Custom portfolio",
                             custom_p_summary["weight"],
                             UST_3_mo,
                             custom_p_summary["mean return"])

            custom_p_summary["Full name"] = custom_p_df["Full name"]
            custom_p_summary_long_name = custom_p_summary.set_index(keys="Full name")
            custom_p_summary_long_name.index.name = None
            r_custom_p = float(KPIs_custom_p["portfolio return"])
            std_custom_p = float(KPIs_custom_p["protfolio std"])
            weight_avg_TER = sum(custom_p_df["TER"] * custom_p_df["weight"])
            ytd_return_custom_p = sum(custom_p_df["YTD return"] * custom_p_df["weight"])
            custom_p_worth = sum(custom_p_df["Current value"])

            # deal with minor holdings
            custom_p_summary_long_name_adjust = custom_p_summary_long_name.copy()
            custom_p_summary_long_name_adjust["index"] = custom_p_summary_long_name.index
            custom_p_summary_long_name_adjust.loc[custom_p_summary_long_name_adjust['weight'] < 0.03, 'index'] = '*Minor holdings'
            custom_p_summary_long_name_adjust.set_index('index', inplace=True)
            custom_p_summary_long_name_adjust = custom_p_summary_long_name_adjust.groupby('index')['weight'].sum().reset_index()
            custom_p_summary_long_name_adjust.set_index('index', inplace=True)

            # list minor holdings
            df_minor_hold = custom_p_summary_long_name[custom_p_summary_long_name["weight"] < 0.03]
            string_list = []
            # Iterate through rows and create string format
            for index, row in df_minor_hold.iterrows():
                weight_percent = "{:.2%}".format(row['weight'])  # Format weight as percentage with 2 decimal places
                string_list.append(f"{weight_percent} {index}")
            

    st.subheader(option)
    if option == "Past performance":
        display_summary = pd.DataFrame()
        display_summary["Full name"] = pd.Series(long_name_dict)
        display_summary["Mean return p.a."] = summary["mean return"].map('{:.2%}'.format)
        display_summary["Volatility p.a."] = summary["standard deviation"].map('{:.2%}'.format)
        display_summary["Sharpe ratio*"] = summary["Sharpe ratio"]
        display_summary["YTD return"] = ytd_returns.map('{:.2%}'.format)
        display_summary["Maximum drawdown"] = max_dds.map('{:.2%}'.format)
        #display_summary["Mean return p.a."] = display_summary["Mean return p.a."].map('{:.2%}'.format)
        #display_summary["Volatility p.a."] = display_summary["Volatility p.a."].map('{:.2%}'.format)
        #display_summary["YTD return"] = display_summary["YTD return"].map('{:.2%}'.format)
        display_summary.sort_values("Sharpe ratio*", inplace=True, ascending=False)
        st.dataframe(display_summary)
        st.markdown(f"<div style='font-size: 13px;'>* The current 3-month U.S. T-bill yield of {UST_3_mo*100}% is used as a proxy for the risk-free rate in the Sharpe ratio calculation.</div>", unsafe_allow_html=True)
        st.markdown("")
        
        visualize_summary(summary)
        st.pyplot()

        tickers_chosen = st.multiselect("Select the assets you want to compare:", tickers)
        visualize_performance(daily_adjusted_closing_prices, tickers_chosen)
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
                create_portfolio_visual(f'{currency_formatter_signs(custom_p_worth, currency=currency)}', custom_p_summary_long_name_adjust, KPIs_custom_p)
                st.pyplot()
            elif display_option == "Asset class view": 
                create_portfolio_visual(f'{currency_formatter_signs(custom_p_worth, currency=currency)}', asset_class_df, KPIs_custom_p)
                st.pyplot()

            if len(string_list) > 0:
                # display minor holdings
                st.markdown("<div style='font-size: 13px;'><b>*Minor holdings:</b></div>", unsafe_allow_html=True)
                s = ''
                for i in string_list:
                    s += f"&nbsp;&nbsp;•  {i}<br>"  # Add non-breaking space to maintain indentation 
                st.markdown(f"<div style='font-size: 13px;'>{s}</div>", unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")

            headline1 = "Benchmarking"
            st.write(f"**{headline1}**")
            
            # benchmark performance
            #custom_p_weighted_daily_price_df = np.sum((daily_adjusted_closing_prices[custom_p_df.index] * custom_p_df["weight"]), axis=1)
            #custom_p_weighted_daily_price_df.rename("Portfolio", inplace=True)
            #custom_p_weighted_monthly_price_df = get_monthly_closing_prices(price_df_daily=custom_p_weighted_daily_price_df).to_frame()
            custom_p_monthly_price_df = montly_adjusted_closing_prices[custom_p_df.index]

            benchmark_p_input = st.text_input("As per default, the custom portfolio is benchmarked against the S&P 500 Index (^GSPC). If you consider another index more suitable for your analysis, you can enter its [Yahoo Finace](https://finance.yahoo.com) ticker below (E.g. STOXX Europe 600: ^STOXX, Dax-Performance-Index: ^GDAXI, FTSE 100 Index: ^FTSE)")
            benchmark_rf_input = st.text_input("As per default, 10-year U.S. Treasury yields (^TNX) are used as the benchmark risk-free rate. You may enter the ticker of a different proxy below (make sure the proxy is quoted in yields, not prices; e.g. 13-week U.S. Treasury yields: ^IRX, 5-year U.S. Treasury yields: ^FVX, 30-year U.S. Treasury yields: ^TYX)")
    
            if benchmark_p_input:
                benchmark_p = benchmark_p_input.upper()
            else:
                benchmark_p = "^GSPC"
            if benchmark_rf_input:
                benchmark_rf = benchmark_rf_input.upper()
            else:
                benchmark_rf = "^TNX"
    
            benchmarks_p_rf = [benchmark_p, benchmark_rf]
    
            CAPM_data_daily = yf.download(benchmarks_p_rf, period='max')["Adj Close"]
            CAPM_data_daily.dropna(inplace=True)
    
            download_sucess2 = False
            if len(CAPM_data_daily) < 1:
                st.error("Asset could not be found.")
            else:
                if benchmark_p_input or benchmark_rf_input:
                    st.success("Benchmark updated!")
                download_sucess2 = True
                CAPM_data = get_monthly_closing_prices(price_df_daily=CAPM_data_daily)
    
            if download_sucess2:
                CAPM_output = run_CAPM(custom_p_monthly_price_df, CAPM_data, benchmarks_p_rf)
                CAPM_summary = CAPM_output[0]
                mean_rf = CAPM_output[1]
                mean_MRP = CAPM_output[2]
                Beta_p = np.sum((CAPM_summary["Beta"] * custom_p_df["weight"]), axis=0)
                Expected_r_p_CAPM = np.sum((CAPM_summary["Mean return"] * custom_p_df["weight"]), axis=0)
                Alpha_p = Expected_r_p_CAPM - (mean_rf + Beta_p*mean_MRP)
                
                col1, col2 = st.columns([1,3])
                col1.metric("Portfolio Beta", f"{Beta_p:.2f}")
                col2.metric("Portfolio Alpha", f"{Alpha_p:.2%}")

                custom_p_weighted_daily_price_df = np.sum((daily_adjusted_closing_prices[custom_p_df.index] * custom_p_df["weight"]), axis=1)
                custom_p_weighted_daily_price_df.rename("Portfolio", inplace=True)
                benchmarking_df = custom_p_weighted_daily_price_df.to_frame().merge(CAPM_data_daily[benchmark_p], left_index=True, right_index=True, how="inner")
                #st.dataframe(benchmarking_df)
                visualize_performance(benchmarking_df, list(benchmarking_df.columns))
                st.pyplot()
                st.markdown("")
                st.markdown("")
            
            headline2 = "Savings plan simulation"
            st.write(f"**{headline2}**")
            # Simulate performance
            num_trials = 10000
           
            val_today_assume = custom_p_worth
            val_today = st.number_input("What is your starting capital i.e. the ammount of money you can invest today?",min_value=1.0, value=val_today_assume)
    
            #Ask user to enter amount of money they want to save each year
            additional_investment_per_month = st.number_input("How much money do you want to save each month?",min_value=0.0, value=100.0)
            additional_investment_per_year = additional_investment_per_month*12
            
            sim_summary = custom_p_summary.copy()
            sim_summary["mean return"] = custom_p_summary["mean return"] / 12
            sim_summary["standard deviation"] = custom_p_summary["standard deviation"] / (12**0.5)
            num_years = st.slider("For how many years do you want to save?",
                              min_value=1, max_value=100, step=1, value=20)
            num_months = 12*num_years
    
            p = st.slider("Define the percentage of simulation outcomes to be contained in a symmetrical bandwidth around the simulation mean:",
                          min_value=0.05, max_value=0.95, step=0.05, value=0.8)
            
            current_year = datetime.now().year
            total_additional_investments = num_years * additional_investment_per_year
            selected_p = "custom portfolio"
            
            if st.button("Run simulation"):
                run_and_display_monte_carlo_sim()

            
        else:
            st.write("Upload a filled out Excel template to view your custom portfolio.")    
        

    if option == "Return correlation":
        visualize_correlation(corr_returns)
        st.pyplot()
    

    if option == "Portfolio optimization (Markowitz model)":
        rf_l = st.slider("What is your risk-free interest rate for lending money?",
                min_value=0.0, max_value=10.0, step=0.01, format='%.2f%%')
        rf_l = rf_l/100
        
        rf_b = st.slider("What is your risk-free interest rate for borrowing money?",
                        min_value=0.0, max_value=10.0, step=0.01, format='%.2f%%')
        rf_b = rf_b/100
        
        A = st.slider("Adjust your risk aversion parameter (higher = more risk averse, lower = less risk averse).",
                        min_value=0.1, max_value=15.0, step=0.1, value=10.0)
        
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

        headline1 = "Minimum varriance frontier and capital allocation line"
        st.write(f"**{headline1}**")
        create_mvf_cal_visual()
        st.pyplot()
        
        headline2 = "Output portfolios"
        st.write(f"**{headline2}**")
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
        
        headline3 = "Savings plan simulation"
        st.write(f"**{headline3}**")
        # Simulate performance
        num_trials = 10000

        r_ocp = float(KPIs_ocp["portfolio return"])
        std_ocp = float(KPIs_ocp["protfolio std"])

        r_orp = float(KPIs_orp["portfolio return"])
        std_orp = float(KPIs_orp["protfolio std"])

        r_mvp = float(KPIs_mvp["portfolio return"])
        std_mvp = float(KPIs_mvp["protfolio std"])

       
        portfolios = ["OCP", "ORP", "MVP"]
        val_today_assume = 1000.00
        val_today = st.number_input("What is your starting capital i.e. the ammount of money you can invest today?",min_value=1.0, value=val_today_assume)

        #Ask user to enter amount of money they want to save each year
        additional_investment_per_month = st.number_input("How much money do you want to save each month?",min_value=0.0, value=100.0)
        additional_investment_per_year = additional_investment_per_month*12

        selected_p = st.selectbox("What portfolio do you want to invest in?", portfolios)

        col1, col2 = st.columns([1,3])
        
        # Here the "abrev" version of the summary must be used. Just the sumamry is not correct anymore at this point for some reason.
        if selected_p == "MVP":
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
        
        current_year = datetime.now().year

        
        total_additional_investments = num_years * additional_investment_per_year
        
        if st.button("Run simulation"):
            run_and_display_monte_carlo_sim()

    if option == "CAPM":
        
        market_proxy_input = st.text_input("As per default, the S&P 500 Index (^GSPC) is used as a proxy for the market portfolio. If you consider another index more suitable for your analysis, you can enter its [Yahoo Finace](https://finance.yahoo.com) ticker below (E.g. STOXX Europe 600: ^STOXX, Dax-Performance-Index: ^GDAXI, FTSE 100 Index: ^FTSE)")
        riskfree_proxy_input = st.text_input("As per default, 10-year U.S. Treasury yields (^TNX) are used as a proxy for the risk-free rate. You may enter the ticker of a different proxy below (make sure the proxy is quoted in yields, not prices; e.g. 13-week U.S. Treasury yields: ^IRX, 5-year U.S. Treasury yields: ^FVX, 30-year U.S. Treasury yields: ^TYX)")

        if market_proxy_input:
            market_proxy = market_proxy_input.upper()
        else:
            market_proxy = "^GSPC"
        if riskfree_proxy_input:
            riskfree_proxy = riskfree_proxy_input.upper()
        else:
            riskfree_proxy = "^TNX"

        proxys_M_rf = [market_proxy, riskfree_proxy]

        CAPM_data = yf.download(proxys_M_rf, period='max')["Adj Close"]
        CAPM_data.dropna(inplace=True) 
        

        download_sucess3 = False
        if len(CAPM_data) < 1:
            st.error("Asset could not be found.")
        else:
            if market_proxy_input or riskfree_proxy_input:
                st.success("Proxy updated!")
            download_sucess3 = True
            CAPM_data = get_monthly_closing_prices(price_df_daily=CAPM_data)

        if download_sucess3:
            CAPM_output = run_CAPM(montly_adjusted_closing_prices, CAPM_data, proxys_M_rf)
            CAPM_summary = CAPM_output[0]
            mean_rf = CAPM_output[1]
            mean_MRP = CAPM_output[2]
            
            display_CAPM_summary = pd.DataFrame()
            display_CAPM_summary["Mean rf"] = CAPM_summary["Mean rf"].map('{:.2%}'.format)
            display_CAPM_summary["Beta"] = CAPM_summary["Beta"].map('{:.2f}'.format)
            display_CAPM_summary["Mean MRP"] = CAPM_summary["Mean MRP"].map('{:.2%}'.format)
            display_CAPM_summary["Fair Return"] = CAPM_summary["Fair return"].map('{:.2%}'.format)
            display_CAPM_summary["Alpha"] = CAPM_summary["Alpha"].map('{:.2%}'.format)
            display_CAPM_summary["Mean Return"] = CAPM_summary["Mean return"].map('{:.2%}'.format)
            st.dataframe(display_CAPM_summary)

            #Visualize SML
            step = 0.001

            color1 = 'cornflowerblue'
            color2 = 'darkmagenta'

            plt.figure(figsize=(15, 10))

            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:,.2}'.format))
            plt.gca().set_xlim(left=min(min(CAPM_summary["Beta"])*1.5,0))
            plt.gca().set_xlim(right=max(CAPM_summary["Beta"])*1.5)

            plt.scatter(CAPM_summary["Beta"], CAPM_summary["Mean return"], color=color2, label="Mean return")

            for i in summary.index:
                alpha_y = np.arange(min(CAPM_summary["Fair return"][i],CAPM_summary["Mean return"][i]), 
                                    max(CAPM_summary["Fair return"][i],CAPM_summary["Mean return"][i]), step)
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
                labels.append(plt.text(CAPM_summary["Beta"][i], CAPM_summary["Mean return"][i], i, size=8))
            adjust_text(labels) 
            
            plt.show()
            st.pyplot()

    
    if option == "Daily leverage simulation":
        daily_log_returns = np.log(daily_adjusted_closing_prices / daily_adjusted_closing_prices.shift(1))
        daily_mean_returns = daily_log_returns.mean() 
        daily_std_returns = daily_log_returns.std() 

        asset_name = st.selectbox("Select the asset you want to lever:", tickers)
        daily_return = daily_mean_returns[asset_name]
        daily_vola = daily_std_returns[asset_name]
        
        assumed_trading_days = 252
        
        reference_rate = st.number_input("Enter a reference interest rate in percent p.a. The default value is the current 90-day average Secured Overnight Financing Rate (SOFR).", value=SOFR_90_day)
        reference_rate = reference_rate/100
        expense_ratio = st.number_input("If applicable, enter an expense ratio in percent p.a. This is appropriate when comparing leveraged ETFs to their unleveraged equivalents, as leveraged ETFs are typically more expensive.", value=0.00)
        expense_ratio = expense_ratio/100
        sim_runs = st.number_input("Choose a number of simulation runs", value=80000)


        # Define leverage levels
        step_ = 0.1
        leverage_levels = np.arange(1, 4 + step_, step_)
        
        # Initialize an empty list to store DataFrames
        dfs = []

        if st.button("Run simulation"):

            # Initialize progress bar
            progress_text = "Simulation in progress..."
            progress_bar = st.progress(0, text=progress_text)
            i = 0
            
            # Iterate over each leverage level
            for leverage in leverage_levels:
                # Call the function for each leverage level
                mean_return, std_return = simulate_leveraged_daily_compounded_annual_return(daily_return, 
                                                                                            daily_vola, 
                                                                                            leverage, 
                                                                                            reference_rate, 
                                                                                            expense_ratio, 
                                                                                            assumed_trading_days,
                                                                                            sim_runs)
                # Create a DataFrame for current leverage level
                df = pd.DataFrame({'Leverage': [leverage], 
                                   'Mean_Return': [mean_return], 
                                   'Std_Return': [std_return]})
                
                # Append the DataFrame to the list
                dfs.append(df)

                # Update progress bar
                progress_percent = i / (len(leverage_levels)-1)
                progress_bar.progress(progress_percent, text=progress_text)
                i += 1
            
            # Concatenate all DataFrames in the list along the rows axis
            results_df = pd.concat(dfs, ignore_index=True)
            
            create_leverage_sim_visual(results_df)
            st.pyplot()

            # Remove progress bar once completed
            progress_bar.empty()

            row_highest_return = results_df.loc[results_df['Mean_Return'].idxmax()]
            sim_mean_daily_compounded_leveraged_annual_return = float(row_highest_return["Mean_Return"])
            sim_std_daily_compounded_leveraged_annual_returns = float(row_highest_return["Std_Return"])
            optimal_leverage = float(row_highest_return["Leverage"])

            row_lev_1 = results_df.loc[results_df['Leverage']==1]
            sim_mean_daily_compounded_unleveraged_annual_return = float(row_lev_1["Mean_Return"])
            sim_std_daily_compounded_unleveraged_annual_returns = float(row_lev_1["Std_Return"])
            leverage = float(row_lev_1["Leverage"])

            delta_returns = sim_mean_daily_compounded_leveraged_annual_return - sim_mean_daily_compounded_unleveraged_annual_return

            st.write("Highest simulated daily compounded annual return (levered)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean return p.a.", f"{sim_mean_daily_compounded_leveraged_annual_return:.2%}", f"{delta_returns:.2%}")
            col2.metric("Volatility p.a.", f"{sim_std_daily_compounded_leveraged_annual_returns:.2%}")
            col3.metric("Optimal leverage", f"{optimal_leverage:.1f}x")

            st.write("Simulated daily compounded annual return (unlevered)")
            col4, col5, col6 = st.columns(3)
            col4.metric("Mean return p.a.", f"{sim_mean_daily_compounded_unleveraged_annual_return:.2%}")
            col5.metric("Volatility p.a.", f"{sim_std_daily_compounded_unleveraged_annual_returns:.2%}")
            col6.metric("Leverage", f"{leverage:.1f}x")

        st.write("The simulation is based on Equation (20) in the 2009 research paper Path-dependence of Leveraged ETF returns by Marco Avellaneda & Stanley Zhang [Avellaneda & Zhang (2009)](https://doi.org/10.1137/090760805).")

    if option == "Technical Analysis":
        asset_name = st.selectbox("Select the asset you want to analyze", tickers)
        asset_data = yf.download(asset_name)

        days_back_period = st.number_input("Days back window", value=int(min(100,len(asset_data)*(9/10))), min_value=50, max_value=int(len(asset_data)*(9/10)))
        
        days_ema = st.number_input("Days exponential moving average (EMA)", value=int(min(200, len(asset_data)/10)), min_value=1, max_value=int(len(asset_data)/10))
        asset_data["ema"] = asset_data["Close"].ewm(span=days_ema, adjust=False).mean()
        
        asset_data = asset_data.tail(days_back_period)

        period_RSI=14
        asset_data = pandas_rsi(df=asset_data, window_length=period_RSI, price="Close")
        asset_data["macd_short_ema"], asset_data["macd_long_ema"], asset_data["macd"], asset_data["macd_signal"], asset_data["macd_hist"] = calculate_macd(asset_data, price="Close", days_fast=12, days_slow=26, days_signal=9)
        asset_data["70_line"] = 70
        asset_data["30_line"] = 30
        
        asset_data["daily_return"] = asset_data["Close"].pct_change()
        asset_data = asset_data.dropna()

        #date formating
        if days_back_period <= 270:
            date_format = '%b %d %Y'
        else:
            date_format = '%b %Y'
        
        # binary red green colormap
        #rg_scaled_colors = create_binary_colormap_for_plt_charts(asset_data["macd_hist"], ["#fd6b6c","#4dc790"])

        #plot mpf charts
        fig = plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(4, 1, height_ratios=[4,1,2,2])
        
        ax4 = plt.subplot(gs[3])
        ax1 = plt.subplot(gs[0], sharex=ax4)
        ax2 = plt.subplot(gs[1], sharex=ax4)
        ax3 = plt.subplot(gs[2], sharex=ax4)
        
        ap0 = [
            # ema
            mpf.make_addplot(data=asset_data["ema"], type="line", width=1.0, color="cornflowerblue", label=f"{days_ema}-day EMA", ax=ax1),
        
            # macd
            mpf.make_addplot((asset_data['macd']), type="line", width=1.0, color='cornflowerblue', ylabel='MACD', label="MACD", ax=ax3),
            mpf.make_addplot((asset_data['macd_signal']), type="line", width=1.0, color='#FFBF00', label="Signal Line", ax=ax3),
            mpf.make_addplot((asset_data['macd_hist'])*(asset_data['macd_hist']>=0), type='bar', color="#4dc790", ax=ax3),
            mpf.make_addplot((asset_data['macd_hist'])*(asset_data['macd_hist']<0), type='bar', color="#fd6b6c", ax=ax3),
        
            # rsi
            mpf.make_addplot(asset_data['rsi'], type="line", width=1.0, ylim=[0, 100], color='cornflowerblue', ylabel='RSI', ax=ax4),
            mpf.make_addplot(asset_data['70_line'], type="line", linestyle='--', width=1.0, color="#fd6b6c", label='Overbought (70)', ax=ax4),
            mpf.make_addplot(asset_data['30_line'], type="line", linestyle='--', width=1.0, color="#4dc790", label='Oversold (30)', ax=ax4)
        ]
        
        s = mpf.make_mpf_style(base_mpf_style="yahoo", y_on_right=False)
        
        mpf.plot(asset_data,
                 ax=ax1,
                 style=s,
                 addplot=ap0,
                 volume=ax2,
                 datetime_format=date_format,
                 xlim=(asset_data.head(1).index.max() - pd.Timedelta(days=1), asset_data.tail(1).index.max() + pd.Timedelta(days=1)),
                 type="candle")
        fig.subplots_adjust(hspace=0.1)
        ax1.tick_params(labelbottom=False, bottom=False, top=False)
        ax1.grid('on', ls="--")
        ax1.legend(fontsize=10, loc=2)
        ax2.tick_params(labelbottom=False, bottom=False, top=False)
        ax2.grid('on', ls="--")
        ax3.tick_params(labelbottom=False, bottom=False, top=False)
        ax3.grid('on', ls="--")
        ax3.legend(fontsize=10, loc=2)
        ax3.yaxis.set_label_position('left')
        ax3.yaxis.tick_left()
        ax4.grid('on', ls="--")
        ax4.legend(fontsize=10, loc=2)
        
        plt.style.use("default")
        plt.minorticks_off()
        
        # Rotate x-axis labels to be horizontal
        plt.xticks(rotation=0, ha='center')
        
        plt.show()
        txt = "Chart"
        st.write(f"**{txt}**")
        st.pyplot()

        #backtest
        end_of_month_dates = get_monthly_closing_prices(asset_data).index
        asset_data["monthly_payments"] = 0
        asset_data.loc[asset_data.index.isin(end_of_month_dates), "monthly_payments"] = 100 / len(end_of_month_dates)
        asset_data["start_100"] = 0
        asset_data["start_100"][0] = 100
        
        
        asset_data["benchmark_monthly_payments"] = get_cumulatieve_investment_values(asset_data["daily_return"], asset_data["monthly_payments"])
        asset_data["benchmark_payment_t0"] = get_cumulatieve_investment_values(asset_data["daily_return"], asset_data["start_100"])
        
        # MACD trigger and signal
        asset_data["macd_trigger"] = np.where((asset_data['macd_hist'] > 0) & (asset_data['macd_hist'].shift(1) <= 0), 1, np.where((asset_data['macd_hist'] < 0) & (asset_data['macd_hist'].shift(1) >= 0), -1,0))
        asset_data["macd_signal"] = asset_data["macd_trigger"].cumsum().shift(1)
        # Normalize the signals to 0,1 format
        if asset_data["macd_signal"].min(skipna=True) < 0:
          asset_data["macd_signal"] = asset_data["macd_signal"] + 1
        
        asset_data["macd_backtest"] = apply_investment_signal(asset_data["daily_return"], 100, asset_data["macd_signal"])
        
        # RSI trigger and signal
        # Initialize the trigger column with zeros
        asset_data["rsi_trigger"] = 0
        
        # Identify the conditions for trigger changes
        cross_70 = (asset_data['rsi'] <= 70) & (asset_data['rsi'].shift(1) > 70)
        cross_30 = (asset_data['rsi'] >= 30) & (asset_data['rsi'].shift(1) < 30)
        
        # Loop through the rows and update the trigger column
        prev_trigger = 0  # Initialize previous trigger value
        for index, row in asset_data.iterrows():
            if cross_70[index]:
                if prev_trigger != -1:  # Check if previous trigger was not -1
                    asset_data.at[index, "rsi_trigger"] = -1
                    prev_trigger = -1
            elif cross_30[index]:
                if prev_trigger != 1:  # Check if previous trigger was not +1
                    asset_data.at[index, "rsi_trigger"] = 1
                    prev_trigger = 1
        
        asset_data["rsi_signal"] = asset_data["rsi_trigger"].cumsum().shift(1)
        # Normalize the signals to 0,1 format
        if asset_data["rsi_signal"].min(skipna=True) < 0:
          asset_data["rsi_signal"] = asset_data["rsi_signal"] + 1
        
        asset_data["rsi_backtest"] = apply_investment_signal(asset_data["daily_return"], 100, asset_data["rsi_signal"])
        
        asset_data_visual = pd.DataFrame()
        asset_data_visual[["Benchmark", "MACD Backtest", "RSI Backtest"]] = asset_data[["benchmark_payment_t0", "macd_backtest", "rsi_backtest"]]
        
        visualize_performance(asset_data_visual, ["Benchmark", "MACD Backtest", "RSI Backtest"])
        
        txt_ = "Backtest"
        st.write(f"**{txt_}**")
        st.pyplot()


    if option == "Call Options":
        asset_name = st.selectbox("Select an underlying", tickers)
        exchanges = mcal.get_calendar_names()
        default_exchange = exchanges.index("NYSE")
        exchange = st.selectbox("Select exchange where the underlying is traded", exchanges, index=default_exchange)

        # maybe add start date input
        input_start_date = None

        # Download stock data
        price_data = yf.download(asset_name, start=input_start_date)[["Close", "Adj Close"]]
        start_date = price_data.index.min()
        return_data = np.log(price_data["Adj Close"]/price_data["Adj Close"].shift(1)).dropna()
        return_data = np.sort(return_data)

        spot_price = price_data["Close"].iloc[-1]
        strike_price = st.number_input("Strike price", value=round(spot_price/100,0)*100)
        current_date = datetime.now()
        expiration_date = st.date_input("Expiration date (dd/mm/yyyy)", value=current_date + timedelta(days=3*365), min_value=current_date + timedelta(days=1), format="DD.MM.YYYY")
        cal = mcal.get_calendar(exchange)
        trading_days = mcal.date_range(cal.schedule(start_date=current_date, end_date=expiration_date), frequency='1D')
        number_trading_days = len(trading_days) - 1                 #excluding current day from trading days
        
        trading_days_per_year = 252                                 #fair to assume 252
        time_in_years = number_trading_days/trading_days_per_year
        delta_t = time_in_years / number_trading_days               #leangth of time step


        # calculate riskfree-rate via the Nelson-Siegel-Svensson method based on U.S. Treasury yields

        # get yield curve data 
        syms = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
        yc = dr(syms, 'fred', start=current_date - timedelta(days=10))
        yield_maturities = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
        yeilds = np.array(yc.iloc[-1,:]).astype(float)/100
        
        #NSS model calibrate
        curve_fit, status = calibrate_nss_ols(yield_maturities,yeilds) 
        
        # risk-free rate
        rf = curve_fit(time_in_years)

        #subscription ratio
        subscription_ratio = st.number_input("Subscription ratio (how many units of the underlying the option refers to)", value=0.01)

        #default_selected_vol = return_data.std() * (trading_days_per_year)**0.5
        #default_black_scholes_value = black_scholes_call_value(S0=spot_price, K=strike_price, rf=rf, T=time_in_years, vol=default_selected_vol) * subscription_ratio
        historic_vol = return_data.std() * (trading_days_per_year)**0.5
        #black_scholes_value = st.number_input("Call opiton price – default assumes volatility equalt to annualized std of daily log returns", value=black_scholes_call_value(S0=spot_price, K=strike_price, rf=rf, T=time_in_years, vol=historic_vol) * subscription_ratio)

        
        #default_selected_vol = call_implied_volatility(S0=spot_price, K=strike_price, rf=rf, T=time_in_years, call_option_market_price = adj_price, a=-2.0, b=2.0, xtol=1e-6)
        #call_implied_volatility(S0=spot_price, K=strike_price, rf=rf, T=time_in_years, call_option_market_price = adj_price, a=-2.0, b=2.0, xtol=1e-6)*100
        #selected_vol = st.number_input("Volatility (in %) – default based on historic standard deviation of returns", value=historic_vol*100)/100
        selected_vol = st.number_input("Volatility (in %) – default 20%", value=20.00)/100

        default_target_price = spot_price * np.exp(return_data.mean()*number_trading_days)
        target_price = st.number_input("Expected price of underlying at expiration – default based on historic mean return", value=default_target_price)
        
        black_scholes_value = black_scholes_call_value(S0=spot_price, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol) * subscription_ratio
        #adj_price = black_scholes_value * (1/subscription_ratio)

        call_price_at_purchase = st.number_input("Call opiton price at purchase – default equal to current price", value=black_scholes_value)
        breakeven_at_expiration = strike_price + call_price_at_purchase/subscription_ratio
        
        call_delta_spot = call_delta(S0=spot_price, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol)
        call_omega_spot = call_omega(S0=spot_price, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol)

        req_r_be = np.log(breakeven_at_expiration/spot_price) * (1/time_in_years)
        
        # Row 1
        colmn_1, colmn_2, colmn_3, colmn_4 = st.columns([0.5, 0.5, 0.5, 0.5]) 
        colmn_1.metric("Black-Scholes option value", f"{black_scholes_value:.2f}")
        colmn_2.metric("Spot price underlying", f"{spot_price:.2f}")
        colmn_3.metric("Breakeven price at expiration", f"{breakeven_at_expiration:.2f}")
        colmn_4.metric("Trading days to expiration", f"{number_trading_days:.0f}")
        # Row 2
        colmn_5, colmn_6, colmn_7, colmn_8 = st.columns([0.5, 0.5, 0.5, 0.5]) 
        colmn_5.metric("Delta", f"{call_delta_spot:.2f}")
        colmn_6.metric("Omega (leverage)", f"{call_omega_spot:.2f}x")
        colmn_7.metric("Return to reach BE (p.a.)", f"{req_r_be:.2%}")
        colmn_8.metric("Assumed risk-free rate", f"{rf:.2%}")

        headline00 = "Volatility guidance"
        st.write(f"**{headline00}**")
        
        #option_price = st.number_input("Input current call opiton price", value=black_scholes_value)
        option_price = st.number_input("Input current call opiton price")
        adj_price = option_price * (1/subscription_ratio)
        impl_vol = call_implied_volatility(S0=spot_price, K=strike_price, rf=rf, T=time_in_years, call_option_market_price = adj_price, a=-2.0, b=2.0, xtol=1e-6)
        st.metric("Implied volatility", f"{impl_vol:.2%}")
        
        rolling_window = st.number_input("Input number of days for rolling volatility calculation", value=30)
        vol_start_date = st.date_input("Set start date for long term volatility calculation", value=price_data.index.min(), max_value= current_date - timedelta(days=3*365), min_value=price_data.index.min(), format="DD.MM.YYYY")
        vol_start_date = pd.Timestamp(vol_start_date)
        lookback_years = st.number_input("How many previous years shall be displayed?", value=3)

        # Create vol_data as a copy of price_data
        vol_data = price_data.copy()
        
        # Calculate log returns in vol_data
        vol_data["Log returns"] = np.log(vol_data["Adj Close"] / vol_data["Adj Close"].shift(1))
        
        # Filter vol_data for long-run volatility calculation
        vol_data = vol_data[vol_data.index >= vol_start_date]
        
        # Calculate long-run volatility
        long_run_vol = vol_data["Log returns"].std() * np.sqrt(trading_days_per_year)
        
        # Calculate rolling volatility in vol_data
        vol_data["Rolling vol"] = (
            vol_data["Log returns"]
            .rolling(rolling_window)
            .std()
            * np.sqrt(trading_days_per_year)
        )
        
        # Filter vol_data for the chart based on lookback_years
        chart_start_date = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        vol_data_chart = vol_data[vol_data.index >= chart_start_date]
        
        # Plotting
        
        # Calculate values for annotations
        last_rolling_vol = vol_data_chart["Rolling vol"].iloc[-1]
        last_rolling_vol_date = vol_data_chart.index[-1]
        days_to_add = (vol_data_chart.index.max() - vol_data_chart.index.min()).days / 120
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.gca().set_xlim(left=vol_data_chart.index.min(), right=vol_data_chart.index.max())
        
        # Plot rolling volatility
        plt.plot(vol_data_chart.index, vol_data_chart["Rolling vol"], label=f"Rolling volatility ({rolling_window} days)", color='cornflowerblue', linewidth=1.25)
        
        # Extract and format the start date for long-run volatility
        vol_start_date_str = pd.Timestamp(vol_start_date).strftime("%b %Y")
        
        # Plot long-run volatility as a horizontal line
        plt.axhline(long_run_vol, color='darkmagenta', linestyle="--", linewidth=1.25, label=f"Long-run average volatility (since {vol_start_date_str})")
        
        # Annotations for rolling volatility
        plt.text(
            last_rolling_vol_date + pd.Timedelta(days=days_to_add), 
            last_rolling_vol, 
            f"{last_rolling_vol:.2%}", 
            color='cornflowerblue', 
            verticalalignment='center'
        )
        
        # Annotations for long-run volatility
        plt.text(
            last_rolling_vol_date + pd.Timedelta(days=days_to_add), 
            long_run_vol, 
            f"{long_run_vol:.2%}", 
            color='darkmagenta', 
            verticalalignment='center'
        )
        
        # Chart details
        plt.grid('on', linestyle="--")
        plt.title("Historic volatility")
        plt.ylabel("Volatility (annualized)")
        plt.legend()
        
        # Format x-axis to show months and years
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(MaxNLocator())
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
        plt.minorticks_off()
        
        plt.tight_layout()
        plt.show()
        st.pyplot()

        
        
        headline0 = "Strike price guidance"
        st.write(f"**{headline0}**")
        
        strike_prices = np.arange(0.001, 2*target_price)
        returns_long_call_at_expiry = return_long_call_at_expiry(ST=target_price, S0=spot_price, K=strike_prices, rf=rf, T=time_in_years, vol=selected_vol, pa=True)
        returns_long_call_series = pd.Series(returns_long_call_at_expiry, index=strike_prices, name='Returns Long Call at Expiry')
        optimal_strike_price = returns_long_call_series.idxmax()
        color1 = 'cornflowerblue'
        color2 = 'darkmagenta'
        color3 = "deepskyblue"
        color4 = "slategrey"
        max_return_long_call_at_expiry = returns_long_call_at_expiry.max()
        top_border = max_return_long_call_at_expiry*1.2
        plt.figure(figsize=(7, 4))
        plt.gca().set_xlim(left=0.3*target_price, right=1.2*target_price)
        plt.gca().set_ylim(bottom=0, top=top_border)
        plt.plot(strike_prices, returns_long_call_at_expiry, color=color1)
        # Text alignment
        if spot_price >= optimal_strike_price:
            ha_spot = "left"
            offset_spot = 1.01
            ha_opt = "right"
            offset_opt = 0.99
        else:
            ha_spot = "right"
            offset_spot = 0.99
            ha_opt = "left"
            offset_opt = 1.01
        # Target price
        plt.axvline(target_price, color=color3, linewidth=1.5, label="Target price")
        plt.text(target_price*1.01, top_border*0.01, str(round(target_price, 2)),color=color3, verticalalignment='bottom')
        # Optimal strike price
        plt.axvline(optimal_strike_price, color=color2, linewidth=1.5, label="Optimal strike")
        plt.text(optimal_strike_price*offset_opt, top_border*0.01, str(round(optimal_strike_price, 2)),color=color2, ha=ha_opt,  verticalalignment='bottom')
        plt.scatter(optimal_strike_price, max_return_long_call_at_expiry, color=color2, zorder=5)
        plt.text(optimal_strike_price*offset_opt, max_return_long_call_at_expiry*offset_opt, f'{max_return_long_call_at_expiry:.2%} ', color=color2, ha=ha_opt, va='bottom')
        # Spot price
        plt.axvline(spot_price, color=color4, linewidth=1.5, label="Spot price")
        plt.text(spot_price*offset_spot, top_border*0.01, str(round(spot_price, 2)),color=color4, verticalalignment='bottom',ha=ha_spot)
        plt.title("Return p.a. per strike price")
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
        # Add labels, title, and legend
        plt.grid('on', ls="--")
        plt.xlabel('Strike price (K)')
        plt.legend(loc='upper left')
        plt.show()
        st.pyplot()

        strike_price_text = (
            f"Based on a target underlying price of **{target_price:.2f}** at expiration and profits calculated using the Black-Scholes formula, setting a strike price of **{optimal_strike_price:.0f}** would yield the highest possible return of **{max_return_long_call_at_expiry:.2%}** p.a. if the option is held until expiration."
            )
        st.write(strike_price_text)

        
        headline1 = "Distribution fitting"
        st.write(f"**{headline1}**")

        #construct empirical distribution function
        edf = np.arange(1, len(return_data)+1)/len(return_data)
        # fit Johnson SU distribution to return data
        su_loc_2, su_scale_2, su_loc_1, su_scale_1  = stats.johnsonsu.fit(return_data)
        cdf = stats.johnsonsu.cdf(return_data, a=su_loc_2, b=su_scale_2, loc=su_loc_1, scale=su_scale_1)
        pdf = stats.johnsonsu.pdf(return_data, a=su_loc_2, b=su_scale_2, loc=su_loc_1, scale=su_scale_1)

        # fit Normal distribution to return data  
        cdf_norm = stats.norm.cdf(return_data, loc=return_data.mean(), scale=return_data.std())
        pdf_nom = stats.norm.pdf(return_data, loc=return_data.mean(), scale=return_data.std())

        #evaluate the goodness-of-fit using Kolmogorov-Smirnov test
        supremum = max(abs(edf - cdf))
        p_value = np.exp(-supremum**2*len(return_data))
        supremum_norm = max(abs(edf - cdf_norm))
        p_value_norm = np.exp(-supremum_norm**2*len(return_data))

        #Plot
        # Create a figure with two subplots side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Define colors
        color1 = 'cornflowerblue'
        color2 = 'darkmagenta'
        color3 = 'royalblue'
        
        # Plot1: Histogram and PDFs
        ax1.hist(return_data, bins=200, density=True, alpha=0.7, color=color1, label="Histogram of daily log-returns")
        ax1.plot(return_data, pdf, color=color2, label="PDF fitted Johnson SU Dist.")
        ax1.plot(return_data, pdf_nom, color=color3, label="PDF fitted Normal Dist.")
        
        # Adding labels, title, and legend to Plot1
        ax1.set_xlabel('Daily log-returns')
        ax1.set_ylabel('Frequency')
        ax1.grid('on', ls="--")
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
        
        # Plot2: Distribution functions
        ax2.plot(return_data, edf, color=color1, label="Empirical distribution function")
        ax2.plot(return_data, cdf, color=color2, label="CDF fitted Johnson SU Dist.")
        ax2.plot(return_data, cdf_norm, color=color3, label="CDF fitted Normal Dist.")
        
        # Adding labels, title, and legend to Plot2
        ax2.set_xlabel('Daily log-returns')
        ax2.set_ylabel('Cumulative probability')
        ax2.grid('on', ls="--")
        ax2.legend(loc='upper left')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
        
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
        st.pyplot()

        # Calculate first four moments
        mean_emp = pd.Series(return_data).mean()
        std_emp = pd.Series(return_data).std()
        skewness_emp = pd.Series(return_data).skew()
        kurtosis_emp = pd.Series(return_data).kurtosis()

        st.markdown(
                    f"""
                    Goodness-of-fit evaluation using Kolmogorov-Smirnov test:
                    - P-value fitted Johnson SU distribution {p_value:.2%} 
                    - P-value fitted Normal distribution {p_value_norm:.2%} 
                    """
                    )
        
        st.markdown(
                    f"""
                    Distribution moments of daily log-returns:
                    - Mean {mean_emp:.2%} 
                    - Standard deviation {std_emp:.2%} 
                    - Skewness {skewness_emp:.2f}
                    - Kurtosis {kurtosis_emp:.2f}
                    """
                    )
        
        headline2 = "Price simulation based on fitted Johnson SU distribution"
        st.write(f"**{headline2}**")
        sim_runs_option = st.number_input("Choose a number of simulation runs", value=10000)
        
        if st.button("Run simulation"):
            returns_option_sim = stats.johnsonsu.ppf(np.random.uniform(cdf.min(), cdf.max(), size=(number_trading_days, sim_runs_option)), a=su_loc_2, b=su_scale_2, loc=su_loc_1, scale=su_scale_1)
            
            random_growth_factors = np.exp(returns_option_sim)
            growth_t0 = np.ones((1, random_growth_factors.shape[1])) * spot_price
            growth_paths = np.vstack((growth_t0, random_growth_factors))
            cumulative_growth_paths = np.cumprod(growth_paths, axis=0)
            
            simulation_df = pd.DataFrame(cumulative_growth_paths)
            simulation_df.set_index(pd.Index(trading_days), inplace=True)

            prices_at_expiry = simulation_df.iloc[-1,:]
            chance_above_be = (prices_at_expiry >= breakeven_at_expiration).mean()
            chance_below_be = 1 - chance_above_be
            path_max_val_at_expiry = simulation_df.iloc[: ,prices_at_expiry.idxmax()]
            path_min_val_at_expiry = simulation_df.iloc[: ,prices_at_expiry.idxmin()]

            #Plot
            color1 = 'cornflowerblue'
            color3 = 'darkmagenta'
            color2 = "deepskyblue"
            color4 = "mediumslateblue"
            
            plt.figure(figsize=(10, 6))
            plt.gca().set_xlim(left=simulation_df.index.min(), right=simulation_df.index.max())
            
            
            plt.plot(simulation_df.iloc[:, :300], color='lightgrey', alpha=0.8)
            #plt.plot(path_max_val_at_expiry, color='lightgrey', alpha=0.8)
            #plt.plot(path_min_val_at_expiry, color='lightgrey', alpha=0.8)
            
            # Calculate the number of days to add
            num_days = (simulation_df.index.max() - simulation_df.index.min()).days
            days_to_add1 = num_days / 120
            
            # Display strike price
            plt.axhline(strike_price, color=color1, linewidth=1.25, label="Strike price")
            plt.text(simulation_df.index[-1] + pd.Timedelta(days=days_to_add1), strike_price, str(round(strike_price, 2)),color=color1, verticalalignment='center')
            
            # Display spot price
            plt.axhline(spot_price, color=color2, linewidth=1.25, label="Spot price")
            plt.text(simulation_df.index[-1] + pd.Timedelta(days=days_to_add1), spot_price, str(round(spot_price, 2)),color=color2, verticalalignment='center')
            
            # Display breakeven price @ expiry
            plt.axhline(breakeven_at_expiration, color=color3, linewidth=1.25, label="Breakeven price @ expiration")
            plt.text(simulation_df.index[-1] + pd.Timedelta(days=days_to_add1), breakeven_at_expiration, str(round(breakeven_at_expiration, 2)),color=color3, verticalalignment='center')
            
            # Display above/below breakeven probabilities
            plt.text(simulation_df.index[-1] + pd.Timedelta(days=days_to_add1), (breakeven_at_expiration+prices_at_expiry.max())/2, str(round(chance_above_be*100, 2))+"%",color=color3, verticalalignment='center')
            plt.text(simulation_df.index[-1] + pd.Timedelta(days=days_to_add1), (breakeven_at_expiration+prices_at_expiry.min())/2, str(round(chance_below_be*100, 2))+"%",color=color3, verticalalignment='center')
            
            mean_sim_prices = simulation_df.mean(axis=1)
            plt.plot(mean_sim_prices, color=color4, linewidth=1.25, label="Mean simulated price")
            plt.text(simulation_df.index[-1] + pd.Timedelta(days=days_to_add1), mean_sim_prices.iloc[-1], str(round(mean_sim_prices.iloc[-1], 2)),color=color4, verticalalignment='center')
            
            
            # Add labels, title, and legend
            plt.grid('on', ls="--")
            plt.title("Price simulation of underlying based on Johnson SU distribution")
            
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format dates to show month and year
            plt.gca().xaxis.set_major_locator(MaxNLocator())
            plt.minorticks_off()
            
            plt.legend()
            plt.show()
            st.pyplot()

            price_sim_text = (
                f"Simulated returns are generated using a Johnson SU distribution fitted to historical daily log-return data of the underlying asset. Based on the simulation, there is a **{chance_above_be:.2%}** chance that the underlying will close above the breakeven price at expiration and a **{chance_below_be:.2%}** chance that it will close below. The mean closing price at expiration is **{mean_sim_prices.iloc[-1]:.2f}**."
                )
            st.write(price_sim_text)

        headline3 = "Option value and yield curve"
        st.write(f"**{headline3}**")
        
        #Plot chart
        S0_prices = np.arange(0.0001, 2*strike_price)
        S0_prices_till_spot = np.arange(0.0001, spot_price)
        black_scholes_value = black_scholes_call_value(S0=spot_price, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol) * subscription_ratio
        black_scholes_values = black_scholes_call_value(S0=S0_prices, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol) * subscription_ratio
        black_scholes_values_till_spot = black_scholes_call_value(S0=S0_prices_till_spot, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol) * subscription_ratio
        call_payoffs = call_payoff(S0=S0_prices, K=strike_price) * subscription_ratio
        call_payoffs_till_spot = call_payoff(S0=S0_prices_till_spot, K=strike_price) * subscription_ratio
        color1 = 'cornflowerblue'
        color2 = 'darkmagenta'
        plt.figure(figsize=(7, 4))
        plt.gca().set_xlim(left=0, right=2*strike_price)
        plt.plot(S0_prices, black_scholes_values, color=color1, label='Option value')
        plt.plot(S0_prices, call_payoffs, color='slategrey', label='Payoff')
        plt.fill_between(S0_prices_till_spot, black_scholes_values_till_spot, call_payoffs_till_spot,
                                color='limegreen', alpha=0.17, label="Time value", edgecolor='green', hatch='\/\/\/\/\/\/')
        plt.fill_between(S0_prices_till_spot, call_payoffs_till_spot, 0,
                                color='limegreen', alpha=0.17, label="Internal value")
        # Add a black horizontal line at y=0
        plt.axhline(0, color='black', linewidth=0.5)
        # Plot a dark gray point at (spot_price, black_scholes_value_at_spot)
        plt.scatter(spot_price, black_scholes_value, color=color2, zorder=5, label="Current option value")
        # Add text annotation for the Black-Scholes value
        plt.text(spot_price, black_scholes_value, f'{black_scholes_value:.2f} ', color=color2, fontsize=9, ha='right', va='bottom')
        plt.title("Black-Scholes option value")
        # Add labels, title, and legend
        plt.grid('on', ls="--")
        plt.xlabel('Price underlying (S)')
        plt.legend()
        plt.show()
        st.pyplot()

        color1 = 'cornflowerblue'
        color2 = 'darkmagenta'
        plot_maturities = np.arange(0, 40, 0.01)
        plot_yields = curve_fit(plot_maturities)
        plt.figure(figsize=(7, 4))
        plt.gca().set_xlim(left=0, right=10)
        plt.plot(plot_maturities, plot_yields, color=color1)
        # Plot a dark gray point at (spot_price, black_scholes_value_at_spot)
        plt.scatter(time_in_years, rf, color=color2, zorder=5, label="Risk-free rate")
        # Add text annotation for the Black-Scholes value
        plt.text(time_in_years, rf+0.002, f'{rf*100:.2f}'+"%", color=color2, fontsize=9, ha='center', va='bottom')
        plt.ylim(0, 0.10) 
        plt.xlim(0) 
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:,.0%}'.format))
        plt.title("Yield curve (US Treasuries)")
        # Add labels, title, and legend
        plt.grid('on', ls="--")
        plt.xlabel('Years to maturity')
        plt.legend()
        plt.show()
        st.pyplot()
        
        txt = "Greeks"
        st.write(f"**{txt}**")
        
        #Delta
        call_deltas = call_delta(S0=S0_prices, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol)
        plt.figure(figsize=(7, 4))
        plt.gca().set_xlim(left=0, right=2*strike_price)
        plt.plot(S0_prices, call_deltas, color=color1)
        # Add a black horizontal line at y=0
        plt.axhline(0, color='black', linewidth=0.5)
        # Plot a dark gray point at (spot_price, black_scholes_value_at_spot)
        plt.scatter(spot_price, call_delta_spot, color=color2, zorder=5, label="Current Delta")
        # Add text annotation for the Black-Scholes value
        plt.text(spot_price, call_delta_spot, f'{call_delta_spot:.2f} ', color=color2, fontsize=9, ha='right', va='bottom')
        plt.title("Delta")
        # Add labels, title, and legend
        plt.grid('on', ls="--")
        plt.xlabel('Price underlying (S)')
        plt.legend()
        plt.show()
        st.pyplot()

        #Gamma
        call_gammas = gamma(S0=S0_prices, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol)
        call_gamma_spot = gamma(S0=spot_price, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol)
        plt.figure(figsize=(7, 4))
        plt.gca().set_xlim(left=0, right=2*strike_price)
        plt.plot(S0_prices, call_gammas, color=color1)
        # Add a black horizontal line at y=0
        plt.axhline(0, color='black', linewidth=0.5)
        # Plot a dark gray point at (spot_price, black_scholes_value_at_spot)
        plt.scatter(spot_price, call_gamma_spot, color=color2, zorder=5, label="Current Gamma")
        # Add text annotation for the Black-Scholes value
        plt.text(spot_price, call_gamma_spot, f'{call_gamma_spot:.4f} ', color=color2, fontsize=9, ha='right', va='bottom')
        plt.title("Gamma")
        # Add labels, title, and legend
        plt.grid('on', ls="--")
        plt.xlabel('Price underlying (S)')
        plt.legend()
        plt.show()
        st.pyplot()

        #Vega
        call_vegas = vega(S0=S0_prices, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol)
        call_vega_spot = vega(S0=spot_price, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol)
        plt.figure(figsize=(7, 4))
        plt.gca().set_xlim(left=0, right=2*strike_price)
        plt.plot(S0_prices, call_vegas, color=color1)
        # Add a black horizontal line at y=0
        plt.axhline(0, color='black', linewidth=0.5)
        # Plot a dark gray point at (spot_price, black_scholes_value_at_spot)
        plt.scatter(spot_price, call_vega_spot, color=color2, zorder=5, label="Current Vega")
        # Add text annotation for the Black-Scholes value
        plt.text(spot_price, call_vega_spot, f'{call_vega_spot:.4f} ', color=color2, fontsize=9, ha='right', va='bottom')
        plt.title("Vega")
        # Add labels, title, and legend
        plt.grid('on', ls="--")
        plt.xlabel('Price underlying (S)')
        plt.legend()
        plt.show()
        st.pyplot()

        #omega
        S0_prices_omega = np.arange(0.5*strike_price, 1.5*strike_price)
        call_omegas = call_omega(S0=S0_prices_omega, K=strike_price, rf=rf, T=time_in_years, vol=selected_vol)
        plt.figure(figsize=(7, 4))
        plt.gca().set_xlim(left=0.5*strike_price, right=1.5*strike_price)
        plt.plot(S0_prices_omega, call_omegas, color=color1)
        # Add a black horizontal line at y=0
        plt.axhline(0, color='black', linewidth=0.5)
        # Plot a dark gray point at (spot_price, black_scholes_value_at_spot)
        plt.scatter(spot_price, call_omega_spot, color=color2, zorder=5, label="Current Omega")
        # Add text annotation for the Black-Scholes value
        plt.text(spot_price, call_omega_spot, f'  {call_omega_spot:.2f}', color=color2, fontsize=9, ha='left', va='bottom')
        plt.title("Omega (Leverage)")
        # Add labels, title, and legend
        plt.grid('on', ls="--")
        plt.xlabel('Price underlying (S)')
        plt.legend()
        plt.show()
        st.pyplot()

      

    
    if option == "Data":
        st.write("Monthly adjusted closing prices:")
        st.dataframe(montly_adjusted_closing_prices)
        st.write("Monthly log returns:")
        st.dataframe(monthly_log_returns)

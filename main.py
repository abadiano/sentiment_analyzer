import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import re

# Part (1)

# Load the tweets dataset
tweets_df = pd.read_csv('C:/Users/temka/tweets_TRUMP_2016_2019.csv', encoding='utf-16', delimiter=',')

# Function to clean the tweet text
def clean_text(txt):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", txt).split())

# Function to analyze the sentiment of a text
def analyze_sentiment(txt):
    clean_txt = clean_text(txt)
    analysis = TextBlob(clean_txt)
    return analysis.sentiment.polarity

# Sentiment with the cleaned text
tweets_df['cleaned_sentiment'] = tweets_df['text'].apply(analyze_sentiment)

# Display the first few rows
tweets_df[['text', 'cleaned_sentiment']].head()

# Part (2)

# Calculate the percentages of positive, negative, and neutral tweets
positive_tweets = (tweets_df['cleaned_sentiment'] > 0).sum()
negative_tweets = (tweets_df['cleaned_sentiment'] < 0).sum()
neutral_tweets = (tweets_df['cleaned_sentiment'] == 0).sum()
total_tweets = len(tweets_df)

# Calculate percentages
positive_percentage = (positive_tweets / total_tweets) * 100
negative_percentage = (negative_tweets / total_tweets) * 100
neutral_percentage = (neutral_tweets / total_tweets) * 100

# Describe overall sentiment
average_sentiment = tweets_df['cleaned_sentiment'].mean()
overall_sentiment = "positive" if average_sentiment > 0 else "negative" if average_sentiment < 0 else "neutral"

positive_percentage, negative_percentage, neutral_percentage, overall_sentiment

# Part (3)

tweets_df['date'] = pd.to_datetime(tweets_df['created_at']).dt.date

# Group by date, calculate the average sentiment, drop nulls
daily_sentiment = tweets_df.groupby('date')['cleaned_sentiment'].mean().reset_index()
daily_sentiment.dropna(inplace=True)

daily_sentiment.head()

# Part (4)

# Load the VIX futures data
vix_futures_df = pd.read_excel('C:/Users/temka/VIX_futures_2016_2019.xlsx')
vix_columns = vix_futures_df.columns.tolist()

# Convert columns to datetime type
daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
vix_futures_df['Code'] = pd.to_datetime(vix_futures_df['Code'])

# Merge the dataframes
merged_df = pd.merge(daily_sentiment, vix_futures_df, left_on='date', right_on='Code', how='inner')

# Plotting
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot daily average sentiment
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily Average Sentiment', color=color)
ax1.plot(merged_df['date'], merged_df['cleaned_sentiment'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Plot VIX futures value
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('VIX Futures Value', color=color)
ax2.plot(merged_df['date'], merged_df['Sett_price'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Title and show plot
plt.title('Daily Average Sentiment and VIX Futures Value Over Time')
fig.tight_layout()
plt.show()

# Part (5)

# Compute the rolling averages for the sentiment
merged_df['30_day_MA'] = merged_df['cleaned_sentiment'].rolling(window=30).mean()
merged_df['5_day_MA'] = merged_df['cleaned_sentiment'].rolling(window=5).mean()

# Generate signals based on the moving averages
merged_df['position'] = merged_df['5_day_MA'] < merged_df['30_day_MA']
merged_df['position'] = merged_df['position'].map({True: 'Long', False: 'Short'})

# Plotting
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot the 30-day and 5-day moving averages of sentiment
ax1.plot(merged_df['date'], merged_df['30_day_MA'], label='30 Day MA (Long-term)', color='orange', alpha=0.8)
ax1.plot(merged_df['date'], merged_df['5_day_MA'], label='5 Day MA (Short-term)', color='green', alpha=0.8)

# Plot VIX futures value
ax2 = ax1.twinx()
ax2.plot(merged_df['date'], merged_df['Sett_price'], label='VIX Futures Value', color='blue', alpha=0.5)

# Annotate the positions on the plot
for i in range(len(merged_df)):
    if merged_df['position'].iloc[i] == 'Long':
        ax1.annotate('L', (merged_df['date'].iloc[i], merged_df['5_day_MA'].iloc[i]), color='red', fontsize=8)
    else:
        ax1.annotate('S', (merged_df['date'].iloc[i], merged_df['5_day_MA'].iloc[i]), color='black', fontsize=8)

# Labels and legend
ax1.set_xlabel('Date')
ax1.set_ylabel('Sentiment Moving Averages', color='green')
ax2.set_ylabel('VIX Futures Value', color='blue')
fig.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95), bbox_transform=ax1.transAxes)
plt.title('Sentiment Moving Averages, VIX Futures Value, and Trading Positions')
fig.tight_layout()

# Show plot
plt.show()

# Part (6)

# Filter the data
mask = (merged_df['date'] >= pd.to_datetime('2016-01-01')) & (merged_df['date'] <= pd.to_datetime('2018-12-31'))
strategy_period_df = merged_df.loc[mask].copy()

# Calculate daily returns of VIX futures
strategy_period_df.loc[:, 'vix_daily_returns'] = strategy_period_df['Sett_price'].pct_change()

# Apply the strategy: go long (buy) when the signal is 'Long' and go short (sell) when the signal is 'Short'
# When going short, the return is the negative of the VIX return
strategy_period_df.loc[:, 'strategy_returns'] = np.where(strategy_period_df['position'] == 'Long',
                                                         strategy_period_df['vix_daily_returns'],
                                                         -strategy_period_df['vix_daily_returns'])

# Calculate cumulative returns by compounding the daily strategy returns
strategy_period_df.loc[:, 'cumulative_returns'] = (1 + strategy_period_df['strategy_returns']).cumprod()

# Calculate average return, standard deviation, and Sharpe ratio
average_return = strategy_period_df['strategy_returns'].mean() * 252
std_dev = strategy_period_df['strategy_returns'].std() * np.sqrt(252)
sharpe_ratio = average_return / std_dev

# Calculate the drawdown
strategy_period_df.loc[:, 'cumulative_max'] = strategy_period_df['cumulative_returns'].cummax()
strategy_period_df.loc[:, 'drawdown'] = strategy_period_df['cumulative_returns'] - strategy_period_df['cumulative_max']
strategy_period_df.loc[:, 'drawdown_percentage'] = strategy_period_df['drawdown'] / strategy_period_df['cumulative_max']

# Find the maximum drawdown
max_drawdown = strategy_period_df['drawdown_percentage'].min()

# Plot the evolution of $1,000 invested in the strategy
initial_investment = 1000
strategy_period_df.loc[:, 'investment_value'] = initial_investment * strategy_period_df['cumulative_returns']
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(strategy_period_df['date'], strategy_period_df['investment_value'], label='Investment Value', color='purple')
ax.set_xlabel('Date')
ax.set_ylabel('Value of $1,000 Investment')
ax.set_title('Evolution of $1,000 Investment in Strategy')
plt.legend()
plt.show()

# Results
average_return, std_dev, sharpe_ratio, max_drawdown


# Part (7) - Finding Perfect Combination

# If want to find results specifically for 2019
# merged_df = merged_df[(merged_df['date'] >= '2019-01-01') & (merged_df['date'] <= '2019-12-31')]

# Function to calculate the Sharpe ratio
def calculate_sharpe_ratio(returns):
    mean_return = returns.mean() * 252
    std_dev = returns.std() * np.sqrt(252)
    return mean_return / std_dev if std_dev != 0 else np.nan


merged_df['vix_daily_returns'] = merged_df['Sett_price'].pct_change()
sharpe_ratios = {}

# Combinations
long_terms = [40, 50, 60, 70, 90]
short_terms = [3, 5, 10, 15]

for long_term in long_terms:
    for short_term in short_terms:
        # Calculate the moving averages
        merged_df[f'{long_term}_day_MA'] = merged_df['cleaned_sentiment'].rolling(window=long_term).mean()
        merged_df[f'{short_term}_day_MA'] = merged_df['cleaned_sentiment'].rolling(window=short_term).mean()

        # Generate strategy returns
        strategy_returns = np.where(merged_df[f'{short_term}_day_MA'] < merged_df[f'{long_term}_day_MA'],
                                    merged_df['vix_daily_returns'],
                                    -merged_df['vix_daily_returns'])

        # Calculate Sharpe ratio
        sharpe_ratio = calculate_sharpe_ratio(pd.Series(strategy_returns[~np.isnan(strategy_returns)]))
        sharpe_ratios[(long_term, short_term)] = sharpe_ratio

# Find the combination with the best Sharpe ratio
best_combination = max(sharpe_ratios, key=sharpe_ratios.get)
best_sharpe_ratio = sharpe_ratios[best_combination]

best_combination, best_sharpe_ratio

# Part (7) - Investing $1000

# Filter the data
test_period_df = merged_df[(merged_df['date'] >= pd.to_datetime('2019-01-01')) &
                           (merged_df['date'] <= pd.to_datetime('2019-12-31'))].copy()

# Calculate the moving averages using the best combination found
test_period_df['60_day_MA'] = test_period_df['cleaned_sentiment'].rolling(window=60).mean()
test_period_df['15_day_MA'] = test_period_df['cleaned_sentiment'].rolling(window=15).mean()

# Generate strategy signals and returns for the test period
test_period_df['signal'] = test_period_df['15_day_MA'] < test_period_df['60_day_MA']
test_period_df['strategy_returns'] = np.where(test_period_df['signal'],
                                              test_period_df['vix_daily_returns'],
                                              -test_period_df['vix_daily_returns'])

# Calculate cumulative returns for the strategy
test_period_df['cumulative_strategy_returns'] = (1 + test_period_df['strategy_returns']).cumprod()

# Calculate metrics for the strategy
test_average_return = test_period_df['strategy_returns'].mean() * 252
test_std_dev = test_period_df['strategy_returns'].std() * np.sqrt(252)
test_sharpe_ratio = test_average_return / test_std_dev

# Calculate the drawdown for the strategy
test_period_df['cumulative_max'] = test_period_df['cumulative_strategy_returns'].cummax()
test_period_df['drawdown'] = test_period_df['cumulative_strategy_returns'] - test_period_df['cumulative_max']
test_period_df['drawdown_percentage'] = test_period_df['drawdown'] / test_period_df['cumulative_max']
test_max_drawdown = test_period_df['drawdown_percentage'].min()

# Plot the evolution of $1,000 invested in the strategy
test_initial_investment = 1000
test_period_df['investment_value'] = test_initial_investment * test_period_df['cumulative_strategy_returns']

# Plot for strategy
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(test_period_df['date'], test_period_df['investment_value'], label='Strategy Investment Value', color='purple')

# Comparing with passive long and short positions
test_period_df['passive_long_value'] = test_initial_investment * (1 + test_period_df['vix_daily_returns']).cumprod()
test_period_df['passive_short_value'] = test_initial_investment / (1 + test_period_df['vix_daily_returns']).cumprod()

ax.plot(test_period_df['date'], test_period_df['passive_long_value'], label='Passive Long Investment Value', color='green')
ax.plot(test_period_df['date'], test_period_df['passive_short_value'], label='Passive Short Investment Value', color='red')

ax.set_xlabel('Date')
ax.set_ylabel('Value of $1,000 Investment')
ax.set_title('Investment Value Comparison')
plt.legend()
plt.show()

# Results for the test period
test_average_return, test_std_dev, test_sharpe_ratio, test_max_drawdown
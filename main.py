from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
prefix_url = "https://finviz.com/quote.ashx?t="
tickers = ["AMZN", "FB", "AAPL", "MSFT"]
news_tables = {} 

for ticker in tickers: 
    url = prefix_url + ticker
    #request for html data
    req = Request(url = url, headers = {"user-agent" : ""})
    response = urlopen(req)
    html = BeautifulSoup(response, features = "html.parser")
    #print(html) prints finviz html

    #in finviz, the article section lies in a table with id = news-table
    #listed in a table
    news_table = html.find(id="news-table") 

    #grabs the news table per ticker
    news_tables[ticker] = news_table

parsed_data = []

for ticker, news_table in news_tables.items(): #for each ticker, find the html for its news table
    for row in news_table.findAll("tr"): #for each row of the html table
        title = row.a.get_text()
        date_data = row.td.text.split(" ")

        if len(date_data) == 1: #its a date
            time = date_data[0] 
        else: #its a timestamp 
            date = date_data[0]
            time = date_data[1]
        parsed_data.append([ticker, date, time, title])

#create a pandas dataframe
df = pd.DataFrame(parsed_data, columns=["ticker", "date", "time", "title"])
vader = SentimentIntensityAnalyzer() #instantiate object
f = lambda title: vader.polarity_scores(title)["compound"] #returns just the compound score

df["compound"] = df["title"].apply(f) 
df["date"] = pd.to_datetime(df.date).dt.date #takes our date column and converts it from string to date-time
mean_df = df.groupby(["ticker", "date"]).mean() #computes the average of the compound score
mean_df = mean_df.unstack()
mean_df = mean_df.xs("compound", axis = "columns").transpose()
mean_df.plot(kind = "bar")
plt.show()


# data_collection.py

# Import the required libraries and modules
import requests
import BeautifulSoup
import Scrapy
import pandas as pd
import numpy as np
import sklearn
import sagemaker
from sagemaker import Session, DataWrangler
from sagemaker.s3 import S3Uploader

# Define the data sources and the types
data_sources = {
    "numerical": [
        "https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC", # S&P 500 historical data
        "https://finance.yahoo.com/quote/%5EIXIC/history?p=%5EIXIC", # NASDAQ historical data
        "https://finance.yahoo.com/quote/%5EDJI/history?p=%5EDJI" # Dow Jones historical data
    ],
    "textual": [
        "https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey=YOUR_API_KEY", # Business news headlines from News API
        "https://api.twitter.com/1.1/search/tweets.json?q=%23stockmarket&result_type=mixed&count=100&lang=en", # Tweets with #stockmarket hashtag from Twitter API
        "https://www.reddit.com/r/StockMarket/.rss" # Posts from r/StockMarket subreddit from RSS feed
    ],
    "visual": [
        "https://www.tradingview.com/chart/?symbol=SPX", # S&P 500 chart from TradingView
        "https://www.tradingview.com/chart/?symbol=NASDAQ:NDX", # NASDAQ chart from TradingView
        "https://www.tradingview.com/chart/?symbol=DJI" # Dow Jones chart from TradingView
    ],
    "audio": [
        "https://www.npr.org/rss/podcast.php?id=510289", # The Indicator from Planet Money podcast from NPR
        "https://feeds.megaphone.fm/WSJ8577185360", # The Journal podcast from The Wall Street Journal
        "https://feeds.bloomberg.fm/BLM4689309739" # Bloomberg Surveillance podcast from Bloomberg
    ]
}

# Define the data folder and the S3 bucket
data_folder = "project/data/"
s3_bucket = "s3://your-bucket-name/"

# Define the Sagemaker session and the DataWrangler flow
session = Session()
flow = DataWrangler(flow_source="project/data_wrangler_flow.flow", role="your-role-arn")

# Define a function to scrape the numerical data from the websites
def scrape_numerical_data(url):
    # Send a GET request to the url and get the response
    response = requests.get(url)
    # Parse the response content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")
    # Find the table element that contains the numerical data
    table = soup.find("table", {"data-test": "historical-prices"})
    # Convert the table element into a pandas dataframe
    df = pd.read_html(str(table))[0]
    # Rename the columns and drop the last row
    df.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df.drop(df.index[-1])
    # Return the dataframe
    return df

# Define a function to access the textual data from the APIs
def access_textual_data(url):
    # Send a GET request to the url and get the response
    response = requests.get(url)
    # Convert the response content into a JSON object
    data = response.json()
    # Extract the textual data from the JSON object
    if "articles" in data: # News API
        df = pd.DataFrame(data["articles"])
        df = df[["title", "description", "url", "publishedAt", "source"]]
    elif "statuses" in data: # Twitter API
        df = pd.DataFrame(data["statuses"])
        df = df[["text", "created_at", "user", "retweet_count", "favorite_count"]]
    elif "feed" in data: # RSS feed
        df = pd.DataFrame(data["feed"]["entries"])
        df = df[["title", "summary", "link", "published", "author"]]
    # Return the dataframe
    return df

# Define a function to scrape the visual data from the websites
def scrape_visual_data(url):
    # Send a GET request to the url and get the response
    response = requests.get(url)
    # Parse the response content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")
    # Find the image element that contains the visual data
    image = soup.find("img", {"id": "logo"})
    # Extract the image source attribute
    src = image["src"]
    # Return the image source
    return src

# Define a function to access the audio data from the RSS feeds
def access_audio_data(url):
    # Send a GET request to the url and get the response
    response = requests.get(url)
    # Convert the response content into a JSON object
    data = response.json()
    # Extract the audio data from the JSON object
    df = pd.DataFrame(data["feed"]["entries"])
    df = df[["title", "summary", "link", "published", "enclosures"]]
    # Return the dataframe
    return df

# Loop through the data sources and the types
for data_type, urls in data_sources.items():
    # Create an empty list to store the dataframes
    dfs = []
    # Loop through the urls for each data type
    for url in urls:
        # Call the appropriate function to get the data
        if data_type == "numerical":
            df = scrape_numerical_data(url)
        elif data_type == "textual":
            df = access_textual_data(url)
        elif data_type == "visual":
            df = scrape_visual_data(url)
        elif data_type == "audio":
            df = access_audio_data(url)
        # Append the dataframe to the list
        dfs.append(df)
    # Concatenate the dataframes in the list
    df = pd.concat(dfs, ignore_index=True)
    # Save the dataframe as a CSV file in the data folder
    file_name = data_folder + data_type + "/" + data_type + ".csv"
    df.to_csv(file_name, index=False)
    # Upload the CSV file to the S3 bucket
    s3_file_name = s3_bucket + file_name
    S3Uploader.upload(file_name, s3_file_name)
    # Run the DataWrangler flow on the CSV file
    flow.run(inputs=s3_file_name)

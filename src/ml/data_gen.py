from src.agents.ricardo_garcia import generate_ricardo_garcia_signals
from dotenv import load_dotenv
import csv

load_dotenv()

def generate_ricardo_garcia_data(tickers, start_date, end_date):
    ricardo_garcia_analysis = generate_ricardo_garcia_signals(tickers, start_date, end_date,"gpt-4o","OpenAI")
    return ricardo_garcia_analysis

# save the data to a csv file
def save_ricardo_garcia_data(tickers, ricardo_garcia_analysis, filename):
    data = {}
    for ticker in tickers:
        if ticker not in ricardo_garcia_analysis:
            print(f"Warning: No data found for {ticker}")
            continue
            
        if "ml_data" not in ricardo_garcia_analysis[ticker]:
            print(f"Warning: No ml_data found for {ticker}")
            continue
            
        row = {}
        ml_data = ricardo_garcia_analysis[ticker]["ml_data"]
        for key, value in ml_data.items():
            row[key] = value
        data[ticker] = row

    if not data:
        print("Error: No valid data to save")
        return

    # Get the header from the keys of the first dictionary
    first_ticker = next(iter(data))
    header = data[first_ticker].keys()

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for ticker, row in data.items():
            writer.writerow(row)


if __name__ == "__main__":
    tickers = ["RDDT", "MSFT", "PLTR", "AMZN", "TSLA"]
    start_date = "2025-05-19"
    end_date = "2025-05-20"
    ricardo_garcia_analysis = generate_ricardo_garcia_data(tickers, start_date, end_date)
    save_ricardo_garcia_data(tickers,ricardo_garcia_analysis, "ricardo_garcia_analysis.csv")


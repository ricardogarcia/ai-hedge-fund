import sys
import os
# Add the src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ricardo_garcia import generate_ricardo_garcia_signals_with_ml
from dotenv import load_dotenv
import csv
from datetime import datetime, timedelta

load_dotenv()

def generate_ricardo_garcia_data(tickers, start_date, end_date):
    print(f"Generating data for {tickers} from {start_date} to {end_date}")
    
    # Parse the date strings
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Dictionary to store all results
    all_results = {}
    
    # Iterate through each day in the range
    current_date = start_dt
    while current_date <= end_dt:
        #start_date_str should be 3 months before current_date
        start_date_str = (current_date - timedelta(days=180)).strftime("%Y-%m-%d")
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Processing date: {date_str}")
        
        try:
            # Call generate_ricardo_garcia_signals for this single day
            ricardo_garcia_analysis, ml_data = generate_ricardo_garcia_signals_with_ml(
                tickers, start_date_str, date_str
            )
            
            # Add each ticker's result with unique date-ticker key
            for ticker, analysis in ricardo_garcia_analysis.items():
                unique_key = f"{ticker}_{date_str}"
                
                # Add date and ticker information to ml_data
                ml_data_with_info = ml_data[ticker]
                ml_data_with_info["date"] = date_str
                ml_data_with_info["ticker"] = ticker
                
                # Add date information to the analysis
                analysis_with_date = analysis.copy()
                analysis_with_date["date"] = date_str
                analysis_with_date["ticker"] = ticker
                analysis_with_date["ml_data"] = ml_data_with_info
                
                all_results[unique_key] = analysis_with_date
                
        except Exception as e:
            print(f"Error processing date {date_str}: {str(e)}")
            # Move to next day
            current_date += timedelta(days=1)
            continue
        
        # Move to next day
        current_date += timedelta(days=180)
        
        
    
    return all_results

# save the data to a csv file
def save_ricardo_garcia_data(tickers, ricardo_garcia_analysis, filename):
    data = {}
    
    # Iterate through all unique date-ticker combinations
    for unique_key, analysis in ricardo_garcia_analysis.items():
        if "ml_data" not in analysis:
            print(f"Warning: No ml_data found for {unique_key}")
            continue
            
        row = {}
        ml_data = analysis["ml_data"]
        for key, value in ml_data.items():
            row[key] = value
        data[unique_key] = row

    if not data:
        print("Error: No valid data to save")
        return

    # Get the header from the keys of the first dictionary
    first_key = next(iter(data))
    header = data[first_key].keys()

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for unique_key, row in data.items():
            writer.writerow(row)
    
    print(f"Saved {len(data)} records to {filename}")


if __name__ == "__main__":
    tickers = ["RDDT", "MSFT", "PLTR", "AMZN", "TSLA","TSM","NVDA","META","AAPL","NFLX","CSCO","INTC","IBM","DAL","MRVL","LLY","ANET","NTNX","NVO","AMD","PFE"]
    start_date = "2008-01-01"
    end_date = "2022-06-15"  # 2-day range to test the functionality
    ricardo_garcia_analysis = generate_ricardo_garcia_data(tickers, start_date, end_date)
    save_ricardo_garcia_data(tickers, ricardo_garcia_analysis, "ricardo_garcia_analysis.csv")


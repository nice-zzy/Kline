#!/usr/bin/env python3
"""
Fetch real Dow 30 data using Alpha Vantage API
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class AlphaVantageFetcher:
    """Fetch real data from Alpha Vantage API"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.dow30_symbols = [
            "AAPL", "MSFT", "UNH", "GS", "HD", "CAT", "AMGN", "CRM", "MCD", "V",
            "BA", "ABBV", "CVX", "AXP", "WMT", "JNJ", "PG", "JPM", "IBM", "NKE",
            "TRV", "VZ", "KO", "DOW", "DIS", "CSCO", "INTC", "MRK", "HON", "WBA"
        ]
        
        # Rate limiting: Alpha Vantage free tier allows 5 calls per minute
        self.call_delay = 12  # 12 seconds between calls to be safe
    
    def fetch_symbol_data(self, symbol, outputsize="full"):
        """Fetch data for a single symbol"""
        print(f"Fetching {symbol}...")
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': outputsize,  # 'compact' for last 100 days, 'full' for all
            'datatype': 'json'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                print(f"  API Error for {symbol}: {data['Error Message']}")
                return pd.DataFrame()
            
            if 'Note' in data:
                print(f"  API Note for {symbol}: {data['Note']}")
                return pd.DataFrame()
            
            if 'Information' in data:
                print(f"  API Info for {symbol}: {data['Information']}")
                return pd.DataFrame()
            
            # Extract time series data
            if 'Time Series (Daily)' not in data:
                print(f"  No time series data found for {symbol}")
                return pd.DataFrame()
            
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'timestamp': pd.to_datetime(date_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"  Successfully fetched {len(df)} records for {symbol}")
            print(f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"  Request error for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_all_symbols(self, outputsize="full", max_symbols=None):
        """Fetch data for all Dow 30 symbols with rate limiting"""
        results = {}
        
        symbols_to_fetch = self.dow30_symbols
        if max_symbols:
            symbols_to_fetch = symbols_to_fetch[:max_symbols]
        
        print(f"Fetching data for {len(symbols_to_fetch)} symbols...")
        print(f"Rate limiting: {self.call_delay} seconds between calls")
        print(f"Estimated time: {len(symbols_to_fetch) * self.call_delay / 60:.1f} minutes")
        
        for i, symbol in enumerate(symbols_to_fetch):
            print(f"\nProgress: {i+1}/{len(symbols_to_fetch)}")
            
            # Fetch data
            data = self.fetch_symbol_data(symbol, outputsize)
            
            if not data.empty:
                results[symbol] = data
            else:
                print(f"  Failed to fetch {symbol}")
            
            # Rate limiting delay (except for last symbol)
            if i < len(symbols_to_fetch) - 1:
                print(f"  Waiting {self.call_delay} seconds...")
                time.sleep(self.call_delay)
        
        return results
    
    def save_data(self, data_dict, output_dir="data", prefix="dow30_real"):
        """Save fetched data to CSV files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        saved_files = []
        for symbol, data in data_dict.items():
            if not data.empty:
                filename = f"{prefix}_{symbol}.csv"
                filepath = output_dir / filename
                data.to_csv(filepath, index=False)
                saved_files.append(str(filepath))
                print(f"Saved {symbol} to {filepath}")
        
        return saved_files
    
    def create_metadata(self, data_dict, output_dir="data"):
        """Create metadata file"""
        metadata = {
            'fetched_at': datetime.now().isoformat(),
            'source': 'Alpha Vantage API',
            'symbols': list(data_dict.keys()),
            'total_symbols': len(data_dict),
            'symbol_details': {}
        }
        
        for symbol, data in data_dict.items():
            if not data.empty:
                metadata['symbol_details'][symbol] = {
                    'records': len(data),
                    'start_date': data['timestamp'].min().isoformat(),
                    'end_date': data['timestamp'].max().isoformat(),
                    'price_range': {
                        'min': float(data['low'].min()),
                        'max': float(data['high'].max())
                    }
                }
        
        metadata_file = Path(output_dir) / "dow30_real_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_file}")
        return metadata_file

def main():
    """Main function"""
    print("=== Alpha Vantage Dow 30 Data Fetcher ===")
    
    # Get API key from user
    api_key = input("Enter your Alpha Vantage API key: ").strip()
    
    if not api_key:
        print("No API key provided. Exiting.")
        return
    
    # Initialize fetcher
    fetcher = AlphaVantageFetcher(api_key)
    
    # Ask user for options
    print("\nOptions:")
    print("1. Fetch all 30 Dow symbols (takes ~6 minutes)")
    print("2. Fetch first 5 symbols for testing (takes ~1 minute)")
    print("3. Fetch compact data (last 100 days only)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        max_symbols = None
        outputsize = "full"
    elif choice == "2":
        max_symbols = 5
        outputsize = "full"
    elif choice == "3":
        max_symbols = None
        outputsize = "compact"
    else:
        print("Invalid choice. Using option 2 (first 5 symbols)")
        max_symbols = 5
        outputsize = "full"
    
    # Fetch data
    print(f"\nStarting data fetch...")
    data_dict = fetcher.fetch_all_symbols(outputsize=outputsize, max_symbols=max_symbols)
    
    if data_dict:
        # Save data
        saved_files = fetcher.save_data(data_dict, "data", "dow30_real")
        
        # Create metadata
        fetcher.create_metadata(data_dict, "data")
        
        print(f"\n=== Success ===")
        print(f"Successfully fetched data for {len(data_dict)} symbols")
        print(f"Saved {len(saved_files)} files to data/ directory")
        
        # Show summary
        total_records = sum(len(data) for data in data_dict.values())
        print(f"Total records: {total_records}")
        
        # Show sample data
        if data_dict:
            sample_symbol = list(data_dict.keys())[0]
            sample_data = data_dict[sample_symbol]
            print(f"\nSample data from {sample_symbol}:")
            print(sample_data.head())
        
        print(f"\nNext steps:")
        print(f"1. Run: python scripts/process_data.py --mode retrieval")
        print(f"2. Or test with: python scripts/test_fetch_with_delay.py")
        
    else:
        print("No data was fetched. Please check your API key and try again.")

if __name__ == "__main__":
    main()

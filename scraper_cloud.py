# scraper_cloud.py - Cloud Scraper for Supabase
import pandas as pd
import numpy as np
from datetime import datetime
import time
from supabase import create_client
import os
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()

class CloudScraper:
    """Cloud-based scraper that saves to Supabase"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        self.timezone = pytz.timezone('Asia/Karachi')
    
    def scrape_psx_data(self):
        """Simulate scraping - replace with your actual scraping logic"""
        try:
            # This is a simulation - replace with your actual scraping code
            # For now, we'll create sample data
            
            print(f"[{datetime.now()}] Starting PSX scrape...")
            
            # Simulate scraping delay
            time.sleep(2)
            
            # Generate sample data
            symbols = ['OGDC', 'PPL', 'HBL', 'UBL', 'MCB', 'BAHL', 'EFERT', 'FCCL', 'LUCK', 'NESTLE']
            sectors = ['Energy', 'Energy', 'Banking', 'Banking', 'Banking', 'Banking', 'Chemicals', 'Cement', 'Cement', 'Food']
            
            data = []
            current_time = datetime.now(pytz.UTC)
            
            for i, symbol in enumerate(symbols):
                row = {
                    'symbol': symbol,
                    'sector': sectors[i % len(sectors)],
                    'listed_in': 'PSX',
                    'ldcp': np.random.uniform(100, 500),
                    'open': np.random.uniform(100, 500),
                    'high': np.random.uniform(110, 550),
                    'low': np.random.uniform(90, 450),
                    'current': np.random.uniform(100, 500),
                    'change': np.random.uniform(-10, 10),
                    'change_percent': np.random.uniform(-5, 5),
                    'volume': int(np.random.uniform(100000, 1000000)),
                    'scrape_time': current_time.isoformat(),
                    'dataset_name': f"psx_{current_time.strftime('%Y%m%d_%H%M')}.csv"
                }
                row['current'] = row['open'] + row['change']
                data.append(row)
            
            print(f"[{datetime.now()}] Scraped {len(data)} records")
            return data
            
        except Exception as e:
            print(f"[{datetime.now()}] Scraping error: {str(e)}")
            return None
    
    def save_to_supabase(self, data):
        """Save scraped data to Supabase"""
        try:
            if not data:
                print("No data to save")
                return False
            
            # Check for duplicates
            existing_data = self.supabase.table('stock_data')\
                .select('symbol', 'scrape_time')\
                .eq('scrape_time', data[0]['scrape_time'])\
                .execute()
            
            existing_symbols = {item['symbol'] for item in existing_data.data} if existing_data.data else set()
            
            # Filter out duplicates
            new_data = [row for row in data if row['symbol'] not in existing_symbols]
            
            if not new_data:
                print("All data already exists in database")
                return True
            
            # Insert new data
            response = self.supabase.table('stock_data').insert(new_data).execute()
            
            if hasattr(response, 'data') and response.data:
                print(f"Successfully saved {len(response.data)} records to Supabase")
                return True
            else:
                print(f"Failed to save data: {response}")
                return False
                
        except Exception as e:
            print(f"Error saving to Supabase: {str(e)}")
            return False
    
    def run_scrape(self):
        """Run one scraping cycle"""
        print(f"\n{'='*50}")
        print(f"Starting scrape at {datetime.now()}")
        
        # Scrape data
        data = self.scrape_psx_data()
        
        if data:
            # Save to Supabase
            success = self.save_to_supabase(data)
            if success:
                print(f"Scrape completed successfully at {datetime.now()}")
            else:
                print(f"Scrape failed at {datetime.now()}")
        else:
            print("No data scraped")
        
        print(f"{'='*50}\n")
        return success

def main():
    """Main function for manual testing"""
    scraper = CloudScraper()
    scraper.run_scrape()

if __name__ == "__main__":
    main()
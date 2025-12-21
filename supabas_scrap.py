from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from datetime import datetime
import csv
import sys
import io
import os
import json
import hashlib
from supabase import create_client, Client  # Added Supabase
from dotenv import load_dotenv

load_dotenv()

# Fix Unicode encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# # Supabase Configuration - Replace with your credentials
# SUPABASE_URL = "https://your-project.supabase.co"
# SUPABASE_KEY = "your-anon-key"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are missing")


supabase: Client = None

def init_supabase():
    """Initialize Supabase client if credentials are provided"""
    global supabase
    try:
        # Check if credentials are provided (not the placeholder values)
        if SUPABASE_URL and SUPABASE_KEY and not "your-project" in SUPABASE_URL:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("✓ Supabase client initialized successfully")
            return True
        else:
            print("⚠ Supabase credentials not configured. Data will only be processed locally.")
            return False
    except Exception as e:
        print(f"⚠ Supabase initialization error: {e}. Data will only be processed locally.")
        return False

def setup_driver():
    """Chrome driver setup - Optimized for speed"""
    print("Setting up Chrome driver...")
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    # Performance optimizations
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-plugins')
    chrome_options.add_argument('--disable-images')  # Speed up loading
    chrome_options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_settings.popups": 0,
    })
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    print("Chrome driver ready!")
    return driver

def extract_correct_psx_data(driver):
    """FIXED: Correct PSX data extraction with ALL columns including SECTOR and LISTED IN"""
    try:
        url = "https://dps.psx.com.pk/market-watch"
        print(f"Loading: {url}")
        
        driver.get(url)
        
        print("Waiting for data to load...")
        # Reduced wait time - use explicit waits instead
        wait = WebDriverWait(driver, 15)
        
        # Wait for table to be present
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        print("Table found, extracting data...")
        
        # Extract with COMPLETE column mapping
        extract_script = """
            let allStocks = [];
            
            let tables = document.querySelectorAll('table');
            let targetTable = null;
            
            // Find the main market watch table
            for (let table of tables) {
                let headers = table.querySelectorAll('th');
                if (headers.length >= 10) {  // PSX table has 11 columns
                    let headerText = Array.from(headers).map(h => h.innerText.trim()).join(' ');
                    if (headerText.includes('Symbol') || headerText.includes('Sector') || headerText.includes('LDCP')) {
                        targetTable = table;
                        break;
                    }
                }
            }
            
            if (!targetTable && tables.length > 0) {
                targetTable = tables[0];
            }
            
            if (!targetTable) return allStocks;
            
            let tbody = targetTable.querySelector('tbody');
            if (!tbody) return allStocks;
            
            let rows = tbody.querySelectorAll('tr');
            
            rows.forEach((row, rowIndex) => {
                let cells = row.querySelectorAll('td');
                
                // CORRECT PSX TABLE STRUCTURE - 11 COLUMNS:
                // 0: Symbol
                // 1: Sector
                // 2: Listed In
                // 3: LDCP (Last Day Close Price)
                // 4: Open
                // 5: High
                // 6: Low
                // 7: Current
                // 8: Change
                // 9: Change%
                // 10: Volume
                
                if (cells.length >= 9) {  // Some rows might have fewer cells
                    let symbol = cells[0]?.innerText.trim() || '';
                    
                    // Skip invalid rows
                    if (!symbol || symbol === '' || symbol.length > 20 || 
                        symbol.includes('Symbol') || symbol.includes('Last') ||
                        symbol === 'PSX' || symbol === 'KSE-100' ||
                        symbol.includes('Open') || symbol.includes('High') ||
                        symbol.includes('Low') || symbol.includes('Volume')) {
                        return;
                    }
                    
                    // COMPLETE COLUMN MAPPING - ALL 11 COLUMNS
                    let stockData = {
                        symbol: symbol,
                        sector: cells[1]?.innerText.trim() || '',           // Column 1 - SECTOR
                        listed_in: cells[2]?.innerText.trim() || '',        // Column 2 - LISTED IN
                        ldcp: cells[3]?.innerText.trim() || '0',           // Column 3 - LDCP
                        open: cells[4]?.innerText.trim() || '0',           // Column 4 - Open
                        high: cells[5]?.innerText.trim() || '0',           // Column 5 - High
                        low: cells[6]?.innerText.trim() || '0',            // Column 6 - Low
                        current: cells[7]?.innerText.trim() || '0',        // Column 7 - Current
                        change: cells[8]?.innerText.trim() || '0',         // Column 8 - Change
                        change_percent: cells[9]?.innerText.trim() || '0', // Column 9 - Change%
                        volume: cells[10]?.innerText.trim() || '0'         // Column 10 - Volume
                    };
                    
                    allStocks.push(stockData);
                }
            });
            
            return allStocks;
        """
        
        raw_data = driver.execute_script(extract_script)
        
        if not raw_data or len(raw_data) == 0:
            print("JavaScript extraction failed, trying robust manual extraction...")
            return extract_manual_complete(driver)
        
        print(f"Extracted {len(raw_data)} stocks successfully with COMPLETE columns")
        return raw_data
        
    except Exception as e:
        print(f"Error in main extraction: {e}")
        return extract_manual_complete(driver)

def extract_manual_complete(driver):
    """Robust manual extraction with COMPLETE column mapping"""
    print("Using robust manual extraction with ALL columns...")
    
    try:
        stocks_data = []
        
        wait = WebDriverWait(driver, 20)
        
        # Find the main table
        table = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        
        # Get all rows
        rows = table.find_elements(By.TAG_NAME, "tr")
        print(f"Found {len(rows)} total rows")
        
        processed_count = 0
        for i, row in enumerate(rows):
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                
                # Skip rows with too few cells or header rows
                if len(cells) < 9:
                    continue
                
                symbol = cells[0].text.strip()
                
                # Skip invalid symbols
                if (not symbol or len(symbol) > 20 or 
                    any(keyword in symbol for keyword in ['Symbol', 'Last', 'Open', 'High', 'Low', 'Current', 'Change', 'Volume']) or
                    'PSX' in symbol or 'KSE' in symbol):
                    continue
                
                # COMPLETE COLUMN MAPPING FOR PSX - ALL 11 COLUMNS:
                stock = {
                    'symbol': symbol,
                    'sector': cells[1].text.strip() if len(cells) > 1 else '',           # SECTOR
                    'listed_in': cells[2].text.strip() if len(cells) > 2 else '',        # LISTED IN
                    'ldcp': cells[3].text.strip() if len(cells) > 3 else '0',           # LDCP
                    'open': cells[4].text.strip() if len(cells) > 4 else '0',           # Open
                    'high': cells[5].text.strip() if len(cells) > 5 else '0',           # High
                    'low': cells[6].text.strip() if len(cells) > 6 else '0',            # Low
                    'current': cells[7].text.strip() if len(cells) > 7 else '0',        # Current
                    'change': cells[8].text.strip() if len(cells) > 8 else '0',         # Change
                    'change_percent': cells[9].text.strip() if len(cells) > 9 else '0', # Change%
                    'volume': cells[10].text.strip() if len(cells) > 10 else '0'        # Volume
                }
                
                stocks_data.append(stock)
                processed_count += 1
                
                if processed_count % 50 == 0:
                    print(f"   {processed_count} valid stocks processed...")
                
            except Exception as e:
                continue
        
        print(f"Robust extraction: {len(stocks_data)} valid stocks with ALL columns")
        return stocks_data
        
    except Exception as e:
        print(f"Robust extraction error: {e}")
        return []

def clean_numeric_value(value):
    """Clean and convert numeric values properly"""
    if not value or value in ['N/A', '-', '', ' ', 'NAN', 'NULL']:
        return '0'
    
    try:
        # Remove commas, spaces, percentage signs, and other non-numeric characters except decimal point and minus
        cleaned = str(value).strip()
        cleaned = cleaned.replace(',', '').replace(' ', '').replace('%', '')
        cleaned = cleaned.replace('(', '').replace(')', '').replace('$', '')
        cleaned = cleaned.replace('Rs.', '').replace('PKR', '')
        
        # Handle negative values properly
        if cleaned.startswith('-'):
            cleaned = '-' + cleaned[1:].lstrip()
        
        # If it's empty after cleaning, return 0
        if not cleaned:
            return '0'
            
        # Convert to float and back to string to validate
        float_val = float(cleaned)
        return str(float_val)
        
    except:
        return '0'

def validate_stock_data(stock):
    """Validate if stock data looks reasonable"""
    try:
        # Check if symbol is reasonable
        if not stock['symbol'] or len(stock['symbol']) > 20:
            return False
        
        # Check if numeric values are reasonable
        current_price = float(clean_numeric_value(stock['current']))
        if current_price <= 0 or current_price > 100000:  # Assuming no stock price > 100,000
            return False
            
        return True
    except:
        return False

def generate_data_hash(stocks_data):
    """Generate a hash of the data for deduplication"""
    try:
        # Create a sorted string representation of the data
        sorted_data = sorted(
            [json.dumps({k: str(v) for k, v in stock.items()}, sort_keys=True) 
             for stock in stocks_data]
        )
        data_string = ''.join(sorted_data)
        
        # Generate MD5 hash
        return hashlib.md5(data_string.encode('utf-8')).hexdigest()
    except Exception as e:
        print(f"Error generating data hash: {e}")
        return None

def check_duplicate_data(data_hash):
    """Check if this data already exists in Supabase"""
    if not supabase:
        return False
    
    try:
        # Check if data with this hash already exists
        response = supabase.table('stock_data') \
            .select('data_hash') \
            .eq('data_hash', data_hash) \
            .execute()
        
        return len(response.data) > 0
    except Exception as e:
        print(f"Error checking duplicate data: {e}")
        return False

def save_to_supabase(stocks_data, timestamp_str, data_hash):
    """Save data to Supabase instead of local files"""
    if not supabase:
        print("⚠ Supabase not initialized. Skipping remote storage.")
        return False
    
    try:
        # Prepare data for Supabase
        supabase_data = {
            'timestamp': timestamp_str,
            'data_hash': data_hash,
            'total_stocks': len(stocks_data),
            'stock_data': json.dumps(stocks_data),  # Store as JSON
            'created_at': datetime.now().isoformat()
        }
        
        # Insert into Supabase
        response = supabase.table('stock_data').insert(supabase_data).execute()
        
        if response.data:
            print(f"✓ Data successfully saved to Supabase with hash: {data_hash[:12]}...")
            return True
        else:
            print("✗ Failed to save data to Supabase")
            return False
            
    except Exception as e:
        print(f"Error saving to Supabase: {e}")
        return False

def save_complete_data(stocks_data, timestamp_str, data_hash):
    """Save data in BOTH CSV and Excel formats with timestamp-based filenames"""
    if not stocks_data:
        print("No data to save")
        return False
    
    try:
        # Create directory for data if it doesn't exist
        os.makedirs('scraped_data', exist_ok=True)
        
        # Use timestamp for filenames (format: YYYYMMDD_HHMMSS)
        safe_timestamp = timestamp_str.replace(":", "-").replace(" ", "_")
        
        # Define timestamp-based filenames
        csv_filename = f"scraped_data/psx_data_{safe_timestamp}.csv"
        excel_filename = f"scraped_data/psx_data_{safe_timestamp}.xlsx"
        
        # Filter and validate data
        validated_data = []
        for item in stocks_data:
            if validate_stock_data(item):
                formatted_item = {
                    'Symbol': item['symbol'],
                    'Sector': item['sector'],
                    'Listed_In': item['listed_in'],
                    'LDCP': clean_numeric_value(item['ldcp']),
                    'Open': clean_numeric_value(item['open']),
                    'High': clean_numeric_value(item['high']),
                    'Low': clean_numeric_value(item['low']),
                    'Current': clean_numeric_value(item['current']),
                    'Change': clean_numeric_value(item['change']),
                    'Change(%)': clean_numeric_value(item['change_percent']),
                    'Volume': clean_numeric_value(item['volume']),
                    'Data_Hash': data_hash,
                    'Timestamp': timestamp_str
                }
                validated_data.append(formatted_item)
        
        if not validated_data:
            print("No validated data to save")
            return False
            
        df = pd.DataFrame(validated_data)
        
        print(f"Data Verification:")
        print(f"   Total Valid Stocks: {len(df)}")
        print(f"   Data Hash: {data_hash}")
        print(f"   Timestamp: {timestamp_str}")
        
        # ==================== SAVE CSV FILE ====================
        print(f"Saving CSV file: {csv_filename}")
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # Metadata
            writer.writerow(['Pakistan Stock Exchange - Time-based Data'])
            writer.writerow(['Timestamp', timestamp_str])
            writer.writerow(['Data Hash', data_hash])
            writer.writerow(['Total Valid Stocks', len(df)])
            writer.writerow([])
            
            # Headers - ALL 11 COLUMNS + metadata
            headers = ['Symbol', 'Sector', 'Listed_In', 'LDCP', 'Open', 'High', 'Low', 
                      'Current', 'Change', 'Change(%)', 'Volume', 'Data_Hash', 'Timestamp']
            writer.writerow(headers)
            
            # Data
            for _, row in df.iterrows():
                writer.writerow([
                    row['Symbol'],
                    row['Sector'],
                    row['Listed_In'],
                    row['LDCP'],
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Current'],
                    row['Change'],
                    row['Change(%)'],
                    row['Volume'],
                    row['Data_Hash'],
                    row['Timestamp']
                ])
        
        print(f"✓ CSV file saved: {csv_filename}")
        
        # ==================== SAVE EXCEL FILE ====================
        print(f"Saving Excel file: {excel_filename}")
        
        # Create Excel writer
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Create a DataFrame for metadata
            metadata_df = pd.DataFrame({
                'Description': [
                    'Pakistan Stock Exchange - Time-based Data',
                    f'Timestamp: {timestamp_str}',
                    f'Data Hash: {data_hash}',
                    f'Total Valid Stocks: {len(df)}'
                ]
            })
            
            # Write metadata to first sheet
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False, header=False)
            
            # Write main data to second sheet
            df.to_excel(writer, sheet_name='Stock Data', index=False)
            
            # Get workbook and worksheets for formatting
            workbook = writer.book
            metadata_sheet = writer.sheets['Metadata']
            data_sheet = writer.sheets['Stock Data']
            
            # Adjust column widths for better readability
            for column in data_sheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                data_sheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✓ Excel file saved: {excel_filename}")
        
        # Display comprehensive sample for verification
        print("=" * 150)
        print("COMPLETE SAMPLE DATA VERIFICATION (WITH ALL COLUMNS):")
        print('=' * 150)
        print(f"{'Symbol':<10} {'Sector':<8} {'Listed':<8} {'LDCP':<8} {'Open':<8} {'High':<8} {'Low':<8} {'Current':<8} {'Change':<8} {'Chg%':<8} {'Volume':<12}")
        print('-' * 150)
        
        for idx in range(min(10, len(df))):
            row = df.iloc[idx]
            print(f"{row['Symbol']:<10} {row['Sector']:<8} {row['Listed_In']:<8} {row['LDCP']:<8} {row['Open']:<8} {row['High']:<8} {row['Low']:<8} {row['Current']:<8} {row['Change']:<8} {row['Change(%)']:<8} {row['Volume']:<12}")
        
        print('=' * 150)
        
        return True
        
    except Exception as e:
        print(f"Save error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution - Optimized and Fixed"""
    print("=" * 100)
    print("PSX COMPLETE DATA SCRAPER - WITH TIMESTAMP & DEDUPLICATION")
    print("=" * 100)
    print("UPDATES APPLIED:")
    print("   1. TIMESTAMP-BASED FILES: New files for each execution")
    print("   2. DEDUPLICATION: Skips saving if data is identical to previous")
    print("   3. SUPABASE STORAGE: Data saved to remote database")
    print("   4. All original functionality preserved")
    print("=" * 100)
    print()
    
    # Initialize Supabase
    supabase_initialized = init_supabase()
    
    driver = None
    
    try:
        driver = setup_driver()
        
        # Extract COMPLETE data with all columns
        stocks_data = extract_correct_psx_data(driver)
        
        if stocks_data and len(stocks_data) > 0:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("=" * 100)
            print(f"SUCCESS! Extracted {len(stocks_data)} stocks with ALL COLUMNS")
            print(f"Timestamp: {timestamp_str}")
            print("=" * 100)
            
            # Generate data hash for deduplication
            data_hash = generate_data_hash(stocks_data)
            print(f"Data Hash: {data_hash}")
            
            # Check for duplicate data in Supabase
            if supabase_initialized:
                is_duplicate = check_duplicate_data(data_hash)
                if is_duplicate:
                    print("⚠ Data already exists in Supabase. Skipping save.")
                    print("=" * 100)
                    print("SCRAPING COMPLETED - No new data to save")
                    print("=" * 100)
                    return
                else:
                    print("✓ New data detected - Proceeding with save...")
            
            # Save with validation in BOTH formats
            print("Saving complete data...")
            success = save_complete_data(stocks_data, timestamp_str, data_hash)
            
            # Save to Supabase if initialized
            if supabase_initialized and success:
                supabase_success = save_to_supabase(stocks_data, timestamp_str, data_hash)
                if supabase_success:
                    print("✓ Data saved to Supabase successfully")
                else:
                    print("⚠ Failed to save data to Supabase")
            
            if success:
                print("=" * 100)
                print("SCRAPING COMPLETED SUCCESSFULLY!")
                print("=" * 100)
                print("NEW FEATURES:")
                print("   ✓ Time-based filenames prevent overwriting")
                print("   ✓ Deduplication check prevents duplicate saves")
                print("   ✓ Supabase integration for remote storage")
                print("")
                print("ORIGINAL FEATURES MAINTAINED:")
                print("   ✓ Symbols correctly extracted")
                print("   ✓ SECTOR column included")
                print("   ✓ LISTED IN column included") 
                print("   ✓ All columns properly aligned")
                print("   ✓ Data validated and cleaned")
                print("   ✓ Both CSV and Excel files created")
                print("=" * 100)
            else:
                print("Failed to save validated data")
        else:
            print("No data extracted after all attempts!")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if driver:
            print("Closing browser...")
            driver.quit()
            print("Done!")

if __name__ == "__main__":
    main()
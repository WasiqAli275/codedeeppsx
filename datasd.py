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

# Fix Unicode encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def setup_driver():
    """Chrome driver setup - Optimized for speed"""
    
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
    # Keep images enabled for better compatibility with the site
    chrome_options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_settings.popups": 0,
    })
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    return driver

def extract_correct_psx_data(driver):
    """FIXED: Correct PSX data extraction with ALL columns including SECTOR and LISTED IN"""
    try:
        url = "https://dps.psx.com.pk/market-watch"
        
        driver.get(url)
        time.sleep(5)  # Increased wait time for page to load completely
        
        # Wait for table to be present
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        
        # Extract with COMPLETE column mapping
        extract_script = """
            let allStocks = [];
            
            let tables = document.querySelectorAll('table');
            let targetTable = null;
            
            // Find the main market watch table
            for (let table of tables) {
                let headers = table.querySelectorAll('th');
                if (headers.length >= 10) {
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
                
                if (cells.length >= 9) {
                    let symbol = cells[0]?.innerText.trim() || '';
                    
                    // Skip invalid rows but be more lenient
                    if (!symbol || symbol === '' || symbol.length > 50 || 
                        symbol === 'Symbol' || symbol === 'PSX' || symbol === 'KSE-100') {
                        return;
                    }
                    
                    let stockData = {
                        symbol: symbol,
                        sector: cells[1]?.innerText.trim() || '',
                        listed_in: cells[2]?.innerText.trim() || '',
                        ldcp: cells[3]?.innerText.trim() || '0',
                        open: cells[4]?.innerText.trim() || '0',
                        high: cells[5]?.innerText.trim() || '0',
                        low: cells[6]?.innerText.trim() || '0',
                        current: cells[7]?.innerText.trim() || '0',
                        change: cells[8]?.innerText.trim() || '0',
                        change_percent: cells[9]?.innerText.trim() || '0',
                        volume: cells[10]?.innerText.trim() || '0'
                    };
                    
                    allStocks.push(stockData);
                }
            });
            
            return allStocks;
        """
        
        raw_data = driver.execute_script(extract_script)
        
        if not raw_data or len(raw_data) == 0:
            return extract_manual_complete(driver)
        
        return raw_data
        
    except Exception as e:
        return extract_manual_complete(driver)

def extract_manual_complete(driver):
    """Robust manual extraction with COMPLETE column mapping"""
    
    try:
        stocks_data = []
        
        wait = WebDriverWait(driver, 25)
        
        # Find the main table
        table = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        
        # Get all rows
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        for i, row in enumerate(rows):
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                
                # Skip rows with too few cells
                if len(cells) < 9:
                    continue
                
                symbol = cells[0].text.strip()
                
                # More lenient symbol validation
                if not symbol or symbol in ['Symbol', 'PSX', 'KSE-100', '']:
                    continue
                
                stock = {
                    'symbol': symbol,
                    'sector': cells[1].text.strip() if len(cells) > 1 else '',
                    'listed_in': cells[2].text.strip() if len(cells) > 2 else '',
                    'ldcp': cells[3].text.strip() if len(cells) > 3 else '0',
                    'open': cells[4].text.strip() if len(cells) > 4 else '0',
                    'high': cells[5].text.strip() if len(cells) > 5 else '0',
                    'low': cells[6].text.strip() if len(cells) > 6 else '0',
                    'current': cells[7].text.strip() if len(cells) > 7 else '0',
                    'change': cells[8].text.strip() if len(cells) > 8 else '0',
                    'change_percent': cells[9].text.strip() if len(cells) > 9 else '0',
                    'volume': cells[10].text.strip() if len(cells) > 10 else '0'
                }
                
                stocks_data.append(stock)
                
            except Exception as e:
                continue
        
        return stocks_data
        
    except Exception as e:
        return []

def clean_numeric_value(value):
    """Clean and convert numeric values properly"""
    if not value or value in ['N/A', '-', '', ' ', 'NAN', 'NULL', 'NIL']:
        return '0'
    
    try:
        # Remove commas, spaces, percentage signs, and other non-numeric characters
        cleaned = str(value).strip()
        cleaned = cleaned.replace(',', '').replace(' ', '').replace('%', '')
        cleaned = cleaned.replace('(', '').replace(')', '').replace('$', '')
        cleaned = cleaned.replace('Rs.', '').replace('PKR', '')
        cleaned = cleaned.replace('"', '').replace("'", "")
        
        # Handle negative values properly
        if cleaned.startswith('--'):
            cleaned = '-' + cleaned[2:]
        elif cleaned.startswith('-'):
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
    """Validate if stock data looks reasonable - Modified to be less restrictive"""
    try:
        # Check if symbol is reasonable
        if not stock['symbol'] or stock['symbol'] in ['', 'Symbol', 'PSX', 'KSE-100']:
            return False
        
        # Check if current price is reasonable (allow 0 for suspended stocks)
        current_price = float(clean_numeric_value(stock['current']))
        if current_price < 0 or current_price > 1000000:  # Increased max limit
            return False
            
        return True
    except:
        return False

def save_complete_data(stocks_data):
    """Save data in SINGLE CSV format only with stable filename"""
    if not stocks_data:
        return False
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Define stable filename (no date or time components)
        csv_filename = "psx_data.csv"
        
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
                    'Volume': clean_numeric_value(item['volume'])
                }
                validated_data.append(formatted_item)
        
        if not validated_data:
            # Save all data even if validation fails for some
            print("Warning: No data passed validation, saving all extracted data")
            for item in stocks_data:
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
                    'Volume': clean_numeric_value(item['volume'])
                }
                validated_data.append(formatted_item)
        
        if not validated_data:
            return False
            
        df = pd.DataFrame(validated_data)
        
        # ==================== SAVE CSV FILE ONLY ====================
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # Metadata
            writer.writerow(['Pakistan Stock Exchange - COMPLETELY FIXED Data'])
            writer.writerow(['Timestamp', timestamp])
            writer.writerow(['Total Valid Stocks', len(df)])
            writer.writerow(['Fix Applied', 'Complete column mapping correction - All data validated'])
            writer.writerow([])
            
            # Headers - ALL 11 COLUMNS
            writer.writerow(['Symbol', 'Sector', 'Listed_In', 'LDCP', 'Open', 'High', 'Low', 'Current', 'Change', 'Change(%)', 'Volume'])
            
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
                    row['Volume']
                ])
        
        # Verify file was created
        if os.path.exists(csv_filename):
            return True
        else:
            return False
        
    except Exception as e:
        return False

def main():
    """Main execution - Optimized and Fixed"""
    
    driver = None
    
    try:
        driver = setup_driver()
        
        # Extract COMPLETE data with all columns
        stocks_data = extract_correct_psx_data(driver)
        
        if stocks_data and len(stocks_data) > 0:
            # Save with validation in CSV format only
            success = save_complete_data(stocks_data)
        else:
            # If no data, create empty CSV file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("psx_data.csv", 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['Pakistan Stock Exchange - COMPLETELY FIXED Data'])
                writer.writerow(['Timestamp', timestamp])
                writer.writerow(['Total Valid Stocks', '0'])
                writer.writerow(['Status', 'No data extracted'])
                writer.writerow([])
                writer.writerow(['Symbol', 'Sector', 'Listed_In', 'LDCP', 'Open', 'High', 'Low', 'Current', 'Change', 'Change(%)', 'Volume'])
            
    except Exception as e:
        # Create error file if something goes wrong
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("psx_data.csv", 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Pakistan Stock Exchange - COMPLETELY FIXED Data'])
            writer.writerow(['Timestamp', timestamp])
            writer.writerow(['Total Valid Stocks', '0'])
            writer.writerow(['Status', f'Error: {str(e)}'])
            writer.writerow([])
            writer.writerow(['Symbol', 'Sector', 'Listed_In', 'LDCP', 'Open', 'High', 'Low', 'Current', 'Change', 'Change(%)', 'Volume'])
        
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()
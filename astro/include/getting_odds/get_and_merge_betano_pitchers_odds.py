from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import logging
import csv
import os
import pandas as pd
from datetime import datetime

"""
    This script is used to scrape the odds for pitchers from Betano. A Brazilians bookmaker.
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
print("[DEBUG] Current working directory:", os.getcwd())
print("[DEBUG] __file__:", __file__)
print("[DEBUG] Files in current dir:", os.listdir("."))

def setup_driver(headless=True):
    print("[DEBUG] Starting Chrome driver configuration...")
    logger.info("Starting Chrome driver configuration...")
    
    # Configure Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")  # New headless mode version
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=900,9000")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    try:
        print("[DEBUG] Attempting to connect to remote ChromeDriver...")
        # Use remote ChromeDriver from Selenium container
        remote_url = "http://remote_chromedriver:4444/wd/hub"
        print(f"[DEBUG] Remote URL: {remote_url}")
        
        driver = webdriver.Remote(remote_url, options=chrome_options)
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
        print("[DEBUG] Remote Chrome driver initialized successfully!")
        logger.info("Remote Chrome driver initialized successfully!")
        return driver
    except Exception as e:
        error_msg = f"Error initializing remote Chrome driver: {str(e)}"
        print(f"[DEBUG] {error_msg}")
        logger.error(error_msg)
        
        # Fallback to local ChromeDriver if remote fails
        print("[DEBUG] Attempting fallback to local ChromeDriver...")
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
            print("[DEBUG] Local Chrome driver initialized successfully!")
            logger.info("Local Chrome driver initialized successfully!")
            return driver
        except Exception as local_e:
            final_error = f"Failed to initialize Chrome driver (both remote and local). Error: {str(e)}. Local fallback error: {str(local_e)}"
            print(f"[DEBUG] {final_error}")
            logger.error(final_error)
            raise Exception(final_error)

def wait_for_element(driver, by, value, timeout=10):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except Exception as e:
        logger.error(f"Timeout waiting for element {value}: {str(e)}")
        return None
    



def merge_with_betano_data(resultados):
    """
    Adds Betano data columns to existing betting data without removing any existing columns
    """
    if not resultados:
        logger.warning("No results to merge")
        return None
    
    # Filter empty rows or rows with empty values
    resultados_filtrados = []
    for resultado in resultados:
        # Check if all values are non-empty
        if all(resultado.values()):
            resultados_filtrados.append(resultado)
    
    if not resultados_filtrados:
        logger.warning("No valid results to merge after filtering empty rows")
        return None
    
    try:
        # Convert results to DataFrame
        betano_df = pd.DataFrame(resultados_filtrados)
        logger.info(f"Betano data columns: {list(betano_df.columns)}")
        logger.info(f"Betano data shape: {betano_df.shape}")
        
        # Get the include/Getting Odds directory (same directory as this script)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_date = datetime.now().strftime("%Y-%m-%d")
        betting_data_path = os.path.join(current_dir, f'betting_data_{current_date}.csv')
        betting_data_path = os.path.abspath(betting_data_path)
        print("[DEBUG] Betting data path:", betting_data_path)
        
        if not os.path.exists(betting_data_path):
            logger.error(f"Betting data file not found: {betting_data_path}")
            return None
        
        logger.info(f"Reading existing betting data from: {betting_data_path}")
        betting_data = pd.read_csv(betting_data_path)
        logger.info(f"Betting data columns: {list(betting_data.columns)}")
        logger.info(f"Betting data shape: {betting_data.shape}")
        
        # Remove the Team column from betano_data
        betano_df = betano_df.drop(columns=['Team'])
        
        # Rename all Betano columns to have 'betano_' prefix
        betano_df = betano_df.rename(columns={
            'Player': 'betano_player',
            'Line': 'betano_line',
            'Over Line': 'betano_over_line',
            'Under Line': 'betano_under_line'
        })
        
        # Perform the join on the Player column (now betano_player)
        # Using left join to keep all records from betting_data
        logger.info("Adding Betano data to existing betting data...")
        
        # Check if Player column exists in betting_data
        if 'Player' not in betting_data.columns:
            logger.error("Player column not found in betting data")
            return None
            
        merged_data = pd.merge(betting_data, betano_df, left_on='Player', right_on='betano_player', how='left')
        
        # Remove the duplicate betano_player column since we already have Player
        if 'betano_player' in merged_data.columns:
            merged_data = merged_data.drop(columns=['betano_player'])
        
        # Save back to the original betting_data file
        merged_data.to_csv(betting_data_path, index=False)
        print(f"[DEBUG] Saved merged data to: {betting_data_path}")
        logger.info(f"Betano data added successfully to: {betting_data_path}")
        logger.info(f"Total rows in merged data: {len(merged_data)}")
        
        return merged_data
        
    except Exception as e:
        logger.error(f"Error adding Betano data: {str(e)}")
        return None

def save_to_csv(resultados, filename=None):
    """
    Saves the results to a CSV file (for debugging purposes)
    """
    if not resultados:
        logger.warning("No results to save")
        return
    
    # Filter empty rows or rows with empty values
    resultados_filtrados = []
    for resultado in resultados:
        # Check if all values are non-empty
        if all(resultado.values()):
            resultados_filtrados.append(resultado)
    
    if not resultados_filtrados:
        logger.warning("No valid results to save after filtering empty rows")
        return
    
    if filename is None:
        # Create filename with date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"betano_strikeouts_{timestamp}.csv"
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Team', 'betano_player', 'betano_line', 'betano_over_line', 'betano_under_line']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for resultado in resultados_filtrados:
                # Rename the keys to match the new column names
                renamed_resultado = {
                    'Team': resultado['Team'],
                    'betano_player': resultado['Player'],
                    'betano_line': resultado['Line'],
                    'betano_over_line': resultado['Over Line'],
                    'betano_under_line': resultado['Under Line']
                }
                writer.writerow(renamed_resultado)
            
            logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {str(e)}")

def scrape_betano(headless=True, merge_data=True, save_debug_csv=False):
    print("[DEBUG] Starting scrape_betano function")
    print(f"[DEBUG] headless={headless}, merge_data={merge_data}, save_debug_csv={save_debug_csv}")
    
    driver = None
    try:
        print("[DEBUG] About to setup driver...")
        driver = setup_driver(headless)
        print("[DEBUG] Driver setup completed successfully")
        
        url = "https://www.betano.bet.br/sport/beisebol/eua/mlb/1662/?bt=strikeouts"
        
        logger.info(f"Accessing URL: {url}")
        print(f"[DEBUG] About to access URL: {url}")
        # Access the page
        driver.get(url)
        print("[DEBUG] URL accessed successfully")
        
        # Wait for the page to load
        logger.info("Waiting for initial page load...")
        print("[DEBUG] Waiting for initial page load...")
        time.sleep(10)  # Increased initial wait time
        print("[DEBUG] Initial wait completed")
        
        # Check if the page loaded correctly
        logger.info("Checking page title...")
        print("[DEBUG] Checking page title...")
        logger.info(f"Page title: {driver.title}")
        print(f"[DEBUG] Page title: {driver.title}")
        
        # Scroll the page to load all elements
        
        # Wait a bit more to ensure all elements are loaded
        print("[DEBUG] Waiting additional time for elements...")
        time.sleep(5)
        print("[DEBUG] Additional wait completed")
        
        # Find all multi-outcome elements
        logger.info("Looking for elements with class 'multi-outcome'...")
        print("[DEBUG] Looking for multi-outcome elements...")
        multi_outcomes = driver.find_elements(By.CLASS_NAME, "multi-outcome")
        logger.info(f"Found {len(multi_outcomes)} multi-outcome elements")
        print(f"[DEBUG] Found {len(multi_outcomes)} multi-outcome elements")
        
        resultados = []
        
        for index, multi_outcome in enumerate(multi_outcomes, 1):
            try:
                logger.info(f"Processing multi-outcome {index} of {len(multi_outcomes)}")
                print(f"[DEBUG] Processing multi-outcome {index} of {len(multi_outcomes)}")
                
                # Find all teams
                times = multi_outcome.find_elements(By.CLASS_NAME, "team")
                logger.info(f"Found {len(times)} teams in this multi-outcome")
                print(f"[DEBUG] Found {len(times)} teams in this multi-outcome")
                
                for time_index, time_element in enumerate(times):
                    try:
                        # Extract team name
                        time_name_element = time_element.find_element(By.CLASS_NAME, "team-header__title")
                        time_name = time_name_element.text if time_name_element else "N/A"
                        logger.info(f"Team name found: {time_name}")
                        print(f"[DEBUG] Team name found: {time_name}")
                        
                        # Find the player
                        jogador_element = time_element.find_element(By.CLASS_NAME, "row-title__text")
                        jogador = jogador_element.text if jogador_element else "N/A"
                        logger.info(f"Player name found: {jogador}")
                        print(f"[DEBUG] Player name found: {jogador}")
                        
                        # Find the line (handicap)
                        linha_element = time_element.find_element(By.CLASS_NAME, "handicap__single-item")
                        linha = linha_element.text if linha_element else "N/A"
                        logger.info(f"Line found: {linha}")
                        print(f"[DEBUG] Line found: {linha}")
                        
                        # Find the odds (over and under)
                        odds_elements = time_element.find_elements(By.CSS_SELECTOR, ".selections__selection span")
                        mais_de = odds_elements[0].text if len(odds_elements) > 0 else "N/A"
                        menos_de = odds_elements[1].text if len(odds_elements) > 1 else "N/A"
                        logger.info(f"Odds found: Over {mais_de}, Under {menos_de}")
                        print(f"[DEBUG] Odds found: Over {mais_de}, Under {menos_de}")
                        
                        # Only add results that have valid data
                        if time_name != "N/A" and jogador != "N/A" and linha != "N/A" and mais_de != "N/A" and menos_de != "N/A":
                            resultado = {
                                "Team": time_name,
                                "Player": jogador,
                                "Line": linha,
                                "Over Line": mais_de,
                                "Under Line": menos_de
                            }
                            
                            resultados.append(resultado)
                            logger.info(f"Team {time_index+1} of multi-outcome {index} processed successfully")
                            print(f"[DEBUG] Team {time_index+1} of multi-outcome {index} processed successfully")
                        else:
                            logger.warning(f"Incomplete data for team {time_index+1} of multi-outcome {index}, ignoring")
                            print(f"[DEBUG] Incomplete data for team {time_index+1} of multi-outcome {index}, ignoring")
                        
                    except Exception as e:
                        logger.error(f"Error processing team {time_index+1} of multi-outcome {index}: {str(e)}")
                        print(f"[DEBUG] Error processing team {time_index+1} of multi-outcome {index}: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error processing multi-outcome {index}: {str(e)}")
                print(f"[DEBUG] Error processing multi-outcome {index}: {str(e)}")
                continue
        
        logger.info(f"Total results collected: {len(resultados)}")
        print(f"[DEBUG] Total results collected: {len(resultados)}")
        
        # Save debug CSV if requested
        if save_debug_csv and resultados:
            print("[DEBUG] Saving debug CSV...")
            save_to_csv(resultados)
        
        # Add Betano data to betting data if requested
        if merge_data and resultados:
            print("[DEBUG] About to call merge_with_betano_data")
            merged_data = merge_with_betano_data(resultados)
            if merged_data is not None:
                logger.info("Betano data added successfully")
                print("[DEBUG] Betano data added successfully")
            else:
                logger.warning("Data addition failed, but scraping was successful")
                print("[DEBUG] Data addition failed, but scraping was successful")
            
        return resultados
        
    except Exception as e:
        error_msg = f"Error during scraping: {str(e)}"
        logger.error(error_msg)
        print(f"[DEBUG] {error_msg}")
        raise Exception(f"Scraping failed: {str(e)}")
        
    finally:
        if driver:
            logger.info("Closing Chrome driver...")
            print("[DEBUG] Closing Chrome driver...")
            driver.quit()

def main():
    print("[DEBUG] Starting main function")
    import argparse
    
    # Configure command line arguments
    parser = argparse.ArgumentParser(description='Scraper for Betano strikeout data')
    parser.add_argument('--no-headless', action='store_true', help='Run browser in visible mode (not headless)')
    parser.add_argument('--no-merge', action='store_true', help='Do not add Betano data to betting data')
    parser.add_argument('--save-debug-csv', action='store_true', help='Save scraped data to CSV file for debugging')
    args = parser.parse_args()
    
    print(f"[DEBUG] Arguments: headless={not args.no_headless}, merge={not args.no_merge}, save_debug={args.save_debug_csv}")
    
    logger.info("Starting scraping script...")
    resultados = scrape_betano(headless=not args.no_headless, merge_data=not args.no_merge, save_debug_csv=args.save_debug_csv)
    
    if resultados:
        logger.info("\nResults found:")
        for resultado in resultados:
            print("\n-------------------")
            print(f"Team: {resultado['Team']}")
            print(f"Betano Player: {resultado['Player']}")
            print(f"Betano Line: {resultado['Line']}")
            print(f"Betano Over: {resultado['Over Line']}")
            print(f"Betano Under: {resultado['Under Line']}")
    else:
        logger.error("Could not obtain results.")

if __name__ == "__main__":
    main() 
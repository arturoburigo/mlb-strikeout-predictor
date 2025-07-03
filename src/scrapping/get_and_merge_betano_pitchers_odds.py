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

def setup_driver(headless=True):
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
        # Use webdriver-manager to automatically download the correct ChromeDriver
        logger.info("Downloading compatible ChromeDriver...")
        service = Service(ChromeDriverManager().install())
        
        # Initialize the driver
        logger.info("Attempting to initialize Chrome driver...")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
        logger.info("Chrome driver initialized successfully!")
        return driver
    except Exception as e:
        logger.error(f"Error initializing Chrome driver: {str(e)}")
        raise

def wait_for_element(driver, by, value, timeout=10):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except Exception as e:
        logger.error(f"Timeout waiting for element {value}: {str(e)}")
        return None
    

def merge_with_betting_data(resultados):
    """
    Merges the scraped Betano data with existing betting data
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
        
        # Get the current working directory and go up one level to the project root
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Read the existing betting data CSV file
        betting_data_path = os.path.join(current_dir, 'betting_data_2025-07-02.csv')
        
        if not os.path.exists(betting_data_path):
            logger.error(f"Betting data file not found: {betting_data_path}")
            return None
        
        logger.info(f"Reading existing betting data from: {betting_data_path}")
        betting_data = pd.read_csv(betting_data_path)
        logger.info(f"Betting data columns: {list(betting_data.columns)}")
        logger.info(f"Betting data shape: {betting_data.shape}")
        
        # Remove rows where Over Line or Under Line is empty
        betting_data = betting_data.dropna(subset=['Over Line', 'Under Line'])
        
        # Remove the specified columns from betting_data - only those that exist
        columns_to_drop = ['Over Line', 'Under Line', 'Over Odds', 'Under Odds']
        # Only drop columns that actually exist in the dataframe
        columns_to_drop = [col for col in columns_to_drop if col in betting_data.columns]
        betting_data = betting_data.drop(columns=columns_to_drop)
        
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
        logger.info("Merging Betano data with existing betting data...")
        
        # Check if Player column exists in betting_data
        if 'Player' not in betting_data.columns:
            logger.error("Player column not found in betting data")
            return None
            
        merged_data = pd.merge(betting_data, betano_df, left_on='Player', right_on='betano_player', how='left')
        
        # Remove the duplicate betano_player column since we already have Player
        if 'betano_player' in merged_data.columns:
            merged_data = merged_data.drop(columns=['betano_player'])
        
        # Remove rows where the betano_line column is empty
        merged_data = merged_data.dropna(subset=['betano_line'])
        
        # Save back to the original betting_data file
        merged_data.to_csv(betting_data_path, index=False)
        
        logger.info(f"Files merged successfully. Original file has been updated: {betting_data_path}")
        logger.info(f"Total rows in merged data: {len(merged_data)}")
        
        return merged_data
        
    except Exception as e:
        logger.error(f"Error merging data: {str(e)}")
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
    driver = None
    try:
        driver = setup_driver(headless)
        url = "https://www.betano.bet.br/sport/beisebol/eua/mlb/1662/?bt=strikeouts"
        
        logger.info(f"Accessing URL: {url}")
        # Access the page
        driver.get(url)
        
        # Wait for the page to load
        logger.info("Waiting for initial page load...")
        time.sleep(10)  # Increased initial wait time
        
        # Check if the page loaded correctly
        logger.info("Checking page title...")
        logger.info(f"Page title: {driver.title}")
        
        # Scroll the page to load all elements
        
        # Wait a bit more to ensure all elements are loaded
        time.sleep(5)
        
        # Find all multi-outcome elements
        logger.info("Looking for elements with class 'multi-outcome'...")
        multi_outcomes = driver.find_elements(By.CLASS_NAME, "multi-outcome")
        logger.info(f"Found {len(multi_outcomes)} multi-outcome elements")
        
        resultados = []
        
        for index, multi_outcome in enumerate(multi_outcomes, 1):
            try:
                logger.info(f"Processing multi-outcome {index} of {len(multi_outcomes)}")
                
                # Find all teams
                times = multi_outcome.find_elements(By.CLASS_NAME, "team")
                logger.info(f"Found {len(times)} teams in this multi-outcome")
                
                for time_index, time_element in enumerate(times):
                    try:
                        # Extract team name
                        time_name_element = time_element.find_element(By.CLASS_NAME, "team-header__title")
                        time_name = time_name_element.text if time_name_element else "N/A"
                        logger.info(f"Team name found: {time_name}")
                        
                        # Find the player
                        jogador_element = time_element.find_element(By.CLASS_NAME, "row-title__text")
                        jogador = jogador_element.text if jogador_element else "N/A"
                        logger.info(f"Player name found: {jogador}")
                        
                        # Find the line (handicap)
                        linha_element = time_element.find_element(By.CLASS_NAME, "handicap__single-item")
                        linha = linha_element.text if linha_element else "N/A"
                        logger.info(f"Line found: {linha}")
                        
                        # Find the odds (over and under)
                        odds_elements = time_element.find_elements(By.CSS_SELECTOR, ".selections__selection span")
                        mais_de = odds_elements[0].text if len(odds_elements) > 0 else "N/A"
                        menos_de = odds_elements[1].text if len(odds_elements) > 1 else "N/A"
                        logger.info(f"Odds found: Over {mais_de}, Under {menos_de}")
                        
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
                        else:
                            logger.warning(f"Incomplete data for team {time_index+1} of multi-outcome {index}, ignoring")
                        
                    except Exception as e:
                        logger.error(f"Error processing team {time_index+1} of multi-outcome {index}: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error processing multi-outcome {index}: {str(e)}")
                continue
        
        logger.info(f"Total results collected: {len(resultados)}")
        
        # Save debug CSV if requested
        if save_debug_csv and resultados:
            save_to_csv(resultados)
        
        # Merge results with betting data if requested
        if merge_data and resultados:
            merged_data = merge_with_betting_data(resultados)
            if merged_data is not None:
                logger.info("Data merging completed successfully")
            else:
                logger.warning("Data merging failed, but scraping was successful")
            
        return resultados
        
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        return None
        
    finally:
        if driver:
            logger.info("Closing Chrome driver...")
            driver.quit()

def main():
    import argparse
    
    # Configure command line arguments
    parser = argparse.ArgumentParser(description='Scraper for Betano strikeout data')
    parser.add_argument('--no-headless', action='store_true', help='Run browser in visible mode (not headless)')
    parser.add_argument('--no-merge', action='store_true', help='Do not merge results with betting data')
    parser.add_argument('--save-debug-csv', action='store_true', help='Save scraped data to CSV file for debugging')
    args = parser.parse_args()
    
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
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import json
import time

def test_web_scraping():
    """
    Test web scraping functionality using a well-known site.
    This will verify that requests and BeautifulSoup are working correctly.
    """
    print("Testing web scraping functionality...")
    
    # Test with a simple, reliable site
    url = "https://httpbin.org/html"
    
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"Making request to: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract some basic information
        title = soup.find('title')
        h1 = soup.find('h1')
        
        print(f"Page title: {title.text if title else 'No title found'}")
        print(f"Main heading: {h1.text if h1 else 'No h1 found'}")
        
        # Test with a more complex site - Wikipedia
        print("\n" + "="*50)
        print("Testing with Wikipedia...")
        
        wiki_url = "https://en.wikipedia.org/wiki/Baseball"
        
        print(f"Making request to: {wiki_url}")
        wiki_response = requests.get(wiki_url, headers=headers)
        wiki_response.raise_for_status()
        
        wiki_soup = BeautifulSoup(wiki_response.content, 'html.parser')
        
        # Find the main title
        wiki_title = wiki_soup.find('h1', {'id': 'firstHeading'})
        print(f"Wikipedia page title: {wiki_title.text if wiki_title else 'No title found'}")
        
        # Find some links
        links = wiki_soup.find_all('a', href=True)[:5]
        print(f"First 5 links found:")
        for i, link in enumerate(links, 1):
            print(f"  {i}. {link.text.strip()} -> {link['href']}")
        
        # Test pandas HTML parsing
        print("\n" + "="*50)
        print("Testing pandas HTML table parsing...")
        
        # Use a site with tables - fix the URL
        table_url = "https://en.wikipedia.org/wiki/List_of_Major_League_Baseball_teams"
        
        print(f"Making request to: {table_url}")
        table_response = requests.get(table_url, headers=headers)
        table_response.raise_for_status()
        
        # Try to find tables
        tables = pd.read_html(table_response.content)
        print(f"Found {len(tables)} tables on the page")
        
        if tables:
            print(f"First table shape: {tables[0].shape}")
            print(f"First table columns: {list(tables[0].columns)}")
            print(f"First few rows of first table:")
            print(tables[0].head(3))
        
        print("\n" + "="*50)
        print("✅ All web scraping tests passed successfully!")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_baseball_reference_with_session():
    """
    Test Baseball Reference using a session with cookies to handle Cloudflare challenges.
    """
    print("\n" + "="*50)
    print("Testing Baseball Reference with session and cookies...")
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    # Enhanced headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    }
    
    session.headers.update(headers)
    
    # Test with a simple Baseball Reference page
    url = "https://www.baseball-reference.com/teams/ATH/2025.shtml"
    
    try:
        print(f"Making request to: {url}")
        print(f"Using session with headers: {json.dumps(dict(session.headers), indent=2)}")
        
        # First, try to access the main site to get cookies
        print("Step 1: Accessing main site to get cookies...")
        main_response = session.get("https://www.baseball-reference.com/", timeout=30)
        print(f"Main site status: {main_response.status_code}")
        
        if main_response.status_code == 200:
            print("✅ Main site accessible, got cookies")
            print(f"Cookies: {dict(session.cookies)}")
        else:
            print(f"⚠️ Main site returned {main_response.status_code}")
        
        # Wait a bit before the next request
        time.sleep(2)
        
        # Now try the specific page
        print("\nStep 2: Accessing specific team page...")
        response = session.get(url, timeout=30)
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if we got a challenge page
            if "Just a moment" in response.text or "cf-mitigated" in response.headers:
                print("❌ Still getting Cloudflare challenge page")
                return False
            
            # Try to find the team name
            team_name = soup.find('h1')
            print(f"Team name: {team_name.text if team_name else 'No team name found'}")
            
            # Try to find tables
            tables = soup.find_all('table')
            print(f"Found {len(tables)} tables on the page")
            
            # Look for the pitching table specifically
            pitching_table = soup.find('table', {'id': 'players_standard_pitching'})
            if pitching_table:
                print("✅ Found pitching table!")
                
                # Try to find some pitcher names
                name_elements = pitching_table.find_all(attrs={'data-append-csv': True})
                print(f"Found {len(name_elements)} potential pitcher elements")
                
                if name_elements:
                    print("First few pitcher names:")
                    for i, element in enumerate(name_elements[:3]):
                        print(f"  {i+1}. {element.text.strip()}")
            else:
                print("❌ Pitching table not found")
            
            print("✅ Baseball Reference test completed successfully!")
            return True
        else:
            print(f"❌ Got status code {response.status_code}")
            print(f"Response content preview: {response.text[:500]}")
            return False
        
    except Exception as e:
        print(f"❌ Baseball Reference test failed: {e}")
        return False

def test_baseball_reference_with_enhanced_headers():
    """
    Test Baseball Reference with enhanced browser-like headers.
    """
    print("\n" + "="*50)
    print("Testing Baseball Reference with enhanced headers...")
    
    # Test with a simple Baseball Reference page
    url = "https://www.baseball-reference.com/teams/ATH/2025.shtml"
    
    # Enhanced headers to look more like a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    }
    
    try:
        print(f"Making request to: {url}")
        print(f"Using headers: {json.dumps(headers, indent=2)}")
        
        response = requests.get(url, headers=headers, timeout=30)
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find the team name
            team_name = soup.find('h1')
            print(f"Team name: {team_name.text if team_name else 'No team name found'}")
            
            # Try to find tables
            tables = soup.find_all('table')
            print(f"Found {len(tables)} tables on the page")
            
            # Look for the pitching table specifically
            pitching_table = soup.find('table', {'id': 'players_standard_pitching'})
            if pitching_table:
                print("✅ Found pitching table!")
                
                # Try to find some pitcher names
                name_elements = pitching_table.find_all(attrs={'data-append-csv': True})
                print(f"Found {len(name_elements)} potential pitcher elements")
                
                if name_elements:
                    print("First few pitcher names:")
                    for i, element in enumerate(name_elements[:3]):
                        print(f"  {i+1}. {element.text.strip()}")
            else:
                print("❌ Pitching table not found")
            
            print("✅ Baseball Reference test completed successfully!")
            return True
        else:
            print(f"❌ Got status code {response.status_code}")
            print(f"Response content preview: {response.text[:500]}")
            return False
        
    except Exception as e:
        print(f"❌ Baseball Reference test failed: {e}")
        return False

def test_curl_vs_requests():
    """
    Compare what curl sees vs what requests sees.
    """
    print("\n" + "="*50)
    print("Testing curl vs requests comparison...")
    
    url = "https://httpbin.org/headers"
    
    try:
        # Test with requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        print("Making requests call...")
        response = requests.get(url, headers=headers)
        print(f"Requests status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Requests User-Agent: {data.get('headers', {}).get('User-Agent', 'Not found')}")
        
        print("\n" + "="*30)
        print("Now test with curl...")
        import subprocess
        result = subprocess.run(['curl', '-s', url], capture_output=True, text=True)
        print(f"Curl exit code: {result.returncode}")
        if result.returncode == 0:
            try:
                curl_data = json.loads(result.stdout)
                print(f"Curl User-Agent: {curl_data.get('headers', {}).get('User-Agent', 'Not found')}")
            except:
                print("Could not parse curl response as JSON")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting web scraping tests...")
    print(f"Timestamp: {datetime.now()}")
    
    # Run basic tests
    basic_success = test_web_scraping()
    
    # Run curl vs requests comparison
    comparison_success = test_curl_vs_requests()
    
    # Run Baseball Reference with session
    br_session_success = test_baseball_reference_with_session()
    
    # Run Baseball Reference with enhanced headers
    br_enhanced_success = test_baseball_reference_with_enhanced_headers()
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Basic web scraping: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    print(f"Curl vs Requests: {'✅ PASSED' if comparison_success else '❌ FAILED'}")
    print(f"Baseball Reference (session): {'✅ PASSED' if br_session_success else '❌ FAILED'}")
    print(f"Baseball Reference (enhanced): {'✅ PASSED' if br_enhanced_success else '❌ FAILED'}")
    
    if basic_success and (br_session_success or br_enhanced_success):
        print("\n🎉 All tests passed! Your web scraping setup is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.") 
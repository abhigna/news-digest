import requests
import json
import os
import xml.etree.ElementTree as ET
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Constants
RSS_URL = "https://hnrss.org/classic"
JINA_API_BASE = "https://r.jina.ai/"
PROCESSED_ITEMS_FILE = "processed_items.json"
ARTICLES_DIRECTORY = "articles"

def load_processed_items():
    """Load the list of already processed items and check articles directory."""
    processed_items = {}
    
    # Load from processed_items.json if it exists
    if os.path.exists(PROCESSED_ITEMS_FILE):
        try:
            with open(PROCESSED_ITEMS_FILE, 'r') as f:
                processed_items = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error reading {PROCESSED_ITEMS_FILE}, starting with empty set")
    else:
        logger.info(f"{PROCESSED_ITEMS_FILE} not found, starting with empty set")
    
    # Check articles directory for already downloaded files
    if os.path.exists(ARTICLES_DIRECTORY):
        for filename in os.listdir(ARTICLES_DIRECTORY):
            if filename.endswith('.json'):
                # Extract guid from filename (remove .json extension)
                guid_id = filename.rsplit('.', 1)[0]
                full_guid = f"item?id={guid_id}"
                if full_guid not in processed_items:
                    logger.info(f"Found existing article not in processed_items: {filename}")
                    processed_items[full_guid] = {
                        "title": "Unknown (found in articles directory)",
                        "processed_at": datetime.now().isoformat()
                    }
    
    logger.info(f"Loaded {len(processed_items)} processed items")
    return processed_items

def save_processed_items(processed_items):
    """Save the updated list of processed items."""
    with open(PROCESSED_ITEMS_FILE, 'w') as f:
        json.dump(processed_items, f, indent=2)
    logger.info(f"Saved {len(processed_items)} processed items to {PROCESSED_ITEMS_FILE}")

def parse_rss_feed(rss_content):
    """Parse the RSS feed and extract items."""
    try:
        # Parse XML content
        root = ET.fromstring(rss_content)
        
        # Extract items
        items = []
        for item in root.findall('.//item'):
            title_elem = item.find('title')
            title = title_elem.text if title_elem is not None else ""
            if title and title.startswith('<![CDATA[') and title.endswith(']]>'):
                title = title[9:-3].strip()
                
            link_elem = item.find('link')
            link = link_elem.text if link_elem is not None else ""
            
            guid_elem = item.find('guid')
            guid = guid_elem.text if guid_elem is not None else ""
            
            pub_date_elem = item.find('pubDate')
            pub_date = pub_date_elem.text if pub_date_elem is not None else ""
            
            creator_elem = item.find('.//{http://purl.org/dc/elements/1.1/}creator')
            creator = creator_elem.text if creator_elem is not None else ""
            
            description_elem = item.find('description')
            description = description_elem.text if description_elem is not None else ""
            if description and description.startswith('<![CDATA[') and description.endswith(']]>'):
                description = description[9:-3].strip()
            
            items.append({
                "title": title,
                "link": link,
                "guid": guid,
                "pubDate": pub_date,
                "creator": creator,
                "description": description
            })
        
        return items
    except ET.ParseError as e:
        logger.error(f"Error parsing RSS feed: {e}")
        return []

def extract_article_content(url):
    """Extract article content using Jina API."""
    jina_url = f"{JINA_API_BASE}{url}"
    try:
        response = requests.get(jina_url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Error extracting article from {url}: {e}")
        return None

def save_article(item, content):
    """Save article metadata and content."""
    # Create directory if it doesn't exist
    if not os.path.exists(ARTICLES_DIRECTORY):
        os.makedirs(ARTICLES_DIRECTORY)
    
    # Create a filename based on the GUID
    filename = f"{ARTICLES_DIRECTORY}/{item['guid'].split('=')[-1]}.json"
    
    # Combine item metadata with content
    article_data = {
        "metadata": item,
        "content": content,
        "collected_at": datetime.now().isoformat(),
        "publish_date": item.get('pubDate', '')
    }
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(article_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved article: {item['title']} to {filename}")

def main():
    """Main function to collect data from Hacker News RSS feed."""
    logger.info("Starting Hacker News data collection")
    
    # Ensure articles directory exists
    if not os.path.exists(ARTICLES_DIRECTORY):
        os.makedirs(ARTICLES_DIRECTORY)
        logger.info(f"Created {ARTICLES_DIRECTORY} directory")
    
    # Load processed items
    processed_items = load_processed_items()
    
    try:
        # Fetch RSS feed
        logger.info(f"Fetching RSS feed from {RSS_URL}")
        response = requests.get(RSS_URL, timeout=10)
        response.raise_for_status()
        
        # Parse RSS feed
        items = parse_rss_feed(response.text)
        logger.info(f"Found {len(items)} items in RSS feed")
        
        # Process new items
        new_items_count = 0
        for item in items:
            guid = item.get('guid', '')
            if not guid:
                continue

            # Check if article already exists on disk
            article_filename = f"{ARTICLES_DIRECTORY}/{guid.split('=')[-1]}.json"
            if os.path.exists(article_filename):
                logger.debug(f"Article already exists on disk: {article_filename}")
                # Ensure it's in processed_items
                if guid not in processed_items:
                    processed_items[guid] = {
                        "title": item['title'],
                        "processed_at": datetime.now().isoformat()
                    }
                continue
                
            # Skip if already processed according to our tracking
            if guid in processed_items:
                logger.debug(f"Skipping already processed item: {item.get('title', 'Unknown title')}")
                continue
                
            # Extract article content if there's a link
            link = item.get('link', '')
            if link:
                logger.info(f"Processing new article: {item['title']}")
                content = extract_article_content(link)
                if content:
                    save_article(item, content)
                    # Mark as processed
                    processed_items[guid] = {
                        "title": item['title'],
                        "processed_at": datetime.now().isoformat()
                    }
                    new_items_count += 1
                    # Small delay to be nice to the API
                    time.sleep(1)
            else:
                logger.warning(f"Item has no link: {item['title']}")
        
        # Save updated processed items
        save_processed_items(processed_items)
        logger.info(f"Processed {new_items_count} new articles")
        
    except requests.RequestException as e:
        logger.error(f"Error fetching RSS feed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    
    logger.info("Data collection completed")

if __name__ == "__main__":
    main()
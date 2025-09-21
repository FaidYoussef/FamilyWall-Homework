
from groq import Groq
import time
import random
import requests
import os
from dotenv import load_dotenv
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("FSQ_API_KEY")

# Foursquare API configuration
url = "https://places-api.foursquare.com/places/search"
headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "X-Places-Api-Version": "2025-06-17"
}

sectors_to_collect = [
    "restaurant",
    "retail store", 
    "law firm",
    "beauty salon",
    "gym",
    "nonprofit organization",
    "medical clinic",
    "nightclub",
    "café / bakery",
    "hotel / motel",
    "real estate agency",
    "construction / home services",
    "cleaning service",
    "veterinary clinic",
    "dentist",
    "physiotherapy clinic",
    "entertainment venue (cinema, bowling, etc.)",
    "transportation / taxi service",
    "accounting firm",
    "insurance broker",
    "financial advisory service",
    "IT services / software consultancy",
    "computer / phone repair shop",
    "coworking space",
    "education / tutoring center",
    "language school",
    "training institute"
]


params = {
    "query": "restaurant",
    "near": "New York, USA",
    "limit": 10
}

def scrape_business_description(website_url):
    """
    Scrapes the business description from a given website URL.

    Args:
        website_url (str): The URL of the business website.

    Returns:
        str: The scraped business description, or a message indicating it could not be found.
    """
    if not website_url:
        return "No website provided."

    try:
        # Set a user-agent to mimic a browser and improve success rate
        scrape_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(website_url, headers=scrape_headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, 'lxml')

        # Attempt to find a meta description tag first, as it's often a good summary
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description and meta_description.get('content'):
            return meta_description.get('content').strip()

        about_page_url = None
        # 2. Search for an "About Us" link
        possible_link_texts = ['about us', 'about', 'our story', 'company']
        for link in soup.find_all('a', href=True):
            if any(keyword in link.text.lower() for keyword in possible_link_texts):
                # We found a promising link!
                # urljoin handles both relative (/about) and absolute links
                about_page_url = urljoin(website_url, link['href'])
                print(f"Found potential 'About Us' page: {about_page_url}")
                break

        # 3. If we found an about page, scrape it instead
        if about_page_url:
            response = requests.get(about_page_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'lxml') # Overwrite soup with the new page's content

        # If no meta description, fall back to scraping paragraph text
        paragraphs = soup.find_all('p')
        # Join the text from all paragraphs and take a sizable portion
        full_text = ' '.join([p.get_text() for p in paragraphs])
        
        # A simple heuristic to get a meaningful chunk of text
        # You might want to refine this based on your needs
        return full_text[:1000].strip() if full_text else "No description found on the website."

    except requests.exceptions.RequestException as e:
        return f"Could not access website: {e}"
    except Exception as e:
        return f"An error occurred during scraping: {e}"

cities = ["New York, USA", "Los Angeles, USA"]



def collect_businesses_data():
    """
    Collect business data for all sectors and cities, storing everything in a single JSON file.
    """
    all_businesses = []
    collection_stats = {
        "total_collected": 0,
        "by_sector": {},
        "by_city": {},
        "errors": []
    }
    
    # Create data directory if it doesn't exist
    os.makedirs("business_data", exist_ok=True)
    
    for city in cities:
        print(f"\n{'='*50}")
        print(f"COLLECTING DATA FOR {city.upper()}")
        print(f"{'='*50}")
        
        collection_stats["by_city"][city] = 0
        
        for sector in sectors_to_collect:
            print(f"\nCollecting {sector} businesses in {city}...")
            
            # Initialize sector stats if not exists
            if sector not in collection_stats["by_sector"]:
                collection_stats["by_sector"][sector] = {"total": 0, "by_city": {}}
            collection_stats["by_sector"][sector]["by_city"][city] = 0
            
            try:
                params = {
                    "query": sector,
                    "near": city,
                    "limit": 50
                }
                
                # Make API call
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    places = data.get("results", [])
                    
                    print(f"Found {len(places)} {sector} businesses in {city}")
                    
                    for i, place in enumerate(places):
                        try:
                            website = place.get("website")
                            
                            # Scrape the business description from the website
                            description = scrape_business_description(website)
                            
                            # Create business entry
                            business_entry = {
                                "fsq_place_id": place.get("fsq_place_id"),
                                "name": place.get("name"),
                                "sector": sector,
                                "city": city,
                                "website": website,
                                "scraped_description": description,
                                "address": place.get("location", {}).get("formatted_address"),
                                "categories": place.get("categories", []),
                                "raw_foursquare_data": place
                            }
                            
                            all_businesses.append(business_entry)
                            
                            # Update stats
                            collection_stats["total_collected"] += 1
                            collection_stats["by_city"][city] += 1
                            collection_stats["by_sector"][sector]["total"] += 1
                            collection_stats["by_sector"][sector]["by_city"][city] += 1
                            
                            print(f"  [{i+1}/{len(places)}] {place.get('name')} - Description: {description[:100]}...")
                            
                            # Add small delay to be respectful to websites
                            time.sleep(random.uniform(0.5, 1.5))
                            
                        except Exception as e:
                            error_msg = f"Error processing business {place.get('name', 'Unknown')} in {sector}, {city}: {str(e)}"
                            print(f"  ERROR: {error_msg}")
                            collection_stats["errors"].append(error_msg)
                            continue
                    
                    # Add delay between sectors to respect rate limits
                    time.sleep(2)
                    
                else:
                    error_msg = f"API Error for {sector} in {city}: {response.status_code} - {response.text}"
                    print(f"ERROR: {error_msg}")
                    collection_stats["errors"].append(error_msg)
                    
            except Exception as e:
                error_msg = f"Exception while collecting {sector} in {city}: {str(e)}"
                print(f"ERROR: {error_msg}")
                collection_stats["errors"].append(error_msg)
                continue
    
    # Save all collected data to JSON file
    output_data = {
        "collection_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_businesses": len(all_businesses),
            "sectors_collected": sectors_to_collect,
            "cities_collected": cities,
            "stats": collection_stats
        },
        "businesses": all_businesses
    }
    
    output_file = "data/all_businesses_data.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total businesses collected: {collection_stats['total_collected']}")
    print(f"Data saved to: {output_file}")
    print(f"Total errors encountered: {len(collection_stats['errors'])}")
    
    # Print stats by city
    print(f"\nBusinesses by city:")
    for city, count in collection_stats["by_city"].items():
        print(f"  {city}: {count}")
    
    # Print stats by sector (top 10)
    print(f"\nTop sectors collected:")
    sector_totals = [(sector, data["total"]) for sector, data in collection_stats["by_sector"].items()]
    sector_totals.sort(key=lambda x: x[1], reverse=True)
    for sector, count in sector_totals[:10]:
        print(f"  {sector}: {count}")
    
    if collection_stats["errors"]:
        print(f"\nFirst 5 errors:")
        for error in collection_stats["errors"][:5]:
            print(f"  - {error}")
    
    return output_data


# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def is_meaningful_description(description: str) -> bool:
    """
    Uses gemma2-9b-it to decide if a scraped description is meaningful.
    Returns True if meaningful, False otherwise.
    """
    if not description:
        return False
    
    prompt = f"""
    You are a strict data cleaning assistant.
    Given a business description, decide if it is meaningful.
    
    Meaningful = provides actual information about the business (services, products, mission, etc.)
    Not meaningful = generic text like 'Welcome to our website', 'Home page', 'Coming soon', 'Best in town', etc.

    Answer only with "YES" if meaningful, or "NO" if not.

    Description: {description}
    """
    
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    answer = response.choices[0].message.content.strip().upper()
    return answer.startswith("YES")

import os
import json
import time
from groq import Groq
from typing import Dict, List, Optional
import re
from pydantic import BaseModel, Field
from datetime import datetime

class BusinessInfo(BaseModel):
    """Structured business information"""
    fsq_place_id: str
    name: str
    scraped_description: str
    normalized_description: Optional[str] = None
    sector: str
    website: Optional[str] = None
    city: str
    categories: Optional[List[str]] = None

class DomainSuggestion(BaseModel):
    """Domain suggestion with metadata"""
    domain: str = Field(..., description="The suggested domain name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Brief explanation for the suggestion")

class ProcessingResult(BaseModel):
    """Result of processing a single business"""
    business: BusinessInfo
    domain_suggestions: List[DomainSuggestion]
    status: str = "success"
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

class ProcessingMetadata(BaseModel):
    """Metadata about the processing run"""
    timestamp: str
    total_processed: int
    successful: int
    errors: int
    total_processing_time: float

class ProcessingOutput(BaseModel):
    """Complete output structure"""
    processing_metadata: ProcessingMetadata
    results: List[ProcessingResult]

class DomainNameGenerator:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.model = "gemma2-9b-it"
        
    def normalize_description(self, business: BusinessInfo) -> str:
        """
        Step 1: Normalize the scraped description into a clean, concise format
        Uses all available business information for context
        """
        # Build comprehensive business context from all non-null fields
        business_context = []
        
        # Core information
        business_context.append(f"Name: {business.name}")
        business_context.append(f"Sector: {business.sector}")
        
        # Location information
        if business.city:
            business_context.append(f"Location: {business.city}")
            
        # Categories from Foursquare
        if business.categories:
            categories_str = ", ".join(business.categories) if isinstance(business.categories, list) else str(business.categories)
            business_context.append(f"Categories: {categories_str}")
        
        # Website for additional context clues
        if business.website:
            business_context.append(f"Website: {business.website}")
        
        context_str = "\n".join(business_context)
        
        prompt = f"""You are a business description normalizer. Your task is to convert scraped website text into a clean, concise business description using all available business information.

RULES:
- Create a 1-2 sentence description (max 150 characters)
- Focus on: services/products, target audience, unique value proposition
- Remove: marketing fluff, generic phrases, website navigation text
- Use professional, clear language
- Use ALL available business information (name, sector, location, categories) to create the best description
- If original description is poor/generic, infer services from business name, sector, and categories

BUSINESS INFORMATION:
{context_str}

Original Scraped Description: {business.scraped_description}

Provide ONLY the normalized description, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            normalized = response.choices[0].message.content.strip()
            # Clean up any quotes or extra formatting
            normalized = re.sub(r'^["\']*|["\']*$', '', normalized)
            return normalized
            
        except Exception as e:
            print(f"Error normalizing description for {business.name}: {e}")
            return f"{business.name} - {business.sector} business providing professional services"
    
    def generate_domain_suggestions(self, business: BusinessInfo) -> List[DomainSuggestion]:
        """
        Step 2: Generate domain name suggestions based on normalized description
        """
        city = business.city.split(',')[0]  # Get city without country
        
        prompt = f"""You are an expert domain name generator. Generate creative, memorable domain names for this business.

BUSINESS INFO:
Name: {business.name}
Sector: {business.sector}
Location: {city}
Description: {business.normalized_description}

DOMAIN REQUIREMENTS:
- Generate exactly 3 domain suggestions
- Use .com extension only
- 6-15 characters (excluding .com)
- Memorable, brandable, easy to spell
- Relevant to business but not too literal
- Avoid hyphens, numbers, or complex words

For each domain, provide:
1. Domain name
2. Confidence score (0.0-1.0)
3. Brief reasoning (one sentence)

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
1. domainname.com | 0.85 | Brief reasoning here
2. anotherdomain.com | 0.78 | Brief reasoning here  
3. thirddomain.com | 0.92 | Brief reasoning here"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            return self._parse_domain_response(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error generating domains for {business.name}: {e}")
            return []
    
    def _parse_domain_response(self, response: str) -> List[DomainSuggestion]:
        """Parse the LLM response into structured domain suggestions"""
        suggestions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            if '|' in line:
                try:
                    # Parse format: "1. domain.com | 0.85 | reasoning"
                    parts = line.split('|')
                    if len(parts) >= 3:
                        domain_part = parts[0].strip()
                        # Extract domain name (remove numbering and clean up)
                        domain_match = re.search(r'([a-zA-Z0-9-]+\.com)', domain_part)
                        if domain_match:
                            domain = domain_match.group(1)
                        else:
                            # Fallback parsing
                            domain = parts[0].strip().split(' ')[-1]
                            if not domain.endswith('.com'):
                                domain += '.com'
                        
                        confidence = float(parts[1].strip())
                        reasoning = parts[2].strip()
                        
                        suggestions.append(DomainSuggestion(
                            domain=domain,
                            confidence=confidence,
                            reasoning=reasoning
                        ))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}, Error: {e}")
                    continue
        
        return suggestions
    
    def process_business(self, business_data: Dict) -> ProcessingResult:
        """
        Complete pipeline: normalize description and generate domains
        """
        start_time = time.time()
        
        try:
            # Convert dict to BusinessInfo model for validation
            business = BusinessInfo(
                fsq_place_id=business_data.get('fsq_place_id', ''),
                name=business_data.get('name', ''),
                scraped_description=business_data.get('scraped_description', ''),
                sector=business_data.get('sector', ''),
                website=business_data.get('website'),
                city=business_data.get('city', ''),
                categories=business_data.get('categories')
            )
            
            # print all business info
            # print(f"Processing business: {business.model_dump_json(indent=2)}")
            # Step 1: Normalize description
            normalized_desc = self.normalize_description(business)
            
            # Update business with normalized description
            business.normalized_description = normalized_desc
            
            # print(f"Processing business: {business.model_dump_json(indent=2)}")
            # Step 2: Generate domain suggestions
            domain_suggestions = self.generate_domain_suggestions(business)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                business=business,
                domain_suggestions=domain_suggestions,
                status="success",
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error processing business {business_data.get('name', 'Unknown')}: {e}")
            
            # Create minimal business info for error case
            business = BusinessInfo(
                fsq_place_id=business_data.get('fsq_place_id', ''),
                name=business_data.get('name', ''),
                scraped_description=business_data.get('scraped_description', ''),
                sector=business_data.get('sector', ''),
                website=business_data.get('website'),
                city=business_data.get('city', ''),
                categories=business_data.get('categories')
            )
            
            return ProcessingResult(
                business=business,
                domain_suggestions=[],
                status="error",
                processing_time=processing_time,
                error_message=str(e)
            )


# Safety filter - only for API deployment
class SafetyFilter:
    """Safety filter for API endpoint - separate from data generation pipeline"""
    
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.model = "gemma2-9b-it"
    
    def is_safe_request(self, business_description: str) -> tuple[bool, str]:
        """
        Check if a business description is appropriate for domain generation
        Returns (is_safe, message)
        """
        safety_prompt = f"""You are a content safety filter. Determine if this business description contains inappropriate content that should be blocked.

BLOCK if description contains:
- Adult/sexual content
- Violence or weapons
- Illegal activities
- Hate speech or discrimination
- Gambling (except licensed establishments)
- Harmful substances

ALLOW legitimate businesses including:
- Bars, nightclubs (non-adult entertainment)
- Medical/health services
- Legal services
- Normal retail/services

Answer only "BLOCK" or "ALLOW"

Business Description: {business_description}"""

        try:
            response = self.client.chat.completions.create(
                model="meta-llama/llama-guard-4-12b",
                messages=[{"role": "user", "content": safety_prompt}],
                temperature=0
            )
            
            result = response.choices[0].message.content.strip().upper()
            is_safe = result == "ALLOW"
            message = "Request approved" if is_safe else "Request contains inappropriate content"
            
            return is_safe, message
            
        except Exception as e:
            print(f"Error in safety check: {e}")
            return True, "Safety check unavailable"  



import json
import os
import time
from datetime import datetime
from typing import Set, List, Dict, Any

def load_existing_results(temp_file: str = "temp_data.json") -> tuple[List[Any], Set[str]]:
    """
    Load existing results from temp file and return processed business IDs
    """
    processed_results = []
    processed_ids = set()
    
    if os.path.exists(temp_file):
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                temp_data = json.load(f)
            
            results_data = temp_data.get('results', [])
            
            # Process each result from the temp file
            for result_data in results_data:
                if isinstance(result_data, dict):
                    processed_results.append(result_data)
                    
                    # Extract business ID from the nested structure
                    business_id = None
                    
                    # Check if this is a ProcessingResult structure with nested business
                    if 'business' in result_data and isinstance(result_data['business'], dict):
                        business = result_data['business']
                        business_id = (business.get('fsq_place_id') or 
                                     business.get('id') or 
                                     business.get('business_id') or
                                     business.get('name'))
                    
                    # Check if this is an error result with direct business_id
                    elif 'business_id' in result_data:
                        business_id = result_data['business_id']
                    
                    # Fallback to other ID fields
                    elif 'id' in result_data:
                        business_id = result_data['id']
                    
                    if business_id:
                        processed_ids.add(str(business_id))
                else:
                    # Non-dict result (shouldn't happen but handle gracefully)
                    processed_results.append(result_data)
            
            print(f"Found {len(processed_results)} previously processed businesses")
            print(f"Unique processed IDs: {len(processed_ids)}")
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Could not load temp file: {e}")
            processed_results = []
            processed_ids = set()
    else:
        print("No temp file found, starting fresh")
    
    return processed_results, processed_ids

def save_progress(processed_results: List[Any], 
                 total_businesses: int, 
                 start_time: float,
                 temp_file: str = "temp_data.json"):
    """
    Save current progress to temp file
    """
    successful = 0
    errors = 0
    
    for result in processed_results:
        if hasattr(result, 'status'):
            # ProcessingResult object
            if result.status == 'success':
                successful += 1
            elif result.status == 'error':
                errors += 1
        elif isinstance(result, dict):
            # Error dict
            if result.get('status') == 'success':
                successful += 1
            elif result.get('status') == 'error':
                errors += 1
    
    temp_metadata = ProcessingMetadata(
        timestamp=datetime.now().isoformat(),
        total_processed=len(processed_results),
        successful=successful,
        errors=errors,
        total_processing_time=time.time() - start_time
    )
    
    temp_output = ProcessingOutput(
        processing_metadata=temp_metadata,
        results=processed_results
    )
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(temp_output.model_dump_json(indent=2))
    
    print(f"Progress saved: {len(processed_results)}/{total_businesses} completed")

def process_collected_data(input_file: str, output_file: str) -> ProcessingOutput:
    """
    Process all collected business data through the domain generation pipeline
    with keyboard interrupt handling and resume capability
    """
    generator = DomainNameGenerator(os.getenv("GROQ_API_KEY"))
    temp_file = "temp_data.json"
    
    # Load collected data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    businesses = data.get('businesses', [])
    
    # Load existing results and get processed IDs
    processed_results, processed_ids = load_existing_results(temp_file)
    
    # Filter out already processed businesses
    businesses_to_process = []
    skipped_count = 0
    
    for business in businesses:
        # Check multiple possible ID fields to match against processed_ids
        business_id = (business.get('fsq_place_id') or 
                      business.get('id') or 
                      business.get('business_id') or
                      business.get('name'))
        
        if business_id and str(business_id) in processed_ids:
            skipped_count += 1
            continue
        businesses_to_process.append(business)
    
    print(f"Total businesses: {len(businesses)}")
    print(f"Already processed: {len(processed_results)} (skipping {skipped_count})")
    print(f"Remaining to process: {len(businesses_to_process)}")
    
    if not businesses_to_process:
        print("All businesses already processed!")
        # Still create final output with existing results
        total_processing_time = 0
        if processed_results:
            # Try to get time from existing metadata
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    temp_data = json.load(f)
                total_processing_time = temp_data.get('processing_metadata', {}).get('total_processing_time', 0)
            except:
                total_processing_time = 0
    else:
        total_start_time = time.time()
        
        try:
            for i, business in enumerate(businesses_to_process):
                business_name = business.get('name', 'Unknown')
                current_total = len(processed_results) + i + 1
                total_businesses = len(businesses)
                
                print(f"Processing [{current_total}/{total_businesses}]: {business_name}")
                
                try:
                    result = generator.process_business(business)
                    processed_results.append(result)
                    
                    # Add small delay to respect API limits
                    time.sleep(1)
                    
                    # Save progress every 10 businesses or if this is the last one
                    if (i + 1) % 10 == 0 or (i + 1) == len(businesses_to_process):
                        save_progress(processed_results, total_businesses, total_start_time, temp_file)
                
                except Exception as e:
                    print(f"Error processing {business_name}: {e}")
                    # Create error result as dict (not ProcessingResult)
                    business_id = (business.get('fsq_place_id') or 
                                  business.get('id') or 
                                  business.get('business_id') or
                                  business.get('name'))
                    
                    error_result = {
                        'business_id': business_id,
                        'status': 'error',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'business': business  # Include the business data for context
                    }
                    processed_results.append(error_result)
        
        except KeyboardInterrupt:
            print("\n⚠️  Keyboard interrupt received!")
            print(f"Processed {len(processed_results)} businesses so far")
            print("Progress saved to temp_data.json")
            print("Run the function again to resume from where you left off")
            
            # Save final progress before exiting
            save_progress(processed_results, len(businesses), total_start_time, temp_file)
            
            # Create partial output - count successes and errors properly
            successful = 0
            errors = 0
            for result in processed_results:
                if hasattr(result, 'status'):
                    if result.status == 'success':
                        successful += 1
                    elif result.status == 'error':
                        errors += 1
                elif isinstance(result, dict):
                    if result.get('status') == 'success':
                        successful += 1
                    elif result.get('status') == 'error':
                        errors += 1
            
            partial_metadata = ProcessingMetadata(
                timestamp=datetime.now().isoformat(),
                total_processed=len(processed_results),
                successful=successful,
                errors=errors,
                total_processing_time=time.time() - total_start_time
            )
            
            return ProcessingOutput(
                processing_metadata=partial_metadata,
                results=processed_results
            )
        
        total_processing_time = time.time() - total_start_time
    
    # Create final output - properly count successes and errors
    successful = 0
    errors = 0
    for result in processed_results:
        if hasattr(result, 'status'):
            # ProcessingResult object
            if result.status == 'success':
                successful += 1
            elif result.status == 'error':
                errors += 1
        elif isinstance(result, dict):
            # Error dict
            if result.get('status') == 'success':
                successful += 1
            elif result.get('status') == 'error':
                errors += 1
    
    metadata = ProcessingMetadata(
        timestamp=datetime.now().isoformat(),
        total_processed=len(processed_results),
        successful=successful,
        errors=errors,
        total_processing_time=total_processing_time
    )
    
    output = ProcessingOutput(
        processing_metadata=metadata,
        results=processed_results
    )
    
    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output.model_dump_json(indent=2))
    
    print(f"\n✅ Processing complete! Results saved to: {output_file}")
    print(f"Success: {successful}")
    print(f"Errors: {errors}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    
    # Clean up temp file after successful completion
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print("Temporary file cleaned up")
    
    return output


def analyze_fallback_patterns(temp_file: str):
    """
    Analyze the temp file to see how many businesses have fallback patterns
    """
    try:
        with open(temp_file, 'r', encoding='utf-8') as f:
            temp_data = json.load(f)
    except FileNotFoundError:
        print(f"Temp file {temp_file} not found")
        return
    
    results = temp_data.get('results', [])
    fallback_count = 0
    total_count = len(results)
    
    print(f"Analyzing {total_count} results in {temp_file}...")
    print("\nBusinesses with fallback descriptions:")
    print("-" * 50)
    
    for result in results:
        business = result.get('business', {})
        normalized_desc = business.get('normalized_description', '')
        business_name = business.get('name', '')
        business_sector = business.get('sector', '')
        
        expected_fallback = f"{business_name} - {business_sector} business providing professional services"
        
        if normalized_desc == expected_fallback:
            fallback_count += 1
            print(f"{fallback_count}. {business_name} ({business_sector})")
    
    print(f"\nSUMMARY:")
    print(f"Total results: {total_count}")
    print(f"Fallback patterns: {fallback_count}")
    print(f"Proper descriptions: {total_count - fallback_count}")
    print(f"Fallback percentage: {(fallback_count/total_count)*100:.1f}%")

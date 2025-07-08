import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
# from IPython.display import HTML
import re
import spacy
from spacy.pipeline import EntityRuler
from openai import OpenAI

import json

# ─── Load ENV ───────────────────────────────────────
load_dotenv()
serpapi_key = os.getenv("SERPAPI_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
print("Loaded key:", serpapi_key)

# ─── OpenAI Client ──────────────────────────────────
client = OpenAI(api_key=openai_api_key)

# ─── NLP Setup ──────────────────────────────────────
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

STRICT_PRICE_RE = re.compile(r"""(?x)
    ^                                # start
    (?:\$|₹|€|£|USD|INR|EUR|GBP)\s*  # currency symbol or code
    \d{1,3}(?:,\d{3})*(?:\.\d+)?     # amount
    $                                # end
""")

def parse_price_string(s: str) -> float | None:
    if not s or not STRICT_PRICE_RE.match(s.strip()):
        return None
    num = re.sub(r"[^\d.]", "", s)
    try:
        return float(num)
    except:
        return None

def is_probable_price_key(key: str, value) -> bool:
    if value is None:
        return False
    key_lower = key.lower()
    if any(word in key_lower for word in ["price", "amount", "cost", "rate", "pay", "mrp"]):
        pass
    else:
        return False
    if isinstance(value, (int, float)) and value > 0:
        return True
    if isinstance(value, str):
        if re.search(r'[\$₹€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?', value):
            return True
        if re.fullmatch(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', value):
            return True
        if re.search(r'per\s?(month|mo|week|year)', value, re.I):
            return True
        if re.search(r'EMI|installment', value, re.I):
            return True
    return False

def extract_price(item: dict) -> float | None:
    if "price" in item:
        val = parse_price_string(str(item["price"]))
        if val is not None:
            return val
    if "extracted_price" in item:
        ep = item["extracted_price"]
        if isinstance(ep, (int, float)):
            return float(ep)
    rich = item.get("rich_snippet", {}).get("bottom", {}).get("detected_extensions", {})
    if "price" in rich:
        return float(rich["price"])
    for key, val in item.items():
        if is_probable_price_key(key, val):
            try:
                v = re.sub(r"[^\d.]", "", str(val)).replace(",", "")
                return float(v)
            except:
                continue
    text = " ".join(filter(None, (item.get("title",""), item.get("description",""), item.get("snippet",""))))
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            num = re.sub(r"[^\d.,]", "", ent.text).replace(",", "")
            try:
                return float(num)
            except:
                continue
    m = re.search(r'[\$₹€£]\s?(\d{1,3}(?:,\d{3})*(?:\.\d+))', text or "")
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except:
            pass
    return None

def parse_product_result_block(data):
    results = []
    seen_links = set()

    product_result = data.get("product_result", {})
    pricing = product_result.get("pricing", [])
    title = product_result.get("title")

    for item in pricing:
        link = item.get("link", "")  # Ensure every item has a link key
        if link not in seen_links:
            results.append({
                "title": title,
                "description": item.get("description"),
                "price": item.get("price"),
                "extracted_price": item.get("extracted_price"),
                "link": link,
                "source": item.get("name"),
                "thumbnail": item.get("thumbnail"),
                "origin": "product_result"
            })
            seen_links.add(link)

    for item in data.get("shopping_results", []):
        link = item.get("link", "")  # Ensure every item has a link key
        if link not in seen_links:
            results.append({
                "title": item.get("title"),
                "description": None,
                "price": item.get("price"),
                "extracted_price": item.get("extracted_price"),
                "link": link,
                "source": item.get("source"),
                "thumbnail": item.get("thumbnail"),
                "origin": "shopping_results"
            })
            seen_links.add(link)

    for item in data.get("immersive_products", []):
        link = item.get("link", "")  # Ensure every item has a link key
        if link not in seen_links:
            results.append({
                "title": item.get("title"),
                "description": None,
                "price": item.get("price"),
                "extracted_price": item.get("extracted_price"),
                "link": link,
                "source": item.get("source"),
                "thumbnail": item.get("thumbnail"),
                "origin": "shopping_results"
            })
            seen_links.add(link)

    for item in data.get("organic_results", []):
        link = item.get("link", "")  # Ensure every item has a link key
        price = extract_price(item)
        if link not in seen_links:
            results.append({
                "title": item.get("title"),
                "description": item.get("snippet"),
                "price": price,
                "extracted_price": price,
                "link": link,
                "source": item.get("displayed_link"),
                "thumbnail": None,
                "origin": "organic_results"
            })
            seen_links.add(link)

    # Filter out non-shopping websites, comparison websites, and blogs
   
    exclusion_keywords = ["compare", "review", "blog", "Leak", "Innovation", "Consumer"]
    results = [
        item for item in results
        if (not any(keyword in item["link"] for keyword in exclusion_keywords) and
            not any(keyword.lower() in item["title"].lower() for keyword in exclusion_keywords) and
            not any(keyword.lower() in item["source"].lower() for keyword in exclusion_keywords))
    ]
    results = [
        item for item in results
        if (not any(keyword in item["link"] for keyword in exclusion_keywords))
    ]

    return results

def normalize_with_llm(raw_items: list[dict]) -> list[dict]:
    prompt = (
        "You are a JSON normalizer. "
        "Given a list of raw product-offer dicts, return only a JSON array "
        "where each item has keys: title, source, link, plan, monthly_price, total_price, extracted_price, currency. "
        "Compute EMI totals, drop unrelated fields."
    )

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Raw data:\n```json\n{json.dumps(raw_items)}\n```"}
        ],
        temperature=0.0,
        max_tokens=1500
    )

    text = completion.choices[0].message.content
    json_text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(json_text)

def get_product_links(query: str, country: str, serpapi_key: str, use_llm: bool=False) -> list[dict]:
    COUNTRY_TO_DOMAIN = {
        "US": "google.com",
        "IN": "google.co.in",
        "UK": "google.co.uk",
    }
    params = {
        "q": query,
        "engine": "google",
        "location": get_location_name(country),
        "hl": "en",
        "gl": country.lower(),
        "google_domain": COUNTRY_TO_DOMAIN.get(country, "google.com"),
        "api_key": serpapi_key,
    }
    print(params)
    search = GoogleSearch(params)
    results = search.get_dict()
    
    offers = parse_product_result_block(results)
    return normalize_with_llm(offers) if use_llm else offers

COUNTRY_CODE_TO_NAME = {
    "US": "United States",
    "IN": "India",
    "UK": "United Kingdom",
    "DE": "Germany",
    "FR": "France",
    "CA": "Canada",
    "AU": "Australia",
    "SG": "Singapore",
    "AE": "United Arab Emirates",
    "JP": "Japan",
    "CN": "China"
}

def get_location_name(country_code: str) -> str:
    return COUNTRY_CODE_TO_NAME.get(country_code.upper(), country_code)

# def show_offers(results):
#     results = [item for item in results if item["price"] is not None]
#     html = "<table><tr><th>Image</th><th>Source</th><th>Title</th><th>Price</th></tr>"
#     for item in results:
#         html += f"<tr><td><img src='{item['thumbnail']}' width='80'></td>"
#         html += f"<td>{item['source']}</td>"
#         html += f"<td><a href='{item['link']}' target='_blank'>{item['title']}</a></td>"
#         html += f"<td>{item['price']}</td></tr>"
#     html += "</table>"
#     return HTML(html)

# Example call
# links = get_product_links(
#     "iPhone 16 Pro",
#     "IN",
#     serpapi_key,
#     use_llm=False
# )
# show_offers(links)

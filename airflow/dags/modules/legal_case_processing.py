from pinecone import Pinecone
import requests
from bs4 import BeautifulSoup
import openai
import os
from dotenv import load_dotenv
import json
import re
import time

load_dotenv()

# Set up OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-case-index")

def get_case_urls(base_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    
    response = requests.get(base_url, headers=headers)
    urls = []
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        article_tags = soup.find_all('article')
        
        for article in article_tags:
            a_tag = article.find('a')
            if a_tag and 'href' in a_tag.attrs:
                url = 'https://www.courtlistener.com' + a_tag['href']
                urls.append(url)
    
    return urls

def process_single_case(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch {url}. Status code: {response.status_code}")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        plaintext_tags = soup.find_all(class_="plaintext")
        document_content = "\n\n".join([tag.text for tag in plaintext_tags])
        
        # Get summary from GPT-4
        prompt = f"""
        The following is a legal case text. Please summarize it into the following categories:
        1. Docket Number: [content]
        2. Facts: [content]
        3. Issues: [content]
        4. Reasoning: [content]
        5. Decision: [content]

        Case Text:
        {document_content}
        """

        gpt_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Please summarize the legal case into the specified categories."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.5
        )

        summary = gpt_response.choices[0].message.content

        # Parse the GPT summary
        regex_patterns = {
            "Docket Number": r"1\. Docket Number: (.*?)(?=\n2\. Facts:|$)",
            "Facts": r"2\. Facts: (.*?)(?=\n3\. Issues:|$)",
            "Issues": r"3\. Issues: (.*?)(?=\n4\. Reasoning:|$)",
            "Reasoning": r"4\. Reasoning: (.*?)(?=\n5\. Decision:|$)",
            "Decision": r"5\. Decision: (.*?)(?=\n|$)"
        }

        parsed_summary = {}
        for key, pattern in regex_patterns.items():
            matches = re.findall(pattern, summary, re.DOTALL)
            parsed_summary[key] = matches[0].strip() if matches else ""

        # Create embedding
        embedding_response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=document_content
        )
        embedding = embedding_response.data[0].embedding

        # Prepare metadata
        metadata = {
            "docket_number": parsed_summary["Docket Number"],
            "facts": parsed_summary["Facts"],
            "issues": parsed_summary["Issues"],
            "reasoning": parsed_summary["Reasoning"],
            "decision": parsed_summary["Decision"]
        }

        # Upload to Pinecone
        index.upsert(
            vectors=[
                {
                    "id": f"case_{parsed_summary['Docket Number'].replace(' ', '_')}",
                    "values": embedding,
                    "metadata": metadata
                }
            ]
        )
        
        print(f"\nProcessed and uploaded case: {parsed_summary['Docket Number']}")
        
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")

def main():
    base_url = "https://www.courtlistener.com/?type=o&q=&type=o&order_by=dateFiled%20desc"
    case_urls = get_case_urls(base_url)
    
    print(f"Found {len(case_urls)} cases to process")
    
    for i, url in enumerate(case_urls):
        print(f"\nProcessing case {i+1}/{len(case_urls)}")
        process_single_case(url)
        time.sleep(2)  # Add delay between requests to avoid rate limiting

if __name__ == "__main__":
    main()
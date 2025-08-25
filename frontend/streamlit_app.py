import streamlit as st
import requests
import json
import feedparser
from typing import Dict, Any
from bs4 import BeautifulSoup
from datetime import datetime
from pages.report_app import run_report_app
from pages.strategy_app import run_strategy_app
from pages.summary_app import run_summary_app

# Must be the first Streamlit command
st.set_page_config(
    page_title="Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = "http://localhost:8000"

def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None


def fetch_legal_news():
    """Fetch legal news from free sources with fallback content"""
    news_items = []
    
    # Try JD Supra Feed
    try:
        feed = feedparser.parse("https://www.jdsupra.com/rss/legalnews.aspx")
        if feed.entries:
            for entry in feed.entries[:3]:
                news_items.append({
                    "title": entry.title,
                    "summary": entry.get('summary', 'Click to read more...'),
                    "url": entry.link,
                    "source": "JD Supra",
                    "date": datetime.now().strftime("%Y-%m-%d")
                })
    except Exception as e:
        st.error(f"Error fetching JD Supra news: {e}")

    # Try ABA Journal Feed
    try:
        feed = feedparser.parse("https://www.abajournal.com/feed/index.rss")
        if feed.entries:
            for entry in feed.entries[:3]:
                news_items.append({
                    "title": entry.title,
                    "summary": entry.get('summary', 'Click to read more...'),
                    "url": entry.link,
                    "source": "ABA Journal",
                    "date": datetime.now().strftime("%Y-%m-%d")
                })
    except Exception as e:
        st.error(f"Error fetching ABA Journal news: {e}")

    # If no news could be fetched, provide fallback content
    if not news_items:
        news_items = [
            {
                "title": "Latest Legal Updates",
                "summary": "Check back later for the latest legal news and updates.",
                "url": "https://www.law.com",
                "source": "Legal News",
                "date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "title": "Supreme Court Updates",
                "summary": "Stay tuned for important Supreme Court decisions and analysis.",
                "url": "https://www.supremecourt.gov",
                "source": "Supreme Court",
                "date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "title": "Legal Industry Trends",
                "summary": "Keep updated with the latest trends in the legal industry.",
                "url": "https://www.law.com/legalnewswire",
                "source": "Legal Industry News",
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        ]

    return news_items

def display_news_sidebar():
    """Display legal news in the sidebar"""
    with st.sidebar:
        st.title("Legal News")
        news_items = fetch_legal_news()
        
        for item in news_items:
            with st.expander(f"📰 {item['title']}", expanded=False):
                st.write(f"**Source:** {item['source']}")
                st.write(f"**Date:** {item['date']}")
                st.write(item['summary'])
                st.markdown(f"[Read more]({item['url']})")

def login_sidebar():
    """Handle login/registration in the sidebar"""
    with st.sidebar:
        st.title("Legal Assistant Login")
        
        if not st.session_state.logged_in:
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                if st.button("Login", use_container_width=True):
                    if username and password:
                        response = requests.post(
                            f"{API_URL}/token",
                            data={"username": username, "password": password}
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.access_token = data["access_token"]
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
            
            with tab2:
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                if st.button("Register", use_container_width=True):
                    if new_username and new_password:
                        response = requests.post(
                            f"{API_URL}/register",
                            json={"username": new_username, "password": new_password}
                        )
                        if response.status_code == 200:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error("Registration failed")

def news_sidebar():
    """Display legal news in the sidebar"""
    with st.sidebar:
        st.title("Legal News")
        news_items = fetch_legal_news()
        for item in news_items:
            with st.expander(item["title"]):
                st.write(item["summary"])

def process_query(query: str) -> Dict[str, Any]:
    """Process user query and determine appropriate agent"""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        response = requests.post(
            f"{API_URL}/process",
            json={"query": query},
            headers=headers
        )
        if response.status_code == 200:
            # For demonstration, using simple keyword matching
            # In production, this would come from your backend
            if "strategy" in query.lower():
                return {
                    "status": "success",
                    "message": "I'll help you develop a strategy.",
                    "agent": "strategy"
                }
            elif "report" in query.lower():
                return {
                    "status": "success",
                    "message": "I'll help you generate a report.",
                    "agent": "report"
                }
            elif "summary" in query.lower():
                return {
                    "status": "success",
                    "message": "I'll help you create a summary.",
                    "agent": "summary"
                }
            else:
                return {
                    "status": "success",
                    "message": "I can help you with generating reports, developing strategies, or creating summaries. What would you like to do?",
                    "agent": None
                }
        else:
            return {
                "status": "error",
                "message": f"Server error: {response.status_code}",
                "agent": None
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "agent": None
        }

def display_news_sidebar():
    """Display legal news in the sidebar"""
    with st.sidebar:
        st.title("Legal News")
        news_items = fetch_legal_news()
        
        for item in news_items:
            with st.expander(f"📰 {item['title']}", expanded=False):
                st.write(f"**Source:** {item['source']}")
                st.write(f"**Date:** {item['date']}")
                st.write(item['summary'])
                st.markdown(f"[Read more]({item['url']})")

def handle_agent_response(agent_type: str):
    """Handle agent-specific UI within chat container"""
    if agent_type == "strategy":
        # Instead of using with st.chat_message here since we're already in a chat message
        option = st.radio(
            "Choose input method",
            ("Upload a file", "Input details manually"),
            horizontal=True,
            label_visibility="collapsed"
        )

        if option == "Upload a file":
            uploaded_file = st.file_uploader(
                "Please upload the case file (txt, pdf, docx, jpg, png)",
                type=["txt", "pdf", "docx", "jpg", "png"]
            )
            if uploaded_file is not None:
                files = {'file': (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                with st.spinner("Analyzing file..."):
                    response = requests.post(f"{API_URL}/process_file", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("strategy"):
                            st.write("### Generated Strategy")
                            st.write(data["strategy"])
                            if data.get("papers"):
                                st.write("### Relevant Research")
                                for title, summary, link in data["papers"]:
                                    with st.expander(f"📄 {title}"):
                                        st.write(summary)
                                        st.markdown(f"[View paper]({link})")
                        else:
                            st.write(data.get("message", "No similar cases found"))
                    
        elif option == "Input details manually":
            with st.form("case_details_form"):
                facts = st.text_area("Enter Facts", placeholder="Describe the key facts of your case...")
                issues = st.text_area("Enter Issues", placeholder="What are the main legal questions?")
                reasoning = st.text_area("Enter Reasoning", placeholder="Any relevant legal principles...")
                decision = st.text_area("Enter Decision", placeholder="Any prior decisions...")
                submitted = st.form_submit_button("Generate Strategy")
                
                if submitted:
                    if not facts.strip() or not issues.strip():
                        st.warning("Please provide at least the facts and legal issues.")
                    else:
                        with st.spinner("Generating strategy..."):
                            response = requests.post(
                                f"{API_URL}/process",
                                json={
                                    "query": "generate strategy",
                                    "context": {
                                        "facts": facts,
                                        "issues": issues,
                                        "reasoning": reasoning,
                                        "decision": decision
                                    }
                                }
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                if data.get("result"):
                                    st.write("### Generated Strategy")
                                    st.write(data["result"])
                                else:
                                    st.warning("No strategy was generated. Please try again with more details.")
                            else:
                                st.error("Error generating strategy. Please try again.")


    elif agent_type == "report":
        run_report_app()
    elif agent_type == "summary":
        run_summary_app()

def chat_interface():
    """Main chat interface"""
    st.title("Legal Assistant")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message.get("avatar", "👤")):
            st.write(message["content"])
            if message.get("agent_response"):
                handle_agent_response(message["agent_response"])
    
    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "avatar": "👤"
        })
        
        # Process query
        result = process_query(prompt)
        
        if result.get("status") == "success":
            response_content = result.get("message", "I'll help you with that.")
            agent_type = result.get("agent")
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_content,
                "avatar": "🤖",
                "agent_response": agent_type
            })
        else:
            # Add error message to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"I apologize, but I encountered an error: {result.get('message')}",
                "avatar": "🤖"
            })
        
        st.rerun()

def main():
    """Main application entry point"""
    init_session_state()
    
    if not st.session_state.logged_in:
        login_sidebar()
    else:
        news_sidebar()
        chat_interface()

if __name__ == "__main__":
    main()

import streamlit as st
import requests

API_URL = "http://localhost:8000"

def run_strategy_app():
    """Strategy Generation Agent Interface"""
    with st.chat_message("assistant", avatar="🤖"):
        st.write("I'll help you develop a legal strategy. How would you like to provide your case information?")
        
        option = st.radio(
            "Select your preferred method:",
            ("Upload a case document", "Enter case details manually"),
            label_visibility="collapsed"
        )

    if option == "Upload a case document":
        with st.chat_message("assistant", avatar="🤖"):
            st.write("Please upload your case document. I can analyze files in PDF, DOCX, TXT, or image formats.")
            uploaded_file = st.file_uploader(
                "Upload your case file",
                type=["txt", "pdf", "docx", "jpg", "png"],
                label_visibility="collapsed"
            )

        if uploaded_file is not None:
            with st.chat_message("user", avatar="👤"):
                st.write(f"Uploaded: {uploaded_file.name}")
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analyzing your document..."):
                    files = {'file': (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/process_file", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("strategy"):
                            st.write("### Recommended Legal Strategy")
                            st.write(data["strategy"])
                            
                            papers = data.get("papers", [])
                            if isinstance(papers, list) and papers:
                                st.write("### Relevant Legal Research")
                                for title, summary, link in papers:
                                    with st.expander(f"📄 {title}"):
                                        st.write(summary)
                                        st.markdown(f"[Read full paper]({link})")
                        else:
                            st.warning(data.get("message", "I couldn't find similar cases. Could you provide more details?"))
                    else:
                        st.error("I encountered an error while processing your document. Please try again.")
                
    elif option == "Enter case details manually":
        with st.chat_message("assistant", avatar="🤖"):
            st.write("Please provide the following details about your case:")
            
            with st.form("case_details_form"):
                facts = st.text_area("📝 Case Facts", 
                    placeholder="Describe the key facts of your case...")
                
                issues = st.text_area("⚖️ Legal Issues", 
                    placeholder="What are the main legal questions or challenges?")
                
                reasoning = st.text_area("🤔 Legal Reasoning", 
                    placeholder="Any relevant legal principles or arguments...")
                
                decision = st.text_area("📋 Prior Decisions", 
                    placeholder="Any relevant court decisions or precedents...")
                
                submit = st.form_submit_button("Generate Strategy")

        if submit:
            if not facts.strip() or not issues.strip():
                with st.chat_message("assistant", avatar="🤖"):
                    st.warning("Please provide at least the case facts and legal issues.")
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Analyzing your case..."):
                        # First, process the query to get the strategy agent
                        process_response = requests.post(
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
                        
                        # Debug prints
                        st.write("Process Response:", process_response.status_code)
                        st.write("Process Data:", process_response.json())
                        
                        if process_response.status_code == 200:
                            process_data = process_response.json()
                            
                            if process_data.get("result"):
                                st.write("### Generated Legal Strategy")
                                st.write(process_data["result"])
                            else:
                                st.warning("No strategy was generated. Please try again with different input.")
                        else:
                            st.error("Error generating strategy. Please try again.")
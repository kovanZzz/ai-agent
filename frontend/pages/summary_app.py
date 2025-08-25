import streamlit as st
import requests

API_URL = "http://localhost:8000"

def run_summary_app():
    """Run the Summary App within chat container"""
    # Remove title since we're in chat
    option = st.radio(
        "Choose how you'd like to provide the case information:",
        ("Upload a file", "Input details manually"),
        horizontal=True,
        label_visibility="collapsed"
    )

    if option == "Upload a file":
        uploaded_file = st.file_uploader(
            "Upload your case document (PDF, DOCX, TXT, or image)",
            type=["txt", "pdf", "docx", "jpg", "png"],
            help="I'll analyze the document and generate a summary for you."
        )
        
        if uploaded_file is not None:
            files = {'file': (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
            
            with st.spinner("Analyzing your document..."):
                response = requests.post(f"{API_URL}/upload_file", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("report"):
                        st.write("### Generated Summary")
                        # Using expander for better chat flow
                        with st.expander("View Complete Summary", expanded=True):
                            st.write(data["report"])
                    else:
                        st.warning(data.get("message", "I couldn't generate a summary from this document."))
                else:
                    st.error("I encountered an error while processing your document. Please try again.")
                
    elif option == "Input details manually":
        # Using a form for better organization
        with st.form("case_details_form"):
            st.write("Please provide the following case details:")
            
            facts = st.text_area(
                "📝 Facts",
                placeholder="Enter the key facts of the case...",
                height=100
            )
            
            issues = st.text_area(
                "⚖️ Legal Issues",
                placeholder="What are the main legal questions or challenges?",
                height=100
            )
            
            reasoning = st.text_area(
                "🤔 Reasoning",
                placeholder="Enter the legal reasoning or analysis...",
                height=100
            )
            
            decision = st.text_area(
                "📋 Decision",
                placeholder="What was the outcome or decision?",
                height=100
            )

            submit_button = st.form_submit_button(
                "Generate Summary",
                use_container_width=True
            )

        if submit_button:
            if not facts.strip() or not issues.strip():
                st.warning("Please provide at least the facts and legal issues.")
            else:
                request_data = {
                    "facts": facts,
                    "issues": issues,
                    "reasoning": reasoning,
                    "decision": decision
                }
                
                with st.spinner("Generating summary..."):
                    response = requests.post(
                        f"{API_URL}/generate_summary",
                        json=request_data
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("summary"):
                            st.write("### Generated Summary")
                            with st.expander("View Complete Summary", expanded=True):
                                st.write(data["summary"])
                                
                                # Optional: Add download buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Download as PDF"):
                                        # Add PDF download logic
                                        pass
                                with col2:
                                    if st.button("Download as DOCX"):
                                        # Add DOCX download logic
                                        pass
                        else:
                            st.warning(data.get("message", "I couldn't generate a summary. Please try with different input."))
                    else:
                        st.error("I encountered an error while generating the summary. Please try again.")

def reset_summary_state():
    """Reset any summary-specific session state"""
    if 'summary_form_data' in st.session_state:
        del st.session_state.summary_form_data
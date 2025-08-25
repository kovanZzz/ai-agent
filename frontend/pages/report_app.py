import streamlit as st
import requests
from datetime import datetime, timedelta
import re

# Configuration
BACKEND_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="Lease Agreement Generator",
    page_icon="🏠",
    layout="wide"
)

def init_session_state():
    if "form_data" not in st.session_state:
        st.session_state.form_data = {}
    if "generated_report" not in st.session_state:
        st.session_state.generated_report = None
    if "current_step" not in st.session_state:
        st.session_state.current_step = "input"

def validate_email(email):
    """Validate email format"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)

def validate_phone(phone):
    """Validate phone number format"""
    pattern = r'^\+?1?\d{9,15}$'
    return re.match(pattern, phone)

def calculate_lease_duration(start_date, end_date):
    """Calculate lease duration in months"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return max(1, months)

def input_form():
    """Display the main input form"""
    st.title("🏠 Professional Lease Agreement Generator")
    
    # Create tabs for different sections
    tabs = st.tabs(["Property Details", "Landlord Info", "Tenant Info", "Lease Terms"])
    
    with tabs[0]:
        st.subheader("📍 Property Information")
        col1, col2 = st.columns(2)
        
        with col1:
            property_type = st.selectbox(
                "Property Type *",
                options=["Apartment", "House", "Condo", "Townhouse", "Room", "Studio"],
                key="prop_type"
            )
            property_address = st.text_input(
                "Street Address *",
                key="prop_address"
            )
            property_unit = st.text_input(
                "Unit/Apt Number",
                key="prop_unit"
            )
        
        with col2:
            property_city = st.text_input("City *", key="prop_city")
            property_state = st.selectbox(
                "State *",
                options=[
                    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
                ],
                key="prop_state"
            )
            property_zipcode = st.text_input("ZIP Code *", key="prop_zip")
    
    with tabs[1]:
        st.subheader("👤 Landlord Information")
        col3, col4 = st.columns(2)
        
        with col3:
            landlord_name = st.text_input("Full Name *", key="landlord_name")
            landlord_address = st.text_input("Street Address *", key="landlord_address")
            landlord_city = st.text_input("City *", key="landlord_city")
            landlord_state = st.selectbox(
                "State *",
                options=[
                    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
                ],
                key="landlord_state"
            )
        
        with col4:
            landlord_zipcode = st.text_input("ZIP Code *", key="landlord_zipcode")
            landlord_email = st.text_input("Email Address *", key="landlord_email")
            landlord_phone = st.text_input("Phone Number *", key="landlord_phone")
    
    with tabs[2]:
        st.subheader("👥 Tenant Information")
        col5, col6 = st.columns(2)
        
        with col5:
            tenant_names = st.text_input(
                "Full Name(s) * (comma-separated for multiple tenants)",
                key="tenant_names"
            )
            tenant_phone = st.text_input("Phone Number *", key="tenant_phone")
            tenant_email = st.text_input("Email Address *", key="tenant_email")
        
        with col6:
            tenant_current_address = st.text_input("Current Address", key="tenant_address")
            tenant_current_city = st.text_input("Current City", key="tenant_city")
            tenant_current_state = st.selectbox(
                "Current State",
                options=[
                    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
                ],
                key="tenant_state"
            )
            tenant_current_zipcode = st.text_input("Current ZIP Code", key="tenant_zipcode")
    
    with tabs[3]:
        st.subheader("📋 Lease Terms")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            monthly_rent = st.number_input(
                "Monthly Rent ($) *",
                min_value=0,
                step=50,
                key="monthly_rent"
            )
            security_deposit = st.number_input(
                "Security Deposit ($) *",
                min_value=0,
                step=50,
                key="security_deposit"
            )
            
        with col8:
            start_date = st.date_input(
                "Lease Start Date *",
                value=datetime.today(),
                key="start_date"
            )
            
        with col9:
            end_date = st.date_input(
                "Lease End Date *",
                value=datetime.today() + timedelta(days=365),
                key="end_date"
            )
        
        st.subheader("📝 Additional Terms and Policies")
        col10, col11 = st.columns(2)
        
        with col10:
            pets_allowed = st.selectbox(
                "Pet Policy",
                options=["No Pets Allowed", "Pets Allowed with Approval", "Pets Allowed"],
                key="pets_allowed"
            )
            if pets_allowed in ["Pets Allowed with Approval", "Pets Allowed"]:
                pet_deposit = st.number_input("Pet Deposit ($)", min_value=0, step=50, key="pet_deposit")
                pet_rent = st.number_input("Monthly Pet Rent ($)", min_value=0, step=10, key="pet_rent")
        
        with col11:
            smoking_policy = st.selectbox(
                "Smoking Policy",
                options=["No Smoking Allowed", "Smoking Allowed in Designated Areas", "Smoking Allowed"],
                key="smoking_policy"
            )
            
        utilities_included = st.multiselect(
            "Utilities Included in Rent",
            options=["Water", "Electricity", "Gas", "Internet", "Cable TV", "Trash", "Sewer"],
            key="utilities"
        )
        
        additional_terms = st.text_area(
            "Additional Terms and Conditions",
            key="additional_terms"
        )

    # Collect form data
    form_data = {
        "property_type": property_type,
        "property_address": property_address,
        "property_unit": property_unit,
        "property_city": property_city,
        "property_state": property_state,
        "property_zipcode": property_zipcode,
        "landlord_name": landlord_name,
        "landlord_address": landlord_address,
        "landlord_city": landlord_city,
        "landlord_state": landlord_state,
        "landlord_zipcode": landlord_zipcode,
        "landlord_email": landlord_email,
        "landlord_phone": landlord_phone,
        "tenant_names": tenant_names,
        "tenant_phone": tenant_phone,
        "tenant_email": tenant_email,
        "tenant_current_address": tenant_current_address,
        "tenant_current_city": tenant_current_city,
        "tenant_current_state": tenant_current_state,
        "tenant_current_zipcode": tenant_current_zipcode,
        "monthly_rent": monthly_rent,
        "security_deposit": security_deposit,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "pets_allowed": pets_allowed,
        "pet_deposit": pet_deposit if 'pet_deposit' in locals() else 0,
        "pet_rent": pet_rent if 'pet_rent' in locals() else 0,
        "smoking_policy": smoking_policy,
        "utilities_included": utilities_included,
        "additional_terms": additional_terms
    }

    # Generate button
    col_button1, col_button2, col_button3 = st.columns([1, 1, 1])
    with col_button2:
        if st.button("Generate Lease Agreement", type="primary", key="generate_button"):
            if validate_form(form_data):
                st.session_state.form_data = form_data
                st.session_state.current_step = "preview"
                generate_report()

def validate_form(form_data):
    """Validate form data"""
    required_fields = [
        "property_address", "property_city", "property_state", "property_zipcode",
        "landlord_name", "landlord_email", "landlord_phone",
        "tenant_names", "tenant_phone", "tenant_email",
        "monthly_rent", "security_deposit"
    ]
    
    missing_fields = [field for field in required_fields if not form_data.get(field)]
    if missing_fields:
        st.error(f"Please fill in all required fields: {', '.join(missing_fields)}")
        return False
    
    if not validate_email(form_data["landlord_email"]):
        st.error("Please enter a valid landlord email address")
        return False
        
    if not validate_email(form_data["tenant_email"]):
        st.error("Please enter a valid tenant email address")
        return False
    
    if not validate_phone(form_data["landlord_phone"]):
        st.error("Please enter a valid landlord phone number")
        return False
        
    if not validate_phone(form_data["tenant_phone"]):
        st.error("Please enter a valid tenant phone number")
        return False
    
    return True

def generate_report():
    """Generate lease agreement"""
    try:
        with st.spinner("Generating lease agreement..."):
            response = requests.post(
                f"{BACKEND_URL}/generate_report",
                json={"form_data": st.session_state.form_data}
            )
            
            if response.status_code == 200:
                report_data = response.json()
                st.session_state.generated_report = report_data["blocks"][0]["text"]
                preview_report()
            else:
                st.error("Failed to generate lease agreement. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def preview_report():
    """Preview generated lease agreement and enable downloads"""
    if st.session_state.generated_report:
        st.title("📄 Generated Lease Agreement")
        
        st.text_area(
            "Preview",
            value=st.session_state.generated_report,
            height=400,
            key="preview_area",
            disabled=True
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            pdf_response = requests.post(
                f"{BACKEND_URL}/generate_pdf",
                json={"form_data": st.session_state.form_data}
            )
            if pdf_response.status_code == 200:
                st.download_button(
                    label="Download PDF",
                    data=pdf_response.content,
                    file_name="lease_agreement.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Failed to generate PDF. Please try again.")

        with col2:
            docx_response = requests.post(
                f"{BACKEND_URL}/generate_docx",
                json={"form_data": st.session_state.form_data}
            )
            if docx_response.status_code == 200:
                st.download_button(
                    label="Download DOCX",
                    data=docx_response.content,
                    file_name="lease_agreement.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.error("Failed to generate DOCX. Please try again.")

        with col3:
            if st.button("Create New Lease Agreement"):
                st.session_state.current_step = "input"
                st.session_state.generated_report = None
                st.session_state.form_data = {}
                st.rerun()

def run_report_app():
    """Run the report app"""
    input_form()
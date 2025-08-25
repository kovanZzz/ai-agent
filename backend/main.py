import os
import requests
from openai import OpenAI
from typing import Union
import logging
from datetime import datetime, timedelta
from getpass import getpass
from typing import TypedDict, Optional, Dict, List, Any, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from docx import Document
import cohere
from fpdf import FPDF
from dotenv import load_dotenv
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from semantic_router.encoders import OpenAIEncoder
from pinecone import Pinecone
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import jwt, JWTError
from report_agent import generate_report, generate_pdf, generate_docx
from strategy_agent import process_text, process_file, get_embedding, similarity_search, fetch_arxiv_papers, extract_text_from_pdf, extract_text_from_file, generate_strategy
from summary_agent import generate_summary_endpoint, upload_file, get_embedding, extract_text_from_file, extract_text_from_zip, generate_summary_file, generate_summary_imput

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-ada-002"

# Load environment variables

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(COHERE_API_KEY)

# Database Setup
SQLALCHEMY_DATABASE_URL = os.getenv("AZURE_SQL_CONNECTION_STRING")
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

SECRET_KEY = os.getenv('SECRET_KEY', 'secret')
ALGORITHM = os.getenv('ALGORITHM', 'HS256')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 30))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize FastAPI
app = FastAPI()

class TextBlock(BaseModel):
    text: str

class InputText(BaseModel):
    text: str

class DocumentRequest(BaseModel):
    form_data: Dict[str, Any]

class ReportOutput(BaseModel):
    title: str
    blocks: List[Union[TextBlock]]
    metadata: Dict[str, Any]

class QueryInput(BaseModel):
    query: str
    context: Optional[Dict] = {}

class LeaseAgreementPDF(FPDF):
    def __init__(self, title="Lease Agreement"):
        super().__init__()
        self.title = title

    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, self.title, align='C', ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

class DocumentRequest(BaseModel):
    form_data: Dict[str, Any]

# State type definition
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    context: Dict
    query: str
    agent_type: Optional[str] 

class QueryInput(BaseModel):
    query: str
    context: Optional[Dict] = {}

# User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, index=True)
    hashed_password = Column(String)

# Token storage model
class UserToken(Base):
    __tablename__ = "user_tokens"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    token = Column(String, unique=True, index=True)

Base.metadata.create_all(bind=engine)

class SummaryAgent:
    @staticmethod
    def generate_summary(state: AgentState) -> AgentState:
        """Generate summary using the backend summary endpoint"""
        try:
            query = state["query"]
            context = state["context"]
            
            # Check if this is a file upload request
            if "file_content" in context:
                # Handle file upload path
                file_content = context["file_content"]
                prompt = f"""
                Summarize the following document into the following categories:
                1. Case Summary
                2. Background
                3. Key Facts
                4. Key Issues
                5. Court's Opinion and Decision
                6. Key Observations
                7. Conclusion

                Document:
                {file_content}
                """
                summary = generate_summary_file(prompt)
            else:
                # Handle manual input path
                summary = generate_summary_imput(f"""
                Facts: {context.get('facts', '')}
                Issues: {context.get('issues', '')}
                Reasoning: {context.get('reasoning', '')}
                Decision: {context.get('decision', '')}
                """)

            state["messages"] = [
                {"role": "system", "content": "Summary Generation Complete"},
                {"role": "assistant", "content": summary}
            ]
            return state
        except Exception as e:
            state["messages"] = [
                {"role": "system", "content": "Error in Summary Generation"},
                {"role": "assistant", "content": f"Error generating summary: {str(e)}"}
            ]
            return state

class SummaryRequest(BaseModel):
    facts: str
    issues: str
    reasoning: str
    decision: str

# Global variables for services
pc = None
index = None
model = None
encoder = None

def validate_env_vars():
    required_vars = {
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }
    
    missing_vars = [key for key, value in required_vars.items() if not value]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return required_vars

LEASE_TEMPLATE = """
STATE OF {state}                                                    Rev. {revision_date}

LEASE AGREEMENT

This Lease Agreement (this "Agreement") is made this {current_date}, by and between {landlord_name} ("Landlord") and {tenant_names} ("Tenant"). Each Landlord and Tenant may be referred to individually as a "Party" and collectively as the "Parties."

1. PREMISES
The premises leased is a {property_type} located at {property_address}, {property_city}, {property_state} {property_zipcode} (the "Premises").

2. TERM
This Agreement will be for a term beginning on {start_date} and ending on {end_date} (the "Term").

3. RENT AND PAYMENTS
3.1 Monthly Rent: ${monthly_rent}
    Payable in advance on the 1st day of each month.
3.2 Security Deposit: ${security_deposit}
    To be returned within 30 days of lease termination, less any deductions.
3.3 Late Fees: 5% of monthly rent if paid after the 5th day of the month.

4. UTILITIES AND SERVICES
Tenant is responsible for the following utilities:
- Electricity
- Water/Sewer
- Gas
- Internet/Cable
- Trash Collection

5. OCCUPANCY AND USE
5.1 The Premises shall be occupied only by the named Tenant(s).
5.2 The Premises shall be used as a residential dwelling only.

6. MAINTENANCE AND REPAIRS
6.1 Tenant Responsibilities:
    - Keep the Premises clean and sanitary
    - Proper use of all appliances and fixtures
    - Promptly report any maintenance issues
6.2 Landlord Responsibilities:
    - Maintain structural elements
    - Ensure working plumbing, heating, and electrical systems
    - Provide pest control as needed

7. RULES AND REGULATIONS
7.1 Noise and Disturbances: Quiet hours 10 PM - 8 AM
7.2 Smoking: {smoking_policy}
7.3 Pets: {pet_policy}
7.4 Alterations: No alterations without written consent

8. ADDITIONAL TERMS
{additional_terms}

9. DEFAULT AND REMEDIES
9.1 Events of Default:
    - Failure to pay rent when due
    - Violation of any term of this Agreement
9.2 Remedies:
    - Termination of lease
    - Legal action for unpaid rent
    - Eviction proceedings

10. SIGNATURES AND DATE

_______________________
Landlord Signature
Date: {current_date}

_______________________
Tenant Signature
Date: {current_date}

"""

def init_services():
    global pc, index, model, encoder
    
    try:
        env_vars = validate_env_vars()
        
        # Initialize Pinecone
        logger.info("Initializing Pinecone...")
        pc = Pinecone(api_key=env_vars["PINECONE_API_KEY"])
        indexes = pc.list_indexes()
        index_name = env_vars["PINECONE_INDEX_NAME"]
        
        if not any(idx.name == index_name for idx in indexes):
            available_indexes = [idx.name for idx in indexes]
            error_msg = f"Index '{index_name}' not found. Available indexes: {available_indexes}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        index = pc.Index(index_name)
        logger.info(f"Successfully connected to Pinecone index: {index_name}")
        
        # Initialize OpenAI
        logger.info("Initializing OpenAI...")
        model = ChatOpenAI(
            api_key=env_vars["OPENAI_API_KEY"],
            model="gpt-4"
        )
        
        # Initialize encoder
        logger.info("Initializing encoder...")
        encoder = OpenAIEncoder()
        
    except Exception as e:
        error_msg = f"Error initializing services: {str(e)}"
        logger.error(error_msg)
        raise

class AgentTools:
    @staticmethod
    def generate_report(state: AgentState) -> AgentState:
        """Generate a detailed report using existing code"""
        try:
            form_data = state.get("context", {})
            current_date = datetime.now().strftime("%B %d, %Y")
            
            # Use your existing comprehensive prompt
            comprehensive_prompt = f"""
Generate a detailed, professional lease agreement using these details:

LANDLORD INFO:
Name: {form_data.get('landlord_name')}
Address: {form_data.get('landlord_address')}
Location: {form_data.get('landlord_city')}, {form_data.get('landlord_state')} {form_data.get('landlord_zipcode')}

PROPERTY INFO:
Type: {form_data.get('property_type')}
Address: {form_data.get('property_address')}
Monthly Rent: ${form_data.get('monthly_rent')}
Security Deposit: ${form_data.get('security_deposit')}

LEASE TERM:
Start: {form_data.get('start_date')}
End: {form_data.get('end_date')}

Create a comprehensive lease agreement that includes all standard protections and clauses.
Use formal legal language but ensure it remains clear and understandable.
Include sections for utilities, maintenance, default conditions, and remedies.
"""

            # Generate using Cohere
            response = cohere_client.generate(
                model="command-xlarge",
                prompt=comprehensive_prompt,
                max_tokens=2500,
                temperature=0.4,
                stop_sequences=["\n\n\n"]
            )
            
            generated_text = response.generations[0].text.strip()
            
            # Fallback to template if generation is insufficient
            if len(generated_text) < 1000:
                # Set default policies
                smoking_policy = "Smoking is strictly prohibited inside the Premises."
                pet_policy = "No pets allowed without prior written consent from Landlord."
                
                # Fill template
                generated_text = LEASE_TEMPLATE.format(
                    current_date=current_date,
                    revision_date=datetime.now().strftime("%Y%m%d"),
                    smoking_policy=smoking_policy,
                    pet_policy=pet_policy,
                    **form_data
                )

            # Create report output using the properly imported model
            report_output = ReportOutput(
                title="Lease Agreement",
                blocks=[TextBlock(text=generated_text)],
                metadata={
                    "generated_by": "LegalAI",
                    "timestamp": datetime.now().isoformat(),
                    "version": "2.0"
                }
            )
            
            # Update state with generated report
            state["messages"] = [
                SystemMessage(content="Report Generation Complete"),
                AIMessage(content=report_output.blocks[0].text)
            ]
            state["report_output"] = report_output.dict()
            return state
            
        except Exception as e:
            logger.error(f"Error in generate_report: {str(e)}")
            state["messages"] = [
                SystemMessage(content="Error in Report Generation"),
                AIMessage(content=f"Error generating report: {str(e)}")
            ]
            return state

    @staticmethod
    def generate_summary(state: AgentState) -> AgentState:
        """Generate a summary using the backend summary endpoint"""
        try:
            query = state["query"]
            context = state["context"]
            
            # Call backend `/generate_summary` endpoint
            response = requests.post(
                "http://localhost:8000/generate_summary",
                json={
                    "facts": context.get("facts", ""),
                    "issues": context.get("issues", ""),
                    "reasoning": context.get("reasoning", ""),
                    "decision": context.get("decision", "")
                }
            )
            
            if response.status_code == 200:
                summary = response.json().get("summary", "No summary available.")
                state["messages"] = [
                    SystemMessage(content="Summary Generation Complete"),
                    AIMessage(content=summary)
                ]
                return state
            else:
                error = response.json().get("detail", "Unknown error.")
                raise ValueError(f"Error from summary agent: {error}")
        except Exception as e:
            state["messages"] = [
                SystemMessage(content="Error in Summary Generation"),
                AIMessage(content=f"Error generating summary: {str(e)}")
            ]
            return state

    @staticmethod
    def generate_strategy(state: AgentState) -> AgentState:
        """Generate a strategy"""
        try:
            query = state["query"]
            context = state["context"]
            
            strategy_prompt = f"""
Generate a comprehensive strategy based on:

QUERY: {query}

CONTEXT:
{context}

Create an actionable strategy that includes:
1. Strategic objectives
2. Action steps
3. Implementation timeline
4. Success metrics
"""

            response = cohere_client.generate(
                model="command-xlarge",
                prompt=strategy_prompt,
                max_tokens=2500,
                temperature=0.4,
                stop_sequences=["\n\n\n"]
            )
            
            generated_text = response.generations[0].text.strip()
            
            state["messages"] = [
                SystemMessage(content="Strategy Generation Complete"),
                AIMessage(content=generated_text)
            ]
            return state
            
        except Exception as e:
            logger.error(f"Error in generate_strategy: {str(e)}")
            state["messages"] = [
                SystemMessage(content="Error in Strategy Generation"),
                AIMessage(content=f"Error generating strategy: {str(e)}")
            ]
            return state

    @staticmethod
    def generate_strategy(state: AgentState) -> AgentState:
        """Generate a legal strategy."""
        try:
            context_metadata = state.get("context", {})
            prompt = f"""
            You are a legal strategy generation agent. Based on the following similar court case metadata, provide a strategy:
            
            Facts: {context_metadata.get('Facts', 'N/A')}
            Issues: {context_metadata.get('Issues', 'N/A')}
            Reasoning: {context_metadata.get('Reasoning', 'N/A')}
            Decision: {context_metadata.get('Decision', 'N/A')}
            
            Please propose a comprehensive legal strategy based on the above metadata:
            """

            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a legal strategy assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.5,
            )

            generated_text = gpt_response.choices[0].message.content
            state["messages"] = [
                SystemMessage(content="Strategy Generation Complete"),
                AIMessage(content=generated_text),
            ]
            return state

        except Exception as e:
            state["messages"] = [
                SystemMessage(content="Error generating strategy"),
                AIMessage(content=f"Error generating strategy: {str(e)}"),
            ]
            return state

def analyze_intent(state: AgentState) -> AgentAction:
    """Analyze query intent to route to the appropriate agent"""
    query = state["query"].lower()
    
    # Define keyword sets for each agent
    report_keywords = {"report", "lease", "document", "analysis", "contract"}
    strategy_keywords = {"strategy", "plan", "approach", "recommendation", "steps"}
    summary_keywords = {"summary", "summarize", "overview", "case summary", "brief"}
    
    # Split query into words
    query_words = set(query.split())
    
    # Calculate scores for each agent type
    report_score = len(report_keywords.intersection(query_words))
    strategy_score = len(strategy_keywords.intersection(query_words))
    summary_score = len(summary_keywords.intersection(query_words))
    
    # Log the scores for debugging purposes
    logger.info(f"Query: {query}")
    logger.info(f"Report Score: {report_score}, Strategy Score: {strategy_score}, Summary Score: {summary_score}")
    
    # Route based on scores
    if summary_score >= max(report_score, strategy_score):
        action = AgentAction(
            tool="generate_summary",
            tool_input={"query": query, "context": state["context"]},
            log="Routing to summary generation"
        )
    elif report_score >= max(summary_score, strategy_score):
        action = AgentAction(
            tool="generate_report",
            tool_input={"query": query, "context": state["context"]},
            log="Routing to report generation"
        )
    elif strategy_score > max(report_score, summary_score):
        action = AgentAction(
            tool="generate_strategy",
            tool_input={"query": query, "context": state["context"]},
            log="Routing to strategy generation"
        )
    else:
        # Default fallback if no clear match
        action = AgentAction(
            tool="unknown",
            tool_input={"query": query, "context": state["context"]},
            log="No matching agent found"
        )
    
    # Log the final routing decision
    logger.info(f"Routing decision: {action.log}")
    
    return action

def route_by_action(state: AgentState) -> AgentState:
    """Route to next node based on the query"""
    query = state["query"].lower()
    
    # Define keyword sets for each agent
    report_keywords = {"report", "lease", "agreement", "document", "contract"}
    strategy_keywords = {"strategy", "plan", "approach", "recommendation", "action", "steps"}
    summary_keywords = {"summary", "summarize", "overview", "case summary", "brief"}
    
    # Calculate scores for each agent
    report_score = len([word for word in report_keywords if word in query])
    strategy_score = len([word for word in strategy_keywords if word in query])
    summary_score = len([word for word in summary_keywords if word in query])
    
    # Determine the agent type based on the highest score
    if summary_score >= max(report_score, strategy_score):
        state["agent_type"] = "summary"
    elif report_score >= max(summary_score, strategy_score):
        state["agent_type"] = "report"
    else:
        state["agent_type"] = "strategy"
    
    # Add routing message
    state["messages"] = [
        SystemMessage(content=f"Routing to {state['agent_type']} agent")
    ]
    
    logger.info(f"Routing query to {state['agent_type']} agent")
    return state

SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.csv', '.xlsx', '.png', '.jpg', '.jpeg', '.py', '.zip', '.pdb', '.pptx']


@app.post("/generate_summary")
async def generate_summary_endpoint(request: SummaryRequest):
    """Generate summary using the agent system"""
    try:
        state = AgentState(
            query="generate summary",
            context={
                "facts": request.facts,
                "issues": request.issues,
                "reasoning": request.reasoning,
                "decision": request.decision
            },
            agent_type="summary"
        )
        
        result = SummaryAgent.generate_summary(state)
        
        if result and result.get("messages"):
            summary = result["messages"][-1].get("content")
            return {"summary": summary}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate summary")
            
    except Exception as e:
        logger.exception("Error generating summary")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_file")
async def upload_file_endpoint(file: UploadFile = File(...)):
    return await upload_file(file)    


def create_agent_graph() -> StateGraph:
    """Create the workflow graph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", route_by_action)
    workflow.add_node("report_generator", AgentTools.generate_report)
    workflow.add_node("strategy_generator", AgentTools.generate_strategy)
    workflow.add_node("summary_generator", AgentTools.generate_summary)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges based on agent_type
    def route_to_agent(state: AgentState) -> str:
        return state["agent_type"]
    
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "report": "report_generator",
            "strategy": "strategy_generator",
            "summary": "summary_generator"
        }
    )
    
    # Add edges to END
    workflow.add_edge("report_generator", END)
    workflow.add_edge("strategy_generator", END)
    workflow.add_edge("summary_generator", END)
    
    return workflow.compile()

@app.post("/process")
async def process_query(request: dict):
    """Process incoming query and route to appropriate agent"""
    try:
        query = request.get("query", "").lower()
        context = request.get("context", {})
        
        # Route to appropriate agent based on query
        if "strategy" in query:
            # Strategy Agent Logic
            input_text = f"""
            Facts: {context.get('facts', '')}
            Issues: {context.get('issues', '')}
            Reasoning: {context.get('reasoning', '')}
            Decision: {context.get('decision', '')}
            """
            
            embedding = get_embedding(input_text)
            search_results = similarity_search(embedding, top_k=1)
            
            if search_results and len(search_results['matches']) > 0:
                best_match = search_results['matches'][0]
                metadata = best_match['metadata']
                strategy = generate_strategy(metadata)
                issues_query = metadata.get('Issues', 'N/A')
                
                papers = []
                if issues_query != 'N/A' and issues_query.strip():
                    papers = fetch_arxiv_papers(issues_query, max_results=3)
                
                return {
                    "status": "success",
                    "message": "I'll help you develop a strategy.",
                    "agent": "strategy",
                    "result": strategy,
                    "papers": papers
                }
            else:
                return {
                    "status": "success",
                    "message": "I'll help you develop a strategy.",
                    "agent": "strategy",
                    "result": "No similar cases found to generate strategy. Please provide more details.",
                    "papers": []
                }

        elif "summary" in query:
            # Summary Agent Logic
            # Check if context has file content or manual input
            if context.get("file_content"):
                # Handle file-based summary
                file_content = context["file_content"]
                prompt = f"""
                Summarize the following document into the following categories:
                1. Case Summary
                2. Background
                3. Key Facts
                4. Key Issues
                5. Court's Opinion and Decision
                6. Key Observations
                7. Conclusion

                Document:
                {file_content}
                """
                summary = generate_summary_file(prompt)
            else:
                # Handle manual input summary
                input_text = f"""
                Facts: {context.get('facts', '')}
                Issues: {context.get('issues', '')}
                Reasoning: {context.get('reasoning', '')}
                Decision: {context.get('decision', '')}
                """
                summary = generate_summary_imput(input_text)

            return {
                "status": "success",
                "message": "I'll help you create a summary.",
                "agent": "summary",
                "result": summary
            }

        elif "report" in query:
            # Report Agent Logic
            if "lease" in query:
                report_type = "lease"
            elif "nda" in query:
                report_type = "nda"
            elif "divorce" in query:
                report_type = "divorce"
            else:
                report_type = "general"

            # Get report template based on type
            template = get_report_template(report_type)
            
            # Generate report using context
            report_content = generate_report(
                template=template,
                context={
                    "property_type": context.get("property_type"),
                    "property_address": context.get("property_address"),
                    "landlord_name": context.get("landlord_name"),
                    "tenant_name": context.get("tenant_name"),
                    "monthly_rent": context.get("monthly_rent"),
                    "lease_term": context.get("lease_term"),
                    "start_date": context.get("start_date"),
                    "end_date": context.get("end_date"),
                    # Add other fields as needed
                }
            )

            return {
                "status": "success",
                "message": f"I'll help you generate a {report_type} report.",
                "agent": "report",
                "result": report_content,
                "report_type": report_type
            }

        # No matching agent found
        return {
            "status": "error",
            "message": "I'm not sure what you'd like me to do. I can help you with generating reports, developing strategies, or creating summaries. What would you like me to do?",
            "agent": None
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "status": "error",
            "message": f"I encountered an error while processing your request: {str(e)}",
            "agent": None
        }

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting application...")
        init_services()
        logger.info("Application started successfully")
    except Exception as e:
        error_msg = f"Startup error: {str(e)}"
        logger.error(error_msg)
        raise

@app.get("/health")
async def health_check():
    try:
        services_status = {
            "pinecone": index is not None,
            "openai": model is not None,
            "encoder": encoder is not None
        }
        
        if all(services_status.values()):
            return {
                "status": "healthy",
                "services": services_status
            }
        else:
            return {
                "status": "unhealthy",
                "services": services_status
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/process_text")
async def process_text_endpoint(input_text: InputText):
    return await process_text(input_text)

@app.post("/process_file")
async def process_file_endpoint(file: UploadFile = File(...)):
    return await process_file(file)

@app.post("/generate_report")
async def generate_report_endpoint(request: DocumentRequest):
    return await generate_report(request)

@app.post("/generate_pdf")
async def generate_pdf_endpoint(request: DocumentRequest):
    return await generate_pdf(request)

@app.post("/generate_docx")
async def generate_docx_endpoint(request: DocumentRequest):
    return await generate_docx(request)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)   
    return encoded_jwt 
    
@app.post("/register")
async def register_user(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user_exists = db.query(User).filter(User.username == username).first()
    if user_exists:
        return {"error": "Username already exists"}

    hashed_password = pwd_context.hash(password)
    new_user = User(username=username, hashed_password=hashed_password)
    db.add(new_user)
    try:
        db.commit()
        db.refresh(new_user)
        return {"message": "Registration successful. Please login."}
    except:
        db.rollback()
        return {"error": "Database error"}

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    
    # Save token to DB
    user_token = UserToken(user_id=user.id, token=access_token)
    db.add(user_token)
    db.commit()
    db.refresh(user_token)

    return {"access_token": access_token, "token_type": "bearer"}

# SmartMetrics-AI-Multi-Agent-Intelligence-Engine
![thumbnail](assets/SmartMetrics AI Multi-Agent Intelligence Engine.PNG)

An autonomous multi-agent pipeline that collects data, analyzes competitors, builds strategies, and generates business reports.
# **SmartMetrics-AI – Multi-Agent Intelligence Engine**
### *Autonomous Pipeline: Data Collection → Analysis → Strategy → Reporting*

This notebook contains the ENTIRE AI agent workflow in one place, including:

- Data Collection Agent  
- Analysis Agent  
- Strategy Agent  
- Report Generator  
- Full Autonomous Pipeline (Agent Collaboration)



# ============================================================
# 1️⃣.0 IMPORTS 
# ============================================================
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ============================================================
# 1️⃣.1️⃣ IMPORTS & ENVIRONMENT SETUP
# ============================================================

import os
import pandas as pd
import numpy as np
from IPython.display import display

# 1. Import the necessary client from the Kaggle library
from kaggle_secrets import UserSecretsClient 

import google.generativeai as genai

# 2. Retrieve the API key from your Kaggle Secrets (assuming you named the secret "GEMINI_API_KEY")
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GEMINI_API_KEY") # This retrieves the key securely

# Configure Gemini API
genai.configure(api_key=api_key) # Pass the retrieved key directly
llm = genai.GenerativeModel("gemini-2.5-pro")

# ============================================================
# 1️⃣.2️⃣ MEMORY BANK (SHARED CONTEXT FOR ALL AGENTS)
# ============================================================

# Memory Management (Shared Digital Notepad)

class MemoryBank:
    """Simple memory store for shared notes and context."""
    def __init__(self):
        self.history = []

    def add(self, agent_name, text):
        self.history.append({"agent": agent_name, "text": text})

    def get_full_history(self):
        return "\n".join([f"{h['agent']}: {h['text']}" for h in self.history])
    
memory = MemoryBank()



# ============================================================
# 1️⃣ DATA COLLECTION AGENT
# ============================================================

def data_collection_agent(input_text):
    """Collects competitor info from Gemini based on input URL or description."""
    prompt = f"""
    You are a Competitor Research Agent.
    Collect market intelligence based on this input:
    {input_text}

    Return in JSON:
    {{
        "name": "...",
        "website": "...",
        "industry": "...",
        "products_services": "...",
        "target_users": "...",
        "pricing": "...",
        "strengths": ["..."],
        "weaknesses": ["..."],
        "market_gaps": ["..."]
    }}
    """
    response = llm.generate_content(prompt).text
    memory.add("DataAgent", response)
    return response


# ============================================================
# 2️⃣ ANALYSIS AGENT
# ============================================================


def analysis_agent(data_list):
    """Aggregates competitor info, generates comparison table and insights."""
    prompt = f"""
    You are a Competitor Analysis Agent.
    Here is a list of competitor data (JSON):
    {data_list}

    Generate:
    1. Competitor comparison table
    2. Summary of weaknesses per competitor
    3. Identification of market gaps
    Return as structured text or JSON.
    """
    response = llm.generate_content(prompt).text
    memory.add("AnalysisAgent", response)
    return response



# ============================================================
#3️⃣ STRATEGY AGENT
# ============================================================


def strategy_agent(analysis):
    """Generates actionable recommendations based on analysis."""
    prompt = f"""
    You are a Strategy Agent.
    Based on the competitor analysis below:
    {analysis}

    Generate:
    1. Strategic recommendations
    2. Actionable steps for next 60 days
    3. Long-term opportunities
    """
    response = llm.generate_content(prompt).text
    memory.add("StrategyAgent", response)
    return response




# ============================================================
#4️⃣.REPORT GENERATION AGENT
# ============================================================

def report_agent(final_data):
    """Generates a professional structured report."""
    prompt = f"""
    You are a Report Generation Agent.
    Based on all previous outputs:
    {final_data}

    Produce a report including:
    - Executive Summary
    - Competitor Comparison Table
    - Weaknesses
    - Market Gaps
    - Actionable Recommendations
    """
    response = llm.generate_content(prompt).text
    memory.add("ReportAgent", response)
    return response


# ============================================================
# 5️⃣. FULL AUTONOMOUS PIPELINE
# ============================================================
#   — Autonomous Pipeline (Agent Collaboration)

def ai_competitor_intelligence_pipeline(inputs):
    """
    Autonomous multi-agent workflow:
    - Real-time data collection
    - Analysis
    - Strategy
    - Report generation
    """
    collected_data = []
    
    # Step 1: Data Collection (parallel simulation for simplicity)
    for inp in inputs:
        print(f"Collecting data for: {inp}")
        data = data_collection_agent(inp)
        collected_data.append(data)
    
    # Step 2: Analysis
    analysis = analysis_agent(collected_data)
    
    # Step 3: Strategy recommendations
    strategy = strategy_agent(analysis)
    
    # Step 4: Final report
    report = report_agent(strategy)
    
    return report

# ============================================================
# 6️⃣. 8.RUN NOTEBOOK
# ============================================================

# Input: list of URLs or business descriptions
inputs = [
     "https://www.decathlon.in/shop/Winter-Collection" 
]

# Run the autonomous competitor intelligence system
final_report = ai_competitor_intelligence_pipeline(inputs)

# Display the final structured report
print(final_report)



**Bidding Agentic AI**

An Agentic AI system designed to analyze RFPs (Request for Proposals), match them with enterprise product catalogs, and generate structured quotations automatically.

**This project integrates:**

PDF/Document Parsing

Product Matching via Vector Similarity (TF-IDF / Embeddings)

LLM-powered Quotation Generation

FastMCP-based Tooling

GraphQL Enterprise Catalog Integration

Features

Automated RFP Parsing → Extracts requirements and specifications.

Product Matching → Matches RFP items with enterprise catalog products.

Quotation Generation → Creates structured JSON + HTML quotations.

Multi-Enterprise Support → Generates quotations per enterprise.

Token Optimization → Reduces cost by filtering catalogs and trimming RFPs.

**Installation**

Follow these steps to set up the project:

1. Set Base Path

Choose a folder where the project will be installed (e.g., ~/bidding).

mkdir -p ~/bidding
cd ~/bidding

2. Clone the Repository
repo link - 
cd Bidding-Agentic-AI

3. Install Dependencies

Make sure you have Python 3.11+ installed. Then run:

pip install -r requirements.txt

4. Install agents

Each agent is located in its own folder (e.g., quotation_tool, summary_tool).
You can install them with:

uv run mcp install main.py
Run this inside each agent’s folder.

after this step do check whether the agents have install propoerly in the cluade config json file.


****Technical requirements:****
- Install claude desktop

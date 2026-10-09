# Prediction AI 🚀

## 🌐 Live Demo
**[Try Prediction AI Live on Streamlit Cloud!](https://prediction-ai-app.streamlit.app/)**

## Overview
An intelligent decision-support platform that leverages Machine Learning and Large Language Models (LLMs) to predict startup and project failure risks, conduct comprehensive market analysis, and generate actionable strategic mitigations.

## Project Objectives
- **Risk Identification:** Early detection of potential vulnerabilities across financial, technical, market, and regulatory domains.
- **Strategic Guidance:** Provide startups and project managers with data-driven pivot recommendations.
- **Automated Intelligence:** Utilize multi-agent workflows to simulate real-world advisory roles and generate instant feedback on business plans.

## System Architecture
The application is built on a seamless, zero-configuration architecture:
- **Frontend:** Streamlit provides a highly interactive, responsive, and dynamic UI resembling a native desktop application.
- **Backend Core:** Python-based logic engines handling algorithmic risk scoring and market approximations.
- **AI Integration:** Google Gemini API acts as the cognitive engine for deep strategic reasoning and mitigation planning.
- **Database:** Local SQLite (`ml_project.db`) ensuring complete portability and zero setup overhead.

## System Workflow
1. **Data Ingestion:** The user submits a startup idea, specifying industry, business model, budget, and target market.
2. **Quantitative Analysis:** The system calculates TAM/SAM/SOM, market share projections, and a baseline feasibility score.
3. **Qualitative AI Assessment:** LLMs evaluate the project description to generate a SWOT analysis and actionable recommendations.
4. **Agentic Processing:** LangGraph workflows process the analysis in sequential, verifiable steps to produce a finalized risk report.

## Milestones Tasks

### 📍 Milestone 1: Data Collection & Market Intelligence
- **Project Submission Engine:** Seamless data ingestion for startup details.
- **Market Analysis Dashboards:** Algorithmic calculation of market size and 5-year trend projections.
- **Competitor Landscape:** Dynamic identification of competitors tracking market share and growth.

### 📍 Milestone 2: Risk Assessment & SWOT Analysis
- **Risk Scoring Algorithm:** Evaluates Team Capability, Financial Stability, Market Competition, and Tech Feasibility.
- **Project Feasibility:** Generates a holistic success probability score.
- **Automated SWOT Analysis:** Dynamically generates Strengths, Weaknesses, Opportunities, and Threats.

### 📍 Milestone 3: Recommendations & Strategic Reasoning
- **AI Strategic Recommendations:** Integrates Google Gemini API for high-level strategic pivots.
- **Dynamic Risk Mitigation:** Categorized mitigation strategies with interactive UI filtering.
- **Zero-Config Database:** Migrated from PostgreSQL to a seamless local SQLite database.

### 📍 Milestone 4: LangGraph Workflow & Cloud Deployment
- **Multi-Agent Orchestration:** Fully integrated LangGraph workflow sequentially connecting Risk Assessment, Recommendations, Mitigation, and Improvements.
- **Dynamic AI Dashboards:** Developed an interactive "Dashboard & Deployment" tab mapping real-time AI output states to Streamlit charts, metrics, and reports.
- **Cloud Deployment:** Live, globally accessible deployment via Streamlit Community Cloud integrated directly with GitHub.

## LangGraph Workflow
The platform utilizes an interactive multi-agent pipeline simulating a complete advisory team:
1. **Data Ingestion Agent:** Collects project details and market data.
2. **Risk Analysis Agent:** Evaluates vulnerabilities and assigns risk scores.
3. **Mitigation Agent:** Formulates strategies to counteract identified risks.
4. **Reporting Agent:** Compiles the findings into a structured, readable format.

## Technology Stack
- **Language:** Python 3.10+
- **Frontend/UI:** Streamlit, Custom HTML/CSS
- **Database:** SQLite3
- **AI / LLMs:** Google Gemini API (`google-genai`)
- **Data Manipulation:** Pandas

## Key Project Components
- `app_streamlit.py`: The main user interface and layout orchestrator.
- `llm_service.py`: Handles all secure communications and prompting with the Gemini API.
- `recommendation_engine.py` & `mitigation_engine.py`: Processes logic for turning raw risks into actionable advice.
- `market_analysis.py`: Algorithmic approximation of market metrics.
- `database.py`: Handles seamless data persistence to local storage.

## Project Outcome
Prediction AI delivers a robust, accessible tool that empowers entrepreneurs to stress-test their ideas before committing heavy resources. By combining deterministic data analysis with generative AI reasoning, the platform successfully bridges the gap between raw market data and strategic business intelligence.

---

### 🛠️ Quick Start
1. Clone the repository: `git clone https://github.com/trijitroy2006/Trijit-Roy-Prediction-AI.git`
2. Set up virtual environment: `python -m venv venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Add API Key: Create a `.env` file and add `GEMINI_API_KEY=your_key_here`
5. Run: `python -m streamlit run app_streamlit.py`

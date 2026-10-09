import streamlit as st
import pandas as pd
from milestone_1 import market_analysis
import textwrap
from milestone_3 import database



from milestone_3.recommendation_engine import generate_recommendations
from milestone_3.mitigation_engine import generate_mitigation
from milestone_3.improvement_engine import generate_improvements
from milestone_4.llm_service import generate_llm_recommendations, generate_llm_mitigation, generate_llm_improvements

try:
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict, Any 
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    from typing import TypedDict, Any 


st.set_page_config(page_title="Prediction AI", layout="wide", initial_sidebar_state="collapsed")

if 'has_analyzed' not in st.session_state:
    st.session_state['has_analyzed'] = False


# --- GLOBAL FONT SIZE OVERRIDE ---
st.markdown('''
<style>
    /* Increase base font size for native Streamlit widgets */
    .stTextInput label, .stSelectbox label, .stNumberInput label, .stTextArea label {
        font-size: 18px !important;
    }
    .stTextInput input, .stSelectbox select, .stNumberInput input, .stTextArea textarea {
        font-size: 16px !important;
    }
    .stMarkdown p, .stMarkdown li {
        font-size: 18px !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
    }
</style>
''', unsafe_allow_html=True)


# Custom CSS to mimic the Mac-style window from the PDF
st.markdown("""
<style>
/* Main container styling to look like a window */
.main .block-container {
border-radius: 12px;
box-shadow: 0 10px 25px rgba(37, 99, 235, 0.1);
padding: 2rem !important;
margin-top: 3rem;
margin-bottom: 3rem;
border: 1px solid #BFDBFE;
}

/* Mac window controls (Red, Yellow, Green dots) */
.mac-controls {
display: flex;
gap: 8px;
margin-bottom: 20px;
}
.mac-dot {
width: 12px;
height: 12px;
border-radius: 50%;
}
.mac-red { background-color: #ff5f56; }
.mac-yellow { background-color: #ffbd2e; }
.mac-green { background-color: #27c93f; }

/* Customizing the tabs to look more like the PDF */
.stTabs [data-baseweb="tab-list"] {
gap: 24px;
border-bottom: 1px solid #e5e7eb;
}
.stTabs [data-baseweb="tab"] {
height: 50px;
white-space: pre-wrap;
background-color: transparent;
border-radius: 4px 4px 0 0;
gap: 1px;
padding-top: 10px;
padding-bottom: 10px;
font-weight: 600;
color: #6b7280;
}
.stTabs [aria-selected="true"] {
color: #111827;
border-bottom: 2px solid #3b82f6;
}

/* Adjusting headers */
h1 {
font-size: 2.2rem !important;
padding-bottom: 0 !important;
}

</style>
""", unsafe_allow_html=True)

# Injecting the Mac dots at the top of the container
st.markdown("""
<h1 style="margin: 0; padding: 0; font-size: 32px; color: #111827; font-weight: 700; font-family: sans-serif; margin-bottom: 14px; margin-top: -16px;">Prediction AI</h1>
<div class="mac-controls">
<div class="mac-dot mac-red"></div>
<div class="mac-dot mac-yellow"></div>
<div class="mac-dot mac-green"></div>
</div>
""", unsafe_allow_html=True)



tab1, tab2, tab3, tab4 = st.tabs(["Project Input", "Risk Assessment", "Recommendations", "Dashboard"])

#M3 langgraph workflow
class M3WorkflowState(TypedDict, total=False):
    project_data: dict
    market_data: dict
    risk_input_data: dict
    risk_score: int
    risk_status: str
    success_probability: int
    swot: dict 
    feasibility_score: int 
    risk_data: list 
    recommendations: dict
    mitigation_results: list
    improvement_results: list 
    final_response: dict 

def analyze_project(state: M3WorkflowState):
    #collect project & market info
    project_data = state.get("project_data", {})
    market_data = market_analysis.get_market_data(
        project_data.get("industry", "Technology"), 
        project_data.get("target_market", ""),
        project_data.get("budget", 0)
    )

    return {
        "project_data": project_data,
        "market_data": market_data
    }

def analyze_risks(state: M3WorkflowState):
    """Node 2: Prepare the risk and feasibility context."""

    return {
        "risk_input_data": state["risk_input_data"],
        "risk_score": state["risk_score"],
        "risk_status": state["risk_status"],
        "success_probability": state["success_probability"],
        "swot": state["swot"],
        "feasibility_score": state["feasibility_score"],
        "risk_data": state["risk_data"]
    }


def generate_recommendation_node(state: M3WorkflowState):
    """Node 3: Generate strategic recommendations using Gemini."""
    
    baseline_recommendations = generate_recommendations(
        state["project_data"],
        state["risk_input_data"],
        state["swot"],
        state["feasibility_score"]
    )

    llm_recommendations = generate_llm_recommendations(
        project_data=state["project_data"],
        risk_input_data=state["risk_input_data"],
        swot=state["swot"],
        feasibility_score=state["feasibility_score"],
        market_data=state.get("market_data", {}),
        risk_data=state.get("risk_data", []),
        base_recommendations=baseline_recommendations
    )

    return {
        "recommendations": llm_recommendations
    }

def generate_mitigation_node(state: M3WorkflowState):
    """Node 4: Generate risk mitigation strategies using Gemini."""

    baseline_mitigation = generate_mitigation(
        state["risk_data"]
    )
    
    llm_mitigation = generate_llm_mitigation(
        project_data=state["project_data"],
        risk_data=state["risk_data"],
        risk_input_data=state["risk_input_data"],
        swot=state["swot"],
        feasibility_score=state["feasibility_score"],
        base_mitigation=baseline_mitigation
    )

    return {
        "mitigation_results": llm_mitigation
    }


def generate_improvements_node(state: M3WorkflowState):
    """Node 5: Generate project improvement suggestions using Gemini."""

    baseline_improvements = generate_improvements(
        state["project_data"],
        state["risk_input_data"],
        state["swot"],
        state["feasibility_score"],
        market_data=state.get("market_data", {}),
        mitigation_results=state.get("mitigation_results", [])
    )
    
    llm_improvements = generate_llm_improvements(
        project_data=state["project_data"],
        risk_input_data=state["risk_input_data"],
        swot=state["swot"],
        feasibility_score=state["feasibility_score"],
        market_data=state.get("market_data", {}),
        risk_data=state.get("risk_data", []),
        mitigation_results=state.get("mitigation_results", []),
        base_improvements=baseline_improvements
    )

    return {
        "improvement_results": llm_improvements
    }

def generate_final_response_node(state: M3WorkflowState):
    """Node 6: Combine all M3 outputs into the final strategic response."""

    project = state.get("project_data", {})
    recommendations = state.get("recommendations", {})
    mitigation = state.get("mitigation_results", [])
    improvements = state.get("improvement_results", [])

    final_response = {
        "project_summary": {
            "project_name": project.get("startup_name", "Unknown"),
            "industry": project.get("industry", "Unknown"),
            "business_model": project.get("business_model", "Unknown"),
            "target_market": project.get("target_market", "Unknown"),
            "budget": project.get("budget", 0),
            "description": project.get("project_description", "")
        },

        "risk_summary": {
            "risk_score": state.get("risk_score", 0),
            "risk_status": state.get("risk_status", "UNKNOWN"),
            "success_probability": state.get("success_probability", 0),
            "feasibility_score": state.get("feasibility_score", 0)
        },

        "key_strategic_recommendations": recommendations.get(
            "recommendations", []
        ),

        "risk_based_recommendations": [
            rec for rec in recommendations.get("recommendations", [])
            if rec.get("category") == "Risk"
        ],

        "mitigation_strategies": mitigation,

        "improvement_suggestions": improvements,

        "short_term_action_plan": recommendations.get(
            "short_term_action_plan", []
        ),

        "long_term_action_plan": recommendations.get(
            "long_term_action_plan", []
        ),

        "final_strategic_assessment": (
            recommendations.get(
                "overall_strategic_recommendation",
                "Review the identified risks and implement the highest-priority mitigation actions."
            )
        )
    }

    return {
        "final_response": final_response
    }


def build_m3_workflow():
    """Build the required six-node LangGraph workflow."""

    if not LANGGRAPH_AVAILABLE:
        return None

    workflow = StateGraph(M3WorkflowState)

    workflow.add_node("analyze_project", analyze_project)
    workflow.add_node("analyze_risks", analyze_risks)
    workflow.add_node(
        "generate_recommendations",
        generate_recommendation_node
    )
    workflow.add_node(
        "generate_mitigation",
        generate_mitigation_node
    )
    workflow.add_node(
        "generate_improvements",
        generate_improvements_node
    )
    workflow.add_node(
        "generate_final_response",
        generate_final_response_node
    )

    workflow.add_edge(START, "analyze_project")
    workflow.add_edge("analyze_project", "analyze_risks")
    workflow.add_edge("analyze_risks", "generate_recommendations")
    workflow.add_edge(
        "generate_recommendations",
        "generate_mitigation"
    )
    workflow.add_edge(
        "generate_mitigation",
        "generate_improvements"
    )
    workflow.add_edge(
        "generate_improvements",
        "generate_final_response"
    )
    workflow.add_edge("generate_final_response", END)

    return workflow.compile()



with tab1:
    st.markdown("""
<h1 style="margin: 0; padding: 0; font-size: 32px; color: #111827; font-weight: 700; font-family: sans-serif;">Data Collection & Market Intelligence</h1>
<p style="margin: 4px 0 24px 0; color: #6B7280; font-size: 19px; font-family: sans-serif;">Gather project data and analyze market landscape</p>
""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Project Submission")
        
        with st.form("project_form"):
            startup_name = st.text_input("Startup/Project Name", key="input_startup_name", placeholder="e.g., TechVenture AI")
            industry = st.selectbox("Industry/Sector", ["Technology", "Healthcare", "Finance", "Education"], index=None, placeholder="Select an industry...", key="input_industry")
            business_model = st.selectbox("Business Model", ["SaaS", "B2B", "B2C", "Marketplace"], index=None, placeholder="Select a business model...", key="input_business_model")
            target_market = st.text_input("Target Market", placeholder="e.g., SMBs", key="input_target_market")
            budget = st.number_input("Budget (USD)", min_value=0.0, value=None, step=10000.0, key="input_budget")
            description = st.text_area("Project Description", placeholder="Brief description of your project idea...", key="input_description")
            
            submitted = st.form_submit_button("Analyze Project")
            
        if submitted:
            project_data = {
                "startup_name": startup_name,
                "industry": industry,
                "business_model": business_model,
                "target_market": target_market,
                "budget": budget,
                "project_description": description
            }

            #this will store the submitted projeect immediately
            st.session_state["project_data"] = project_data
            st.session_state["assessment_saved"] = False
            
            with st.spinner("Generating AI Analysis Results..."):
                try:
                    from milestone_4.llm_service import generate_project_analysis
                    st.session_state['analysis_results'] = generate_project_analysis(project_data)
                except Exception as e:
                    st.error(f"Failed to load AI Engine: {e}")
                    st.session_state['analysis_results'] = None
                    
            st.session_state['has_analyzed'] = True

            #now to save it to postgresql
            try:
                project_id = database.insert_project(project_data)
                st.session_state["project_id"] = project_id
                st.success("Project analyzed successfully.")
            except Exception as error:
                st.warning("Project analysis is available, but the database could not be updated.")
                st.error(f"Database error: {error}")

    with col2:
        st.subheader("Market Analysis")
        if st.session_state.get('has_analyzed', False):
            data = st.session_state['project_data']
            m_data = market_analysis.get_market_data(data['industry'], data['target_market'], data['budget'])
            
            st.write("**Market Size & Growth Rate**")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("TAM", m_data['TAM']['value'], m_data['TAM']['growth'])
            m2.metric("SAM", m_data['SAM']['value'], m_data['SAM']['growth'])
            m3.metric("SOM", m_data['SOM']['value'], m_data['SOM']['growth'])
            
            st.write("**Market Trends (2020-2026)**")
            chart_data = pd.DataFrame(
                [30, 40, 45, 55, 70, 85, 100],
                columns=["Trend"],
                index=["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
            )
            st.bar_chart(chart_data)
            
        else:
            st.info("Submit a project idea on the left to view the market analysis.")

    with col3:
        st.subheader("Competitor Landscape")
        if st.session_state.get('has_analyzed', False):
            data = st.session_state['project_data']
            c_data = market_analysis.get_competitor_data(data['startup_name'], data['industry'], data['business_model'])
            
            for comp in c_data:
                with st.container():
                    st.markdown(f"**{comp['name']}** `{comp['type']}`")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Market Share", comp['market_share'])
                    c2.metric("Revenue", comp['revenue'])
                    c3.metric("Growth", comp['growth'])
                    st.progress(comp['position'] / 100, text="Market Position")
                    st.divider()
                    
        else:
            st.info("Submit a project idea to generate competitor insights.")

with tab2:
    st.markdown("""
<h1 style="margin: 0; padding: 0; font-size: 32px; color: #111827; font-weight: 700; font-family: sans-serif;">Risk Assessment & SWOT Analysis</h1>
<p style="margin: 4px 0 24px 0; color: #6B7280; font-size: 19px; font-family: sans-serif;">AI-powered risk scoring and strategic evaluation</p>
""", unsafe_allow_html=True)

    if not st.session_state.get('has_analyzed', False):
        st.info("💡 Please enter project details and click 'Analyze Project' to view Risk Assessment.")
    else:
        data = st.session_state.get('project_data', {})


        st.subheader("Risk Assessment Inputs")

        col1, col2 = st.columns(2)

        with col1:
            market_competition = st.selectbox("Market Competition", ["Low", "Medium", "High"])
            team_expertise = st.selectbox("Team Expertise", ["Low", "Medium", "High"])
            resource_availability = st.selectbox("Resource Availability", ["Limited", "Moderate", "Good"])

        with col2:
            innovation_level = st.selectbox("Innovation Level", ["Low", "Medium", "High"])
            market_research = st.selectbox("Market Research Quality", ["Limited", "Moderate", "Strong"])

        st.subheader("Project Feasibility Inputs")

        col1, col2 = st.columns(2)

        with col1:
            market_opportunity = st.slider("Market Opportunity Score", 0, 100, 50)
            team_capability = st.slider("Team Capability Score", 0, 100, 50)

        with col2:
            competitive_advantage = st.slider("Competitive Advantage Score", 0, 100, 50)
            resource_score = st.slider("Resource Availability Score", 0, 100, 50)



        # Empty State Handling
        if not (st.session_state.get('has_analyzed', False) or 'risk_assessment_data' in st.session_state):
            st.info("💡 Complete project details on the Project Input tab and click 'Analyze Project', or click 'Evaluate Risk' below to generate risk insights.")
            if st.button("Evaluate Risk"):
                st.session_state['has_analyzed'] = True
                st.rerun()

        if st.session_state.get('has_analyzed', False) or 'risk_assessment_data' in st.session_state:
            from milestone_2.risk_engine import calculate_risk, get_risk_status, calculate_success_probability
    #         from milestone_3.mitigation_engine import generate_mitigation
    #         from milestone_3.improvement_engine import generate_improvements
            from milestone_2.swot_analysis import generate_swot
            from milestone_2.feasibility import calculate_feasibility
            from milestone_1 import market_analysis
            from milestone_3 import database

            def calculate_risk_and_swot(mc, te, ra, il, mr, mo, tc, ca, rs):
                r_score = calculate_risk(mc, te, ra, il, mr)
                r_status = get_risk_status(r_score)
                s_prob = calculate_success_probability(r_score)

                r_data = [
                    {
                        "risk_category": "Market",
                        "risk_score": 80 if mc == "High" else 50,
                        "risk_description": "High competitor density",
                        "priority_level": "High" if mc == "High" else "Medium"
                    },
                    {
                        "risk_category": "Financial",
                        "risk_score": 75 if data.get("budget", 0) < 50000 else 45,
                        "risk_description": "Budget constraints",
                        "priority_level": "High" if data.get("budget", 0) < 50000 else "Medium"
                    },
                    {
                        "risk_category": "Technical",
                        "risk_score": 80 if te == "Low" else 40,
                        "risk_description": "Limited technical expertise",
                        "priority_level": "High" if te == "Low" else "Medium"
                    }
                ]

                sw = generate_swot(te, il, mc, ra, mr)
                f_score = calculate_feasibility(mo, tc, ca, rs)

                return {
                    'risk_score': r_score,
                    'risk_status': r_status,
                    'success_probability': s_prob,
                    'risk_data': r_data,
                    'swot': sw,
                    'feasibility_score': f_score
                }

            # Update state dynamically with current inputs
            st.session_state['risk_assessment_data'] = calculate_risk_and_swot(
                market_competition, team_expertise, resource_availability, 
                innovation_level, market_research, 
                market_opportunity, team_capability, 
                competitive_advantage, resource_score
            )

            # Read dynamically from session state
            results = st.session_state['risk_assessment_data']
            risk_score = results['risk_score']
            risk_status = results['risk_status']
            success_probability = results['success_probability']
            risk_data = results['risk_data']
            swot = results['swot']
            feasibility_score = results['feasibility_score']

            mitigation_results = generate_mitigation(risk_data)
            market_data_for_improvements = market_analysis.get_market_data(
                data.get("industry", "Technology"),
                data.get("target_market", ""),
                data.get("budget", 0),
            )

            from milestone_2.swot_analysis import generate_swot
            swot = generate_swot(team_expertise, innovation_level, market_competition, resource_availability, market_research)

            from milestone_2.feasibility import calculate_feasibility
            feasibility_score = calculate_feasibility(market_opportunity, team_capability, competitive_advantage, resource_score)

            risk_input_data = {
                "market_competition": market_competition,
                "team_expertise": team_expertise,
                "resource_availability": resource_availability,
                "innovation_level": innovation_level,
                "market_research": market_research,
                "risk_score": risk_score
            }

                # ============================================================
                # M3 - STRATEGIC RECOMMENDATIONS
                # ============================================================

            recommendation_results = generate_recommendations(
                data,
                risk_input_data,
                swot,
                feasibility_score
            )

                # ============================================================
                # M3 - IMPROVEMENT PLAN
                # ============================================================

            improvement_results = generate_improvements(
                data,
                risk_input_data,
                swot,
                feasibility_score,
                market_data=market_data_for_improvements,
                mitigation_results=mitigation_results
            )

            project_id = st.session_state.get("project_id")
            if project_id and not st.session_state.get("assessment_saved", False):
                try:
                    database.save_assessment(
                        project_id=project_id,
                        swot_data=swot,
                        risk_score=risk_score,
                        risk_status=risk_status,
                        success_probability=success_probability,
                        recommendations=recommendation_results["recommendations"],
                    )
                    st.session_state["assessment_saved"] = True
                except Exception as error:
                    print(f"Could not save the assessment to the database: {error}")

            # Format SWOT bullets as HTML dots
            def format_swot(items):
                return "".join([f'<div style="margin-bottom:4px;">• {item}</div>' for item in items])
                st.markdown("<br>", unsafe_allow_html=True)

            # ============================================================
            # M2 RESULTS DASHBOARD
            # ============================================================

            st.divider()
            st.subheader("Risk Assessment Results")

            # ------------------------------------------------------------
            # TOP SUMMARY
            # ------------------------------------------------------------

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric(
                    "Overall Risk Score",
                    f"{risk_score}/100",
                    risk_status
                )

            with summary_col2:
                st.metric(
                    "Success Probability",
                    f"{success_probability}%"
                )

            with summary_col3:
                st.metric(
                    "Feasibility Score",
                    f"{feasibility_score}%"
                )

            st.divider()

            # ------------------------------------------------------------
            # MAIN RESULTS AREA
            # LEFT = KEY RISKS
            # RIGHT = SWOT
            # ------------------------------------------------------------

            left_col, right_col = st.columns([1, 2])

            # ============================================================
            # LEFT COLUMN - KEY RISK FACTORS
            # ============================================================

            with left_col:

                st.subheader("Key Risk Factors")

                st.markdown(
                    f"""
                    **👥 Team Expertise**

                    {team_expertise} technical experience
                    """
                )

                st.markdown(
                    f"""
                    **💡 Innovation Level**

                    {innovation_level} innovation potential
                    """
                )

                st.markdown(
                    f"""
                    **📊 Market Competition**

                    {market_competition} competition
                    """
                )

                st.markdown(
                    f"""
                    **📦 Resource Availability**

                    {resource_availability}
                    """
                )

                st.markdown(
                    f"""
                    **🔎 Market Research**

                    {market_research}
                    """
                )

                # Risk status card
                if risk_status == "HIGH RISK":
                    st.error(f"⚠️ **{risk_status}**")
                elif risk_status == "MEDIUM RISK":
                    st.warning(f"⚠️ **{risk_status}**")
                else:
                    st.success(f"✓ **{risk_status}**")

            # ============================================================
            # RIGHT COLUMN - SWOT
            # ============================================================

            with right_col:

                st.subheader("SWOT Analysis")

                swot_col1, swot_col2 = st.columns(2)

                with swot_col1:

                    st.success("### 💪 Strengths")

                    for item in swot["Strengths"]:
                        st.markdown(f"- {item}")

                    st.info("### 🚀 Opportunities")

                    for item in swot["Opportunities"]:
                        st.markdown(f"- {item}")

                with swot_col2:

                    st.error("### ⚠️ Weaknesses")

                    for item in swot["Weaknesses"]:
                        st.markdown(f"- {item}")

                    st.warning("### 🔥 Threats")

                    for item in swot["Threats"]:
                        st.markdown(f"- {item}")

            st.divider()

            # ------------------------------------------------------------
            # PROJECT FEASIBILITY
            # ------------------------------------------------------------

            st.subheader("Project Feasibility")

            feasibility_col1, feasibility_col2 = st.columns([1, 2])

            with feasibility_col1:

                st.metric(
                    "Feasibility Score",
                    f"{feasibility_score}%"
                )

                if feasibility_score >= 70:
                    st.success("Good Feasibility")
                elif feasibility_score >= 40:
                    st.warning("Moderate Feasibility")
                else:
                    st.error("Low Feasibility")

            with feasibility_col2:

                st.write("**Assessment Metrics**")

                st.progress(
                    team_capability / 100,
                    text=f"Team Capability — {team_capability}%"
                )

                st.progress(
                    competitive_advantage / 100,
                    text=f"Competitive Advantage — {competitive_advantage}%"
                )

                st.progress(
                    resource_score / 100,
                    text=f"Resource Availability — {resource_score}%"
                )

                st.progress(
                    market_opportunity / 100,
                    text=f"Market Opportunity — {market_opportunity}%"
                )


with tab3:
    st.markdown("""
<h1 style="margin: 0; padding: 0; font-size: 32px; color: #111827; font-weight: 700; font-family: sans-serif;">Recommendations & Strategic Reasoning</h1>
<p style="margin: 4px 0 24px 0; color: #6B7280; font-size: 19px; font-family: sans-serif;">AI-powered mitigation strategies and agent workflows</p>
""", unsafe_allow_html=True)

    if not st.session_state.get('has_analyzed', False):
        st.info("💡 Please enter project details and click 'Analyze Project' to view AI Recommendations.")
    else:
        data = st.session_state.get('project_data', {})


        recommendation_tab, risk_mitigation_tab, improvements_tab, langgraph_tab = st.tabs([
            "Strategic Recommendations",
            "Risk Mitigation",
            "Improvements",
            "LangGraph Agent"
        ])

        # ============================================================
        # STRATEGIC RECOMMENDATIONS
        # ============================================================

        with recommendation_tab:

            st.subheader("AI Strategic Recommendations")

            if recommendation_results:

                overall = recommendation_results.get(
                    "overall_strategic_recommendation",
                    "Focus on reducing the highest-priority project risks."
                )

                st.info(overall)

                recommendations = recommendation_results.get(
                    "recommendations",
                    []
                )

                if recommendations:

                    for rec in recommendations:

                        priority = rec.get("priority", "Medium")
                        category = rec.get("category", "Strategy")
                        title = rec.get("title", "Recommendation")

                        st.markdown(
                            f"### {title}"
                        )

                        st.markdown(
                            f"**Category:** {category}  \n"
                            f"**Priority:** {priority}"
                        )

                        st.markdown(
                            f"**Problem:** {rec.get('problem', 'N/A')}"
                        )

                        if rec.get("explanation"):
                            st.markdown(
                                f"**Why it matters:** {rec['explanation']}"
                            )

                        st.markdown(
                            f"**Recommended Action:** "
                            f"{rec.get('action', 'N/A')}"
                        )

                        st.markdown(
                            f"**Risk Reduction:** "
                            f"{rec.get('risk_reduction', 'N/A')}"
                        )

                        st.divider()

                else:
                    st.warning("No strategic recommendations were generated.")

                # Short-term plan
                st.subheader("Short-Term Action Plan")

                short_term = recommendation_results.get(
                "short_term_action_plan",
                []
                )

                for action in short_term:
                    st.markdown(f"- {action}")

                # Long-term plan
                st.subheader("Long-Term Action Plan")

                long_term = recommendation_results.get(
                    "long_term_action_plan",
                    []
                )

                for action in long_term:
                    st.markdown(f"- {action}")

            else:
                st.warning("No strategic recommendations available.")


        with improvements_tab:
            st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
            <h3 style="margin:0; font-size: 20px; color: #111827; font-family: sans-serif;">Improvements</h3>
            <span style="background: #6D28D9; color: white; padding: 2px 8px; border-radius: 12px; font-size: 14px; font-weight: bold; font-family: sans-serif;">AI Powered</span>
            </div>
        <style>
        .improvement-card-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 12px;
        }
        .improvement-flip-card {
        min-height: 238px;
        perspective: 1200px;
        font-family: sans-serif;
        }
        .improvement-flip-inner {
        position: relative;
        width: 100%;
        min-height: 238px;
        transition: transform 0.7s cubic-bezier(.2,.7,.2,1);
        transform-style: preserve-3d;
        }
        .improvement-flip-card:hover .improvement-flip-inner,
        .improvement-flip-card:focus-within .improvement-flip-inner {
        transform: rotateY(180deg);
        }
        .improvement-face {
        position: absolute;
        inset: 0;
        min-height: 238px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.72);
        border-radius: 16px;
        box-sizing: border-box;
        backface-visibility: hidden;
        -webkit-backface-visibility: hidden;
        box-shadow: 0 16px 32px rgba(31, 41, 55, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.8);
        overflow: hidden;
        }
        .improvement-front {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.82), rgba(239, 246, 255, 0.58));
        backdrop-filter: blur(18px) saturate(135%);
        -webkit-backdrop-filter: blur(18px) saturate(135%);
        }
        .improvement-back {
        background: linear-gradient(135deg, rgba(245, 243, 255, 0.94), rgba(224, 242, 254, 0.84));
        backdrop-filter: blur(18px) saturate(135%);
        -webkit-backdrop-filter: blur(18px) saturate(135%);
        transform: rotateY(180deg);
        overflow-y: auto;
        }
        .improvement-face h4 {
        margin: 0 0 10px 0;
        color: #111827;
        font-size: 19px;
        line-height: 1.35;
        }
        .improvement-face p,
        .improvement-face li {
        color: #4B5563;
        font-size: 16px;
        line-height: 1.55;
        }
        .improvement-face ul {
        margin: 6px 0 14px 18px;
        padding: 0;
        }
        .improvement-label {
        color: #6D28D9;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
        }
        .improvement-priority {
        float: right;
        color: #6D28D9;
        background: rgba(237, 233, 254, 0.9);
        border: 1px solid rgba(196, 181, 253, 0.7);
        border-radius: 999px;
        padding: 3px 9px;
        font-size: 14px;
        font-weight: 700;
        }
        .improvement-hint {
        margin-top: 16px;
        color: #9CA3AF;
        font-size: 14px;
        }
        @media (max-width: 900px) {
        .improvement-card-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        }
        @media (max-width: 620px) {
        .improvement-card-grid {
            grid-template-columns: 1fr;
        }
        }
        </style>
            """, unsafe_allow_html=True)

            improvement_cards = []
            for rec in improvement_results:
                steps = rec.get("steps") or [rec.get("action", "Review this project area and define a corrective action.")]
                steps_html = "".join(f"<li>{step}</li>" for step in steps)
                recommendation_html = "".join(
                    f"{index}. {step}<br>" for index, step in enumerate(steps, start=1)
                )
                card_html = f"""
        <div class="improvement-flip-card" tabindex="0">
          <div class="improvement-flip-inner">
        <div class="improvement-face improvement-front">
          <span class="improvement-label">{rec["category"]}</span>
          <span class="improvement-priority">{rec["priority"]}</span>
          <h4>{rec["title"]}</h4>
          <div class="improvement-label">Problem</div>
          <p>{rec["problem"]}</p>
        </div>
        <div class="improvement-face improvement-back">
          <div class="improvement-label">Solution Steps</div>
          <ul>{steps_html}</ul>
          <div class="improvement-label">Recommendation</div>
        <p>{recommendation_html}</p>
          <div class="improvement-label">Risk Reduction</div>
          <p>{rec["risk_reduction"]}</p>
        </div>
          </div>
        </div>
        """
                improvement_cards.append(textwrap.dedent(card_html))
            st.markdown(
                '<div class="improvement-card-grid">'
                + "".join(improvement_cards)
                + "</div>",
                unsafe_allow_html=True,
            )
        with risk_mitigation_tab:
            st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <h3 style="margin:0; font-size: 20px; color: #111827; font-family: sans-serif;">Risk Mitigation</h3>
        <span style="color: #6B7280;">&#128116;</span>
        </div>
        <style>
        .risk-card-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 12px;
        }
        .risk-glass-card {
        min-height: 220px;
        padding: 16px;
        box-sizing: border-box;
        border: 1px solid rgba(255, 255, 255, 0.58);
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.38), rgba(153, 246, 228, 0.2));
        backdrop-filter: blur(24px) saturate(165%);
        -webkit-backdrop-filter: blur(24px) saturate(165%);
        box-shadow: 0 14px 28px rgba(31, 41, 55, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.72), inset 0 -1px 0 rgba(255, 255, 255, 0.18);
        font-family: sans-serif;
        }
        .risk-glass-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 10px;
        }
        .risk-glass-card-risk {
        color: #DC2626;
        font-size: 16px;
        font-weight: 700;
        line-height: 1.4;
        }
        .risk-glass-card-impact {
        flex-shrink: 0;
        color: #047857;
        background: rgba(209, 250, 229, 0.86);
        border: 1px solid rgba(110, 231, 183, 0.7);
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 14px;
        font-weight: 700;
        }
        .risk-glass-card h4 {
        margin: 0 0 10px 0;
        color: #111827;
        font-size: 17px;
        line-height: 1.4;
        }
        .risk-glass-card p {
        margin: 0 0 8px 0;
        color: #4B5563;
        font-size: 15px;
        line-height: 1.5;
        }
        .risk-glass-card strong {
        color: #374151;
        }
        @media (max-width: 900px) {
        .risk-card-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        }
        @media (max-width: 620px) {
        .risk-card-grid {
            grid-template-columns: 1fr;
        }
        }
        </style>
        """,  unsafe_allow_html=True)

            selected_risk = st.pills(
                "Risk Category", 
                ["All Risks", "Financial", "Market", "Technical"], 
                default="All Risks", 
                label_visibility="collapsed",
                key="selected_risk_category"
            )

            if not selected_risk:
                selected_risk = "All Risks"

            mitigation_cards = []
            for mitigation in mitigation_results:

                category = mitigation["category"]

                # Apply selected category filter
                if selected_risk != "All Risks" and category != selected_risk:
                    continue

                card_html = f"""
        <div class="risk-glass-card">
        <div class="risk-glass-card-header">
            <div class="risk-glass-card-risk">&#9888; {mitigation["risk"]}</div>
            <span class="risk-glass-card-impact">{mitigation["impact"]} Impact</span>
        </div>
        <h4>
            {mitigation["mitigation_strategy"]}
        </h4>
        <p><strong>Preventive Action:</strong> {mitigation["preventive_action"]}</p>
        <p><strong>Contingency Action:</strong> {mitigation["contingency_action"]}</p>
        </div>
        """
                mitigation_cards.append(textwrap.dedent(card_html))
            st.markdown(
                '<div class="risk-card-grid">'
                + "".join(mitigation_cards)
                + "</div>",
                unsafe_allow_html=True,
            )
        with langgraph_tab:
            st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <h3 style="margin:0; font-size: 20px; color: #111827; font-family: sans-serif;">LangGraph Agent</h3>
        <span style="color: #8B5CF6;">&#9881;</span>
        </div>
        """, unsafe_allow_html=True)


            # ========================================================
            # ACTUAL LANGGRAPH WORKFLOW
            # ========================================================

            run_agent_workflow = st.button(
                "Run LangGraph Agent Workflow",
                use_container_width=True,
                type="primary"
            )

            if run_agent_workflow:

                if not LANGGRAPH_AVAILABLE:
                    st.error(
                        "LangGraph is not installed. "
                        "Install it with: pip install langgraph"
                    )

                else:
                    with st.spinner("Running M3 LangGraph workflow..."):

                        workflow = build_m3_workflow()

                        initial_state = {
                            "project_data": data,
                            "risk_input_data": risk_input_data,
                            "risk_score": risk_score,
                            "risk_status": risk_status,
                            "success_probability": success_probability,
                            "swot": swot,
                            "feasibility_score": feasibility_score,
                            "risk_data": risk_data
                        }

                        workflow_result = workflow.invoke(
                            initial_state
                        )

                        st.session_state["m3_workflow_result"] = workflow_result

                    st.success(
                        "LangGraph workflow completed successfully."
                    )

            # ========================================================
            # FINAL STRATEGIC RESPONSE
            # ========================================================

            workflow_result = st.session_state.get(
                "m3_workflow_result"
            )

            if workflow_result:

                final_response = workflow_result.get(
                    "final_response",
                    {}
                )

                st.divider()
                st.subheader("Final Strategic Response")

                # Project Summary
                st.markdown("### PROJECT SUMMARY")

                project_summary = final_response.get(
                    "project_summary",
                    {}
                )

                st.write(
                    f"**Project:** "
                    f"{project_summary.get('project_name', 'N/A')}"
                )

                st.write(
                    f"**Industry:** "
                    f"{project_summary.get('industry', 'N/A')}"
                )

                st.write(
                    f"**Target Market:** "
                    f"{project_summary.get('target_market', 'N/A')}"
                )

                # Risk Summary
                st.markdown("### RISK SUMMARY")

                risk_summary = final_response.get(
                    "risk_summary",
                    {}
                )

                r1, r2, r3, r4 = st.columns(4)

                r1.metric(
                    "Risk Score",
                    risk_summary.get("risk_score", 0)
                )

                r2.metric(
                    "Risk Status",
                    risk_summary.get("risk_status", "N/A")
                )

                r3.metric(
                    "Success Probability",
                    f"{risk_summary.get('success_probability', 0)}%"
                )

                r4.metric(
                    "Feasibility",
                    f"{risk_summary.get('feasibility_score', 0)}%"
                )

                # Strategic Recommendations
                st.markdown(
                    "### KEY STRATEGIC RECOMMENDATIONS"
                )

                for rec in final_response.get(
                    "key_strategic_recommendations",
                    []
                ):
                    st.markdown(
                        f"**{rec.get('title', 'Recommendation')}** "
                        f"- {rec.get('action', 'N/A')}"
                    )

                # Risk-Based Recommendations
                st.markdown(
                    "### RISK-BASED RECOMMENDATIONS"
                )

                risk_recommendations = final_response.get(
                    "risk_based_recommendations",
                    []
                )

                if risk_recommendations:
                    for rec in risk_recommendations:
                        st.markdown(
                            f"- **{rec.get('title', 'Risk Recommendation')}**: "
                            f"{rec.get('action', 'N/A')}"
                        )
                else:
                    st.write(
                        "No separate risk-category recommendations."
                    )

                # Mitigation
                st.markdown(
                    "### MITIGATION STRATEGIES"
                )

                for mitigation in final_response.get(
                    "mitigation_strategies",
                    []
                ):
                    st.markdown(
                        f"**{mitigation.get('risk', 'Risk')}**"
                    )
                    st.write(
                        mitigation.get(
                            "mitigation_strategy",
                            "N/A"
                        )
                    )

                # Improvements
                st.markdown(
                    "### IMPROVEMENT SUGGESTIONS"
                )

                for improvement in final_response.get(
                    "improvement_suggestions",
                    []
                ):
                    st.markdown(
                        f"**{improvement.get('title', 'Improvement')}** "
                        f"({improvement.get('priority', 'Medium')})"
                    )

                    st.write(
                        improvement.get(
                            "improvement",
                            improvement.get(
                                "risk_reduction",
                                "N/A"
                            )
                        )
                    )

                # Short-Term Plan
                st.markdown(
                    "### SHORT-TERM ACTION PLAN"
                )

                for action in final_response.get(
                    "short_term_action_plan",
                    []
                ):
                    st.markdown(f"- {action}")

                # Long-Term Plan
                st.markdown(
                    "### LONG-TERM ACTION PLAN"
                )

                for action in final_response.get(
                    "long_term_action_plan",
                    []
                ):
                    st.markdown(f"- {action}")

                # Final Assessment
                st.markdown(
                    "### FINAL STRATEGIC ASSESSMENT"
                )

                st.info(
                    final_response.get(
                        "final_strategic_assessment",
                        "No final assessment available."
                    )
                )




with tab4:
    st.markdown('''
<h1 style="margin: 0; padding: 0; font-size: 32px; color: #111827; font-weight: 700; font-family: sans-serif; margin-bottom: 4px;">Dashboard & Deployment</h1>
<p style="margin: 0 0 24px 0; color: #6B7280; font-size: 19px; font-family: sans-serif;">Risk analytics dashboard and comprehensive assessment reports</p>
''', unsafe_allow_html=True)

    if not st.session_state.get('has_analyzed', False):
        st.info("💡 Please enter project details and click 'Analyze Project' to view the Dashboard.")
    else:
        project = st.session_state.get("project_data", {})
        # Grab local variables calculated in earlier tabs if they exist
        mitigation_results = locals().get(
            "mitigation_results",
            st.session_state.get("m3_workflow_result", {}).get("mitigation_results", [])
        )
        improvement_results = locals().get(
            "improvement_results",
            st.session_state.get("m3_workflow_result", {}).get("improvement_results", [])
        )
        recommendations = locals().get(
            "recommendation_results",
            st.session_state.get("m3_workflow_result", {}).get("recommendations", {})
        )
        final_response = locals().get(
            "final_response",
            st.session_state.get("m3_workflow_result", {}).get("final_response", {})
        )

        # ========================================================
        # PROJECT OVERVIEW
        # ========================================================

        st.subheader("Project Overview")

        overview_col1, overview_col2, overview_col3 = st.columns(3)

        with overview_col1:
            st.markdown(
                f"""
                **Project**

                {project.get("startup_name", "N/A")}

                **Industry**

                {project.get("industry", "N/A")}
                """
            )

        with overview_col2:
            st.markdown(
                f"""
                **Business Model**

                {project.get("business_model", "N/A")}

                **Target Market**

                {project.get("target_market", "N/A")}
                """
            )

        with overview_col3:
            budget = project.get("budget", 0)

            try:
                budget_display = f"${float(budget):,.0f}"
            except (TypeError, ValueError):
                budget_display = str(budget)

            st.markdown(
                f"""
                **Budget**

                {budget_display}

                **Project Status**

                {"Analysis Complete" if st.session_state.get("has_analyzed")
                else "Pending Analysis"}
                """
            )

        description = project.get(
            "project_description",
            ""
        )

        if description:
            st.markdown("**Project Description**")
            st.info(description)

        st.divider()

        data = st.session_state.get('project_data', {})

        data = st.session_state.get('project_data', {})
        industry = data.get('industry', 'Technology')
        budget = data.get('budget', 100000)
        # Dynamically map from actual AI workflow state
        rad = st.session_state.get('risk_assessment_data', {})
        workflow_result = st.session_state.get('m3_workflow_result', {})
        rec_res = workflow_result.get('recommendations', {})
        final_res = workflow_result.get('final_response', {})

        overall_risk = int(rad.get('risk_score', 50))
        success_prob = int(rad.get('success_probability', 50))
        
        risk_data_arr = rad.get('risk_data', [])
        market_risk = int(next((r.get('risk_score', 50) for r in risk_data_arr if r.get('risk_category') == 'Market'), 50))
        tech_risk = int(next((r.get('risk_score', 50) for r in risk_data_arr if r.get('risk_category') == 'Technical'), 50))
        
        # Recommendations
        recs = []
        if isinstance(rec_res, list):
            recs = [r.get('action', r.get('title', '')) for r in rec_res]
        elif isinstance(rec_res, dict) and isinstance(rec_res.get('recommendations'), list):
            recs = [r.get('action', r.get('title', '')) for r in rec_res['recommendations']]
        
        # Key Findings
        kf = [
            {"title": "Overall Feasibility", "desc": f"Feasibility Score is {rad.get('feasibility_score', 0)}%"},
            {"title": "Risk Status", "desc": rad.get('risk_status', 'Unknown')}
        ]
        
        # Risk Assessment Details
        ra = [{"title": r.get('risk_category', 'Risk'), "desc": r.get('risk_description', '')} for r in risk_data_arr]
        
        # Next Steps
        ns = []
        if isinstance(final_res, dict) and final_res.get("short_term_action_plan"):
            ns = [a.get("action", "") if isinstance(a, dict) else a for a in final_res.get("short_term_action_plan", [])]
        elif isinstance(rec_res, dict) and rec_res.get("short_term_action_plan"):
            ns = [a for a in rec_res.get("short_term_action_plan", [])]
        else:
            ns = recs[:3]
            ns = recs[:3]

        report = {
            "overall_risk": overall_risk,
            "success_prob": success_prob,
            "market_risk": market_risk,
            "tech_risk": tech_risk,
            "key_findings": kf,
            "risk_assessment": ra,
            "recommendations": recs,
            "funding_strategy": final_res.get("project_summary", {}).get("target_market", "Focus on core market penetration and user acquisition."),
            "tech_advantage": final_res.get("project_summary", {}).get("business_model", "Leverage existing infrastructure for rapid scaling."),
            "next_steps": ns
        }

        def safe_get(lst, idx, key=None):
            if idx < len(lst):
                return lst[idx].get(key, '') if key else lst[idx]
            return "N/A"


        col1, col2, col3 = st.columns([1, 1.5, 1])

        with col1:
            st.markdown('<div style="font-weight: 600; color: #374151; font-size: 18px; margin-bottom: 12px; font-family: sans-serif;">Risk Analytics</div>', unsafe_allow_html=True)

            html_metrics = f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; font-family: sans-serif;">
                <div style="border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px; text-align: center; background: white;">
                    <div style="color: #6B7280; font-size: 15px; font-weight: 500; margin-bottom: 8px;">Overall Risk</div>
                    <div style="color: #EF4444; font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 8px;">{overall_risk}%</div>
                    <div style="color: #10B981; font-size: 14px; font-weight: 600;">&uarr; +5%</div>
                </div>
                <div style="border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px; text-align: center; background: white;">
                    <div style="color: #6B7280; font-size: 15px; font-weight: 500; margin-bottom: 8px;">Success Prob.</div>
                    <div style="color: #F59E0B; font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 8px;">{success_prob}%</div>
                    <div style="color: #EF4444; font-size: 14px; font-weight: 600;">&darr; -8%</div>
                </div>
                <div style="border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px; text-align: center; background: white;">
                    <div style="color: #6B7280; font-size: 15px; font-weight: 500; margin-bottom: 8px;">Market Risk</div>
                    <div style="color: #EF4444; font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 8px;">{market_risk}%</div>
                    <div style="color: #10B981; font-size: 14px; font-weight: 600;">&uarr; +12%</div>
                </div>
                <div style="border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px; text-align: center; background: white;">
                    <div style="color: #6B7280; font-size: 15px; font-weight: 500; margin-bottom: 8px;">Tech Risk</div>
                    <div style="color: #F59E0B; font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 8px;">{tech_risk}%</div>
                    <div style="color: #EF4444; font-size: 14px; font-weight: 600;">&darr; -3%</div>
                </div>
            </div>
            """
            st.markdown(html_metrics, unsafe_allow_html=True)

            st.markdown('<div style="font-weight: 600; color: #374151; font-size: 16px; margin-top: 12px; margin-bottom: 8px; font-family: sans-serif;">Risk Trend (6 Months)</div>', unsafe_allow_html=True)

            # Create realistic trend data ending exactly at the calculated overall_risk
            trend_values = [
                max(0, min(100, overall_risk + 12)),
                max(0, min(100, overall_risk + 5)),
                max(0, min(100, overall_risk + 8)),
                max(0, min(100, overall_risk + 2)),
                max(0, min(100, overall_risk + 4)),
                overall_risk
            ]
            chart_data = pd.DataFrame(
                trend_values,
                index=["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"],
                columns=["Risk Level (%)"]
            )
            st.line_chart(chart_data, height=280)

        with col2:
            header_col1, header_col2, header_col3 = st.columns([4, 2, 2])
            with header_col1:
                st.markdown('<div style="font-weight: 600; color: #374151; font-size: 18px; margin-bottom: 12px; margin-top: 5px; font-family: sans-serif;">Assessment Report</div>', unsafe_allow_html=True)

                kf_md = "\n".join([f"- {item.get('title', '')}: {item.get('desc', '')}" for item in kf])
                ra_md = "\n".join([f"- {item.get('title', '')}: {item.get('desc', '')}" for item in ra])
                recs_md = "\n".join([f"{idx+1}. {item}" for idx, item in enumerate(recs)])
        
                report_markdown = f"""# Prediction AI - Risk Assessment Report
**Project:** {data.get('startup_name', 'Unknown')}
**Industry:** {industry}

## Key Findings
{kf_md}

## Risk Assessment
{ra_md}

## Recommendations
{recs_md}
"""
            with header_col2:
                st.download_button(label="📥 Export", data=report_markdown, file_name="risk_assessment_report.md", mime="text/markdown", use_container_width=True)
            with header_col3:
                with st.popover("🔗 Share", use_container_width=True):
                    st.write("**Share this report securely:**")
                    report_url = "http://localhost:8501/?view=dashboard&report_id=latest"
                    report_title = "Prediction AI - Risk Assessment Report"
                    st.code(report_url)
                    if st.button("Copy Link", use_container_width=True):
                        st.toast("Report link copied to clipboard! ✅")

                    st.divider()
                    st.markdown('<div style="font-size: 14px; font-weight: 600; margin-bottom: 8px;">Share via:</div>', unsafe_allow_html=True)

                    s1, s2 = st.columns(2)
                    with s1:
                        import urllib.parse
                        subject = urllib.parse.quote(report_title)
                        body = urllib.parse.quote(f"Check out my project risk report: {report_url}")
                        st.markdown(f'<a href="mailto:?subject={subject}&body={body}" style="display: block; text-align: center; background: #EA4335; color: white; padding: 6px; border-radius: 4px; text-decoration: none; font-size: 14px; margin-bottom: 8px; font-weight: 600;">📧 Email</a>', unsafe_allow_html=True)
                        st.markdown(f'<a href="https://wa.me/?text=Check%20out%20my%20project%20risk%20report:%20{report_url}" target="_blank" style="display: block; text-align: center; background: #25D366; color: white; padding: 6px; border-radius: 4px; text-decoration: none; font-size: 14px; margin-bottom: 8px; font-weight: 600;">💬 WhatsApp</a>', unsafe_allow_html=True)
                    with s2:
                        st.markdown(f'<a href="sms:?body=Check out my project risk report: {report_url}" target="_blank" style="display: block; text-align: center; background: #3B82F6; color: white; padding: 6px; border-radius: 4px; text-decoration: none; font-size: 14px; margin-bottom: 8px; font-weight: 600;">📱 Messages</a>', unsafe_allow_html=True)
                        st.markdown(f'<a href="https://www.linkedin.com/sharing/share-offsite/?url=http://localhost:8501" target="_blank" style="display: block; text-align: center; background: #0A66C2; color: white; padding: 6px; border-radius: 4px; text-decoration: none; font-size: 14px; margin-bottom: 8px; font-weight: 600;">💼 LinkedIn</a>', unsafe_allow_html=True)

            st.markdown(f'''
            <div style="border: 1px solid #E5E7EB; border-radius: 6px; padding: 16px; margin-bottom: 16px; background: white; font-family: sans-serif;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-weight: 600; font-size: 17px; color: #111827;">Key Findings</span>
                    <span style="color: #D97706; background: #FEF3C7; font-size: 14px; font-weight: 600; padding: 2px 6px; border-radius: 12px;">High Priority</span>
                </div>
                <ul style="color: #6B7280; font-size: 16px; padding-left: 16px; margin-bottom: 0;">
                {"".join([f'<li style="font-size: 16px;"><b>{item.get("title", "")}</b>: {item.get("desc", "")}</li>' for item in kf])}
            </ul>
            </div>

            <div style="border: 1px solid #E5E7EB; border-radius: 6px; padding: 16px; margin-bottom: 16px; background: white; font-family: sans-serif;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-weight: 600; font-size: 17px; color: #111827;">Risk Assessment</span>
                    <span style="color: #2563EB; background: #DBEAFE; font-size: 14px; font-weight: 600; padding: 2px 6px; border-radius: 12px;">Medium Priority</span>
                </div>
                <p style="color: #6B7280; font-size: 16px; margin-bottom: 0;">
                    {"".join([f'<b>{item.get("title", "")}:</b> {item.get("desc", "")}<br>' for item in ra])}
                </p>
            </div>

            <div style="border: 1px solid #E5E7EB; border-radius: 6px; padding: 16px; background: white; font-family: sans-serif;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-weight: 600; font-size: 17px; color: #111827;">Recommendations</span>
                    <span style="color: #DC2626; background: #FEE2E2; font-size: 14px; font-weight: 600; padding: 2px 6px; border-radius: 12px;">Action Required</span>
                </div>
                <ol style="color: #6B7280; font-size: 16px; padding-left: 16px; margin-bottom: 0;">
                {"".join([f'<li style="font-size: 16px;">{item}</li>' for item in recs])}
            </ol>
            </div>
            ''', unsafe_allow_html=True)

        with col3:
            st.markdown('<div style="font-weight: 600; color: #374151; font-size: 18px; margin-bottom: 12px; font-family: sans-serif;">Strategic Insights 🚀</div>', unsafe_allow_html=True)

            st.markdown(f'''
            <div style="border-left: 3px solid #EF4444; padding-left: 12px; margin-bottom: 8px; font-family: sans-serif;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-weight: 600; font-size: 16px; color: #111827;">Funding Strategy</span>
                    <span style="color: #EF4444; background: #FEE2E2; font-size: 13px; font-weight: 600; padding: 2px 6px; border-radius: 4px;">Critical Impact</span>
                </div>
                <p style="color: #6B7280; font-size: 15px; margin-bottom: 0px; line-height: 1.4;">{report.get("funding_strategy", "N/A")}</p>
            </div>
            ''', unsafe_allow_html=True)

            with st.popover("See funding options 👉", use_container_width=True):
                st.markdown("**Recommended Funding Options**")
                st.info("🏙️ **Bridge Round:** Seek $500k convertible note from existing investors.")
                st.success("🤝 **Strategic Partnership:** Co-develop with enterprise client to offset R&D costs.")
                st.warning("🔄 **Pivot:** Shift to a high-margin B2B SaaS model to achieve faster profitability.")

            st.markdown('<div style="margin-bottom: 16px;"></div>', unsafe_allow_html=True)

            st.markdown(f'''
            <div style="border-left: 3px solid #F59E0B; padding-left: 12px; margin-bottom: 8px; font-family: sans-serif;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-weight: 600; font-size: 16px; color: #111827;">Technical Advantage</span>
                    <span style="color: #F59E0B; background: #FEF3C7; font-size: 13px; font-weight: 600; padding: 2px 6px; border-radius: 4px;">Medium Impact</span>
                </div>
                <p style="color: #6B7280; font-size: 15px; margin-bottom: 0px; line-height: 1.4;">{report.get("tech_advantage", "N/A")}</p>
            </div>
            ''', unsafe_allow_html=True)

            with st.popover("View comparison 📊", use_container_width=True):
                st.markdown("**Technical Performance vs Competitors**")
                comparison_data = pd.DataFrame({
                    "Accuracy (%)": [92, 75, 68, 54]
                }, index=["Your AI", "Competitor A", "Competitor B", "Industry Avg"])
                st.bar_chart(comparison_data)

            st.markdown('<div style="margin-bottom: 16px;"></div>', unsafe_allow_html=True)

            st.markdown(f'''

            <div style="font-weight: 600; color: #374151; font-size: 17px; margin-bottom: 12px; font-family: sans-serif;">Recommended Next Steps</div>
            {"".join([f'''<div style="display: flex; align-items: center; margin-bottom: 8px; font-family: sans-serif;">
                <div style="background: #3B82F6; color: white; width: 16px; height: 16px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 13px; font-weight: bold; margin-right: 8px;">{idx+1}</div>
                <span style="color: #4B5563; font-size: 15px;">{item}</span>
            </div>''' for idx, item in enumerate(report.get('next_steps', []))])}
            ''', unsafe_allow_html=True)

        # ========================================================
        # MITIGATION STRATEGIES
        # ========================================================

        st.subheader("Risk Mitigation Strategies")

        if mitigation_results:

            for mitigation in mitigation_results:

                risk_name = mitigation.get(
                    "risk",
                    "Unknown Risk"
                )

                category = mitigation.get(
                    "category",
                    "General"
                )

                impact = mitigation.get(
                    "impact",
                    "Unknown"
                )

                strategy = mitigation.get(
                    "mitigation_strategy",
                    "N/A"
                )

                preventive = mitigation.get(
                    "preventive_action",
                    "N/A"
                )

                contingency = mitigation.get(
                    "contingency_action",
                    "N/A"
                )

                with st.expander(
                    f"⚠️ {risk_name} - {impact} Impact"
                ):

                    st.markdown(
                        f"**Category:** {category}"
                    )

                    st.markdown(
                        f"**Mitigation Strategy:** {strategy}"
                    )

                    st.markdown(
                        f"**Preventive Action:** {preventive}"
                    )

                    st.markdown(
                        f"**Contingency Action:** {contingency}"
                    )

        else:
            st.info(
                "No mitigation strategies are currently available."
            )

        st.divider()

        # ========================================================
        # IMPROVEMENT PLAN
        # ========================================================

        st.subheader("Project Improvement Plan")

        if improvement_results:

            improvement_cols = st.columns(3)

            for index, improvement in enumerate(
                improvement_results
            ):

                with improvement_cols[
                    index % 3
                ]:

                    title = improvement.get(
                        "title",
                        "Improvement"
                    )

                    category = improvement.get(
                        "category",
                        "General"
                    )

                    priority = improvement.get(
                        "priority",
                        "Medium"
                    )

                    problem = improvement.get(
                        "problem",
                        ""
                    )

                    steps = improvement.get(
                        "steps",
                        []
                    )

                    risk_reduction = improvement.get(
                        "risk_reduction",
                        ""
                    )

                    st.markdown(
                        f"""
                        <div style="
                            border: 1px solid #E5E7EB;
                            border-radius: 12px;
                            padding: 18px;
                            margin-bottom: 16px;
                            min-height: 230px;
                            background: white;
                        ">

                        <div style="
                            color: #6D28D9;
                            font-size: 13px;
                            font-weight: 700;
                            text-transform: uppercase;
                        ">
                            {category}
                        </div>

                        <h4 style="
                            margin: 8px 0;
                            color: #111827;
                        ">
                            {title}
                        </h4>

                        <div style="
                            color: #6B7280;
                            font-size: 14px;
                        ">
                            Priority: <b>{priority}</b>
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if problem:
                        st.markdown(
                            f"**Problem:** {problem}"
                        )

                    if steps:

                        st.markdown("**Action Steps:**")

                        for step in steps:
                            st.markdown(
                                f"- {step}"
                            )

                    if risk_reduction:
                        st.markdown(
                            f"**Expected Risk Reduction:** "
                            f"{risk_reduction}"
                        )

        else:
            st.info(
                "No improvement suggestions are currently available."
            )

        st.divider()

        # ========================================================



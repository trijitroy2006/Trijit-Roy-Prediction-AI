import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

_client = None

if GEMINI_API_KEY:
    try:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        _client = None


def is_llm_available():
    """Return True when Gemini is configured and available."""
    return _client is not None


# ============================================================
# GEMINI JSON GENERATION
# ============================================================

import time


def _generate_json(prompt, schema):
    """
    Send a prompt to Gemini and return structured JSON.
    Retries transient 503/429 errors.
    """

    if not _client:
        return None

    max_attempts = 4

    for attempt in range(max_attempts):
        try:
            response = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.4,
                ),
            )

            if not response.text:
                return None

            return json.loads(response.text)

        except Exception as error:

            error_text = str(error)

            # Retry temporary server/rate-limit errors
            if "503" in error_text or "UNAVAILABLE" in error_text:
                wait_time = 2 ** attempt

                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)
                continue

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                wait_time = 2 ** attempt

                print(
                    f"Gemini rate limit reached. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)
                continue

            # Don't retry permanent errors
            print(f"Gemini API error: {error}")
            return None

    print("Gemini request failed after all retry attempts.")
    return None


# ============================================================
# PROJECT ANALYSIS
# ============================================================

def generate_project_analysis(project_data):
    """
    Generate an AI-based high-level analysis of the submitted project.

    This is used during project submission and provides:
    - project summary
    - market considerations
    - competitive considerations
    - initial risk observations
    """

    if not is_llm_available():
        return {
            "mode": "DEMO",
            "project_summary": (
                f"{project_data.get('startup_name', 'Project')} "
                f"is a {project_data.get('business_model', 'business')} "
                f"project operating in the "
                f"{project_data.get('industry', 'technology')} sector."
            ),
            "market_observations": [
                "Validate target customer demand before major investment.",
                "Assess the size and growth potential of the target market.",
                "Monitor competitor positioning and differentiation."
            ],
            "initial_risk_observations": [
                "Market acceptance risk",
                "Competitive risk",
                "Resource and execution risk"
            ]
        }

    schema = {
        "type": "object",
        "properties": {
            "project_summary": {
                "type": "string"
            },
            "market_observations": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "initial_risk_observations": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": [
            "project_summary",
            "market_observations",
            "initial_risk_observations"
        ]
    }

    prompt = f"""
You are an expert startup and project risk analyst.

Analyze the following project.

PROJECT DATA:
{json.dumps(project_data, indent=2, default=str)}

Provide:
1. A concise project summary.
2. Important market observations.
3. Initial risk observations.

Do not invent specific market statistics or competitor numbers.
Base the analysis only on the information provided.
"""

    result = _generate_json(prompt, schema)

    if result is None:
        return {
            "mode": "DEMO",
            "project_summary": "AI analysis unavailable.",
            "market_observations": [],
            "initial_risk_observations": []
        }

    result["mode"] = "GEMINI"

    return result


# ============================================================
# M3 STRATEGIC RECOMMENDATIONS
# ============================================================

def generate_llm_recommendations(
    project_data,
    risk_input_data,
    swot,
    feasibility_score,
    market_data=None,
    risk_data=None,
    base_recommendations=None
):
    """
    Generate M3 strategic recommendations using Gemini.

    The output structure intentionally matches the existing
    recommendation/dashboard structure.
    """

    # --------------------------------------------------------
    # DEMO MODE
    # --------------------------------------------------------

    if not is_llm_available():

        if base_recommendations:
            result = dict(base_recommendations)
            result["mode"] = "DEMO"
            return result

        return {
            "mode": "DEMO",
            "overall_strategic_recommendation": (
                "Address the highest-priority risks first, "
                "validate market demand, strengthen execution "
                "capability, and control resource allocation."
            ),
            "recommendations": [],
            "short_term_action_plan": [
                "Validate the target customer problem.",
                "Review the highest-priority project risks.",
                "Validate budget and resource assumptions."
            ],
            "long_term_action_plan": [
                "Build sustainable competitive differentiation.",
                "Scale only after validating product-market fit.",
                "Establish continuous risk monitoring."
            ]
        }

    # --------------------------------------------------------
    # GEMINI SCHEMA
    # --------------------------------------------------------

    schema = {
        "type": "object",
        "properties": {
            "overall_strategic_recommendation": {
                "type": "string"
            },

            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string"
                        },
                        "category": {
                            "type": "string"
                        },
                        "priority": {
                            "type": "string"
                        },
                        "problem": {
                            "type": "string"
                        },
                        "explanation": {
                            "type": "string"
                        },
                        "action": {
                            "type": "string"
                        },
                        "risk_reduction": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "title",
                        "category",
                        "priority",
                        "problem",
                        "explanation",
                        "action",
                        "risk_reduction"
                    ]
                }
            },

            "short_term_action_plan": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },

            "long_term_action_plan": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": [
            "overall_strategic_recommendation",
            "recommendations",
            "short_term_action_plan",
            "long_term_action_plan"
        ]
    }

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are the AI Strategic Recommendation Engine for a
Startup & Project Risk Analyzer.

Your job is to analyze the project and produce practical,
risk-linked strategic recommendations.

IMPORTANT:
- Do not invent facts.
- Use the supplied project information.
- Recommendations must be directly connected to identified risks.
- Every recommendation must explain:
  1. the problem/risk,
  2. why it matters,
  3. what should be done,
  4. how the action reduces risk.
- Prioritize the most important risks.
- Be practical and specific.
- Avoid generic motivational advice.
- Do not claim certainty about business success.

PROJECT INFORMATION:
{json.dumps(project_data, indent=2, default=str)}

RISK INPUTS:
{json.dumps(risk_input_data, indent=2, default=str)}

OVERALL RISK SCORE:
{json.dumps({
    "risk_score": risk_input_data.get("risk_score"),
    "feasibility_score": feasibility_score
}, indent=2, default=str)}

SWOT:
{json.dumps(swot, indent=2, default=str)}

MARKET DATA:
{json.dumps(market_data or {}, indent=2, default=str)}

IDENTIFIED RISKS:
{json.dumps(risk_data or [], indent=2, default=str)}

EXISTING BASELINE RECOMMENDATIONS:
{json.dumps(base_recommendations or {}, indent=2, default=str)}

Generate:

A. Overall Strategic Recommendation

B. Strategic Recommendations

Each recommendation must contain:
- title
- category
- priority
- problem
- explanation
- action
- risk_reduction

Categories may include:
- Risk
- Market
- Technical
- Financial
- Operational
- Product
- Marketing

C. Short-Term Action Plan

D. Long-Term Action Plan

Return ONLY the requested structured JSON.
"""

    result = _generate_json(prompt, schema)

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if result is None:

        if base_recommendations:
            fallback = dict(base_recommendations)
            fallback["mode"] = "FALLBACK"
            return fallback

        return {
            "mode": "FALLBACK",
            "overall_strategic_recommendation": (
                "Review the highest-priority risks and implement "
                "targeted mitigation actions before scaling."
            ),
            "recommendations": [],
            "short_term_action_plan": [],
            "long_term_action_plan": []
        }

    result["mode"] = "GEMINI"

    print("✅ Gemini AI generated recommendations")

    return result

def generate_llm_mitigation(
    project_data,
    risk_data,
    risk_input_data,
    swot,
    feasibility_score,
    base_mitigation=None
):
    """
    Generate M3 risk mitigation strategies using Gemini.
    """

    # --------------------------------------------------------
    # DEMO MODE
    # --------------------------------------------------------

    if not is_llm_available():

        print("⚠️ Using Demo Mode mitigation strategies")

        return base_mitigation or []

    # --------------------------------------------------------
    # GEMINI SCHEMA
    # --------------------------------------------------------

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "risk": {
                    "type": "string"
                },
                "category": {
                    "type": "string"
                },
                "description": {
                    "type": "string"
                },
                "impact": {
                    "type": "string"
                },
                "priority": {
                    "type": "string"
                },
                "mitigation_strategy": {
                    "type": "string"
                },
                "preventive_action": {
                    "type": "string"
                },
                "contingency_action": {
                    "type": "string"
                }
            },
            "required": [
                "risk",
                "category",
                "description",
                "impact",
                "priority",
                "mitigation_strategy",
                "preventive_action",
                "contingency_action"
            ]
        }
    }

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are the Risk Mitigation Engine for a
Startup & Project Risk Analyzer.

Analyze the supplied project and identified risks.

For every major identified risk, generate a practical
mitigation strategy.

IMPORTANT:
- Use only the supplied project information.
- Do not invent facts.
- Focus on the identified risks.
- Do not create random unrelated risks.
- Prioritize high-impact risks.
- Recommendations must be practical and actionable.
- Preventive actions should reduce the probability of the risk.
- Contingency actions should explain what to do if the risk occurs.

PROJECT INFORMATION:
{json.dumps(project_data, indent=2, default=str)}

RISK INPUTS:
{json.dumps(risk_input_data, indent=2, default=str)}

RISK DATA:
{json.dumps(risk_data, indent=2, default=str)}

SWOT:
{json.dumps(swot, indent=2, default=str)}

FEASIBILITY SCORE:
{json.dumps(feasibility_score, indent=2, default=str)}

BASELINE MITIGATION:
{json.dumps(base_mitigation or [], indent=2, default=str)}

For every major risk return:

- risk
- category
- description
- impact
- priority
- mitigation_strategy
- preventive_action
- contingency_action

Return ONLY valid structured JSON.
"""

    result = _generate_json(prompt, schema)

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if result is None:

        print("⚠️ Gemini unavailable — using fallback mitigation strategies")

        return base_mitigation or []

    print("✅ Gemini AI generated mitigation strategies")

    return result

def generate_llm_improvements(
    project_data,
    risk_input_data,
    swot,
    feasibility_score,
    market_data=None,
    risk_data=None,
    mitigation_results=None,
    base_improvements=None
):
    """
    Generate M3 project improvement suggestions using Gemini.
    """

    # --------------------------------------------------------
    # DEMO MODE
    # --------------------------------------------------------

    if not is_llm_available():

        print("⚠️ Using Demo Mode improvement suggestions")

        return base_improvements or []

    # --------------------------------------------------------
    # GEMINI SCHEMA
    # --------------------------------------------------------

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string"
                },
                "category": {
                    "type": "string"
                },
                "priority": {
                    "type": "string"
                },
                "problem": {
                    "type": "string"
                },
                "improvement": {
                    "type": "string"
                },
                "reason": {
                    "type": "string"
                },
                "steps": {
                    "type": "string"
                },
                "expected_benefit": {
                    "type": "string"
                },
                "risk_reduction": {
                    "type": "string"
                }
            },
            "required": [
                "title",
                "category",
                "priority",
                "problem",
                "improvement",
                "reason",
                "steps",
                "expected_benefit",
                "risk_reduction"
            ]
        }
    }

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are the Project Improvement Suggestion Engine for a
Startup & Project Risk Analyzer.

Analyze the project, risks, SWOT, feasibility,
market information and mitigation strategies.

Generate practical improvements that can make the project
more viable, competitive and executable.

Organize improvements under these categories:

- Product
- Market
- Technical
- Financial
- Operational
- Marketing

IMPORTANT:
- Do not invent facts.
- Improvements must be relevant to the supplied project.
- Improvements should address weaknesses, risks or opportunities.
- Avoid generic motivational advice.
- Prioritize improvements with the greatest practical impact.
- Explain why each improvement matters.
- Explain the expected benefit.
- Explain how it can reduce project risk where applicable.

PROJECT INFORMATION:
{json.dumps(project_data, indent=2, default=str)}

RISK INPUTS:
{json.dumps(risk_input_data, indent=2, default=str)}

IDENTIFIED RISKS:
{json.dumps(risk_data or [], indent=2, default=str)}

SWOT:
{json.dumps(swot, indent=2, default=str)}

FEASIBILITY SCORE:
{json.dumps(feasibility_score, indent=2, default=str)}

MARKET DATA:
{json.dumps(market_data or {}, indent=2, default=str)}

MITIGATION STRATEGIES:
{json.dumps(mitigation_results or [], indent=2, default=str)}

BASELINE IMPROVEMENTS:
{json.dumps(base_improvements or [], indent=2, default=str)}

For every improvement return:

- title
- category
- priority
- problem
- improvement
- reason
- steps
- expected_benefit
- risk_reduction

Return ONLY valid structured JSON.
"""

    result = _generate_json(prompt, schema)

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if result is None:

        print("⚠️ Gemini unavailable — using fallback improvement suggestions")

        return base_improvements or []

    print("✅ Gemini AI generated improvement suggestions")

    return result


from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from milestone_3 import database
from milestone_1 import market_analysis
import secrets
import feasibility
from milestone_2.risk_engine import (
    calculate_risk,
    get_risk_status,
    calculate_success_probability
)
from milestone_2.swot_analysis import generate_swot

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        project_data = {
            'startup_name': request.form.get('startup_name'),
            'industry': request.form.get('industry'),
            'business_model': request.form.get('business_model'),
            'target_market': request.form.get('target_market'),
            'budget': request.form.get('budget'),
            'project_description': request.form.get('project_description')
        }
        try:
            project_id = database.insert_project(project_data)
            session['project_id'] = project_id
        except Exception as e:
            print(f"Database error: {e}")
            session['fallback_project_data'] = project_data
            
        return redirect(url_for('dashboard'))
        
    project_id = session.get('project_id')
    project = session.get('fallback_project_data')
    
    if project_id:
        try:
            db_project = database.get_project(project_id)
            if db_project:
                project = db_project
        except Exception as e:
            print(f"Database error fetching project: {e}")

    industry = "Technology"
    startup_name = ""
    target_market = ""
    budget = 0
    business_model = "SaaS"

    if project:
        industry = project.get('industry', 'Technology')
        startup_name = project.get('startup_name', '')
        target_market = project.get('target_market', '')
        business_model = project.get('business_model', 'SaaS')
        try:
            budget = float(project.get('budget', 0))
        except:
            budget = 0

    market_data = market_analysis.get_market_data(industry, target_market, budget)
    competitors = market_analysis.get_competitor_data(startup_name, industry, business_model)
    
    return render_template('dashboard.html', market_data=market_data, competitors=competitors, project=project)

@app.route('/risk_assessment', methods=['GET', 'POST'])
def risk_assessment():
    project_id = session.get('project_id')
    project = session.get('fallback_project_data')

    if project_id:
        try:
            db_project = database.get_project(project_id)
            if db_project:
                project = db_project
        except Exception as e:
            print(f"Database error fetching project: {e}")

    # Calculate risk when the form is submitted
    if request.method == 'POST':
        market_competition = request.form.get('market_competition')
        team_expertise = request.form.get('team_expertise')
        resource_availability = request.form.get('resource_availability')
        innovation_level = request.form.get('innovation_level')
        market_research = request.form.get('market_research')
        market_opportunity = int(request.form.get('market_opportunity', 50))
        team_capability = int(request.form.get('team_capability', 50))
        competitive_advantage = int(request.form.get('competitive_advantage', 50))
        resource_availability_score = int(
            request.form.get('resource_availability_score', 50)
        )

        
        risk_score = calculate_risk(
            market_competition,
            team_expertise,
            resource_availability,
            innovation_level,
            market_research
        )

        
        risk_status = get_risk_status(risk_score)

        
        success_probability = calculate_success_probability(risk_score)

        
        swot = generate_swot(
            team_expertise,
            innovation_level,
            market_competition,
            resource_availability,
            market_research
        )

        if isinstance(swot, dict):
            swot_data = swot
        else:
            swot_data = {
                'strengths': str(swot),
                'weaknesses': 'N/A',
                'opportunities': 'N/A',
                'threats': 'N/A'
            }

        
        feasibility_score = feasibility.calculate_feasibility(
            market_opportunity,
            team_capability,
            competitive_advantage,
            resource_availability_score
        )

       
        print(f"DEBUG: Current project_id from session is {project_id}")
        if project_id:
            try:
                swot_id = database.insert_swot_analysis(project_id, swot_data)
                print(f"DEBUG: Inserted SWOT with ID {swot_id}")

                
                risk_id = database.insert_risk_assessment(
                    project_id=project_id,
                    risk_category="Overall",
                    risk_score=risk_score,
                    risk_description=f"Status: {risk_status}",
                    priority_level="High" if risk_score > 60 else "Medium"
                )
                print(f"DEBUG: Inserted Risk Assessment with ID {risk_id}")

                # Insert Success Prediction record
                prediction_id = database.insert_success_prediction(
                    project_id=project_id,
                    success_probability=success_probability,
                    overall_risk_score=risk_score
                )
                print(f"DEBUG: Inserted Success Prediction with ID {prediction_id}")
            except Exception as e:
                print(f"DATABASE ERROR ON INSERT: {e}")
        else:
            print("DEBUG WARNING: No project_id found in session. Fill out dashboard form first!")

        return render_template(
            'risk_assessment.html',
            project=project,
            risk_score=risk_score,
            risk_status=risk_status,
            success_probability=success_probability,
            swot=swot,
            feasibility_score=feasibility_score
        )

    # Normal GET request
    return render_template(
        'risk_assessment.html',
        project=project
    )

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.json
    industry = data.get('industry', 'Technology')
    startup_name = data.get('startup_name', '')
    target_market = data.get('target_market', '')
    business_model = data.get('business_model', 'SaaS')
    try:
        budget = float(data.get('budget', 0))
    except:
        budget = 0
        
    market_data = market_analysis.get_market_data(industry, target_market, budget)
    competitors = market_analysis.get_competitor_data(startup_name, industry, business_model)
    
    return jsonify({
        "market_data": market_data,
        "competitors": competitors
    })

@app.route('/placeholder')
def placeholder():
    return render_template('placeholder.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)










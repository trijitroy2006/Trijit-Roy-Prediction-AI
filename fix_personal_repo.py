import shutil
import os
import re

src = r"c:\Users\user\Desktop\ins\ML_Market_Intelligence_Project"
dest = r"c:\Users\user\Desktop\ins\ML_Market_Intelligence_Project_Personal"

# 1. Create milestone_4
m4_dir = os.path.join(dest, "milestone_4")
os.makedirs(m4_dir, exist_ok=True)
open(os.path.join(m4_dir, "__init__.py"), "w").close()

# 2. Copy llm_service.py to milestone_4
shutil.copy2(os.path.join(src, "llm_service.py"), os.path.join(m4_dir, "llm_service.py"))

# 3. Copy app_streamlit.py to root
app_src = os.path.join(src, "app_streamlit.py")
app_dest = os.path.join(dest, "app_streamlit.py")
shutil.copy2(app_src, app_dest)

# 4. Copy requirements.txt and .env.example
shutil.copy2(os.path.join(src, "requirements.txt"), os.path.join(dest, "requirements.txt"))
if os.path.exists(os.path.join(src, ".env.example")):
    shutil.copy2(os.path.join(src, ".env.example"), os.path.join(dest, ".env.example"))

# 5. Fix imports in app_streamlit.py
with open(app_dest, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("import market_analysis", "from milestone_1 import market_analysis")
content = content.replace("import database", "from milestone_3 import database")
content = content.replace("from recommendation_engine import generate_recommendations", "from milestone_3.recommendation_engine import generate_recommendations")
content = content.replace("from mitigation_engine import generate_mitigation", "from milestone_3.mitigation_engine import generate_mitigation")
content = content.replace("from improvement_engine import generate_improvements", "from milestone_3.improvement_engine import generate_improvements")
content = content.replace("from llm_service import generate_llm_recommendations, generate_llm_mitigation, generate_llm_improvements", "from milestone_4.llm_service import generate_llm_recommendations, generate_llm_mitigation, generate_llm_improvements")
content = content.replace("from risk_engine import calculate_risk", "from milestone_2.risk_engine import calculate_risk")
content = content.replace("from swot_analysis import generate_swot", "from milestone_2.swot_analysis import generate_swot")
content = content.replace("from feasibility import calculate_feasibility", "from milestone_2.feasibility import calculate_feasibility")

with open(app_dest, "w", encoding="utf-8") as f:
    f.write(content)

print("Files copied and imports fixed.")

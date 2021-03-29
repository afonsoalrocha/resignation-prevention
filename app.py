from flask import Flask
from flask import request
import joblib as jb
import pandas as pd
import json

app = Flask(__name__)

mdl = jb.load("model.pkl.z")

@app.route("/")
def main():
    
    print(request.args)

    satisfaction_level = request.args.get('sl')
    last_evaluation = request.args.get("le")
    number_project = request.args.get("nup")
    average_monthly_hours = request.args.get("amh")
    time_spend_company = request.args.get("tec")
    department = request.args.get("dep")
    salary = request.args.get("sal")

    pred = pd.DataFrame([[satisfaction_level, last_evaluation, number_project, average_monthly_hours,
                    time_spend_company, department, salary]])

    pred.rename(columns={0:'satisfaction_level', 1:'last_evaluation', 2:'number_project',
                     3:'average_monthly_hours', 4:'time_spend_company', 5:'department', 6:'salary'}, inplace=True)
    
    
    res = {"p": int(mdl.predict(pred)[0])}
    return json.dumps(res)

if __name__ == "__main__":
        app.run(port=3300)
import requests

url = 'http://localhost:9696/predict'

patient_name = "John Doe"
patient = {
        "Gender": "Male",
        "Age": 16.30687,
        "Height": 1.752755,
        "Weight": 50.0,
        "family_history": "no",
        "FAVC": "yes",
        "FCVC": 2.310423,
        "NCP": 3.558637,
        "CAEC": "Sometimes",
        "SMOKE": "no",
        "CH2O": 1.787843,
        "SCC": "no",
        "FAF": 1.926592,
        "TUE": 0.828549,
        "CALC": "Sometimes",
        "MTRANS": "Public_Transportation"
        }

response = requests.post(url, json=patient).json()
print(response)

if response["obesity"] == True:
    print("Please consult your doctor for fighting obesity.")
else:
    print("You are not obese. Keep up the good work!")

#to just check if app is working 
#result = subprocess.run(
#    ["curl", "http://127.0.0.1:9696/ping"], 
#    capture_output=True, 
#    text=True, 
#    timeout=30
#)
#print(f"Return code: {result.returncode}")
#print(f"STDOUT:\n{result.stdout}")
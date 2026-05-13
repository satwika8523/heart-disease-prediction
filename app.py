from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset and train the model
df = pd.read_csv('HDD(4).csv')  # Replace 'HDD(4).csv' with the path to your dataset file

# Handle missing values if any
df.ffill(inplace=True)  # Forward fill missing values

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Gender', 'Smoking_Habit', 'Diet', 'Family_History', 'Exercise_Habits', 'Heart_Condition', 'Swelling', 'Dizziness', 'Irregular_Heart_Rate'])

# Split dataset into features (X) and target variable (y)
X = df_encoded.drop('Diagnosed_Heart_Disease', axis=1)
y = df_encoded['Diagnosed_Heart_Disease']

# Train a Machine Learning Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    new_patient_data = {
        'Patient_ID': int(request.form['Patient_ID']),
        'Age': int(request.form['Age']),
        'Systolic_BP': int(request.form['systolicBP']),
        'Diastolic_BP': int(request.form['diastolicBP']),
        'Gender_Female': 1 if request.form['gender'] == 'female' else 0,
        'Gender_Male': 1 if request.form['gender'] == 'male' else 0,
        'Smoking_Habit_Yes': 1 if request.form['smoking'] == 'yes' else 0,
        'Smoking_Habit_No': 1 if request.form['smoking'] == 'no' else 0,
        'Diet_High Fat': 1 if request.form['diet'] == 'high fat' else 0,
        'Diet_Balanced': 1 if request.form['diet'] == 'balanced' else 0,
        'Diet_Low Fat': 1 if request.form['diet'] == 'low fat' else 0,
        'Family_History_Yes': 1 if request.form['familyHistory'] == 'yes' else 0,
        'Family_History_No': 1 if request.form['familyHistory'] == 'no' else 0,
        'Exercise_Habits_Active': 1 if request.form['exercise'] == 'Active' else 0,
        'Exercise_Habits_Moderate': 1 if request.form['exercise'] == 'Moderate' else 0,
        'Exercise_Habits_Sedentary': 1 if request.form['exercise'] == 'Sedentary' else 0,
        'Heart_Condition_Normal': 1 if request.form['heartCondition'] == 'normal' else 0,
        'Heart_Condition_Hypertension': 1 if request.form['heartCondition'] == 'hypertension' else 0,
        'Heart_Condition_Coronary Disease': 1 if request.form['heartCondition'] == 'coronary disease' else 0,
        'Heart_Condition_Aortic Disease': 1 if request.form['heartCondition'] == 'aortic disease' else 0,
        'Heart_Condition_Heart Disease': 1 if request.form['heartCondition'] == 'heart disease' else 0,
        'Swelling_Yes': 1 if request.form['swelling'] == 'yes' else 0,
        'Swelling_No': 1 if request.form['swelling'] == 'no' else 0,
        'Dizziness_Yes': 1 if request.form['dizziness'] == 'yes' else 0,
        'Dizziness_No': 1 if request.form['dizziness'] == 'no' else 0,
        'Irregular_Heart_Rate_Yes': int(request.form['irregularHeartRate']),
        'Irregular_Heart_Rate_No': 1 - int(request.form['irregularHeartRate'])
    }

    # Convert new_patient_data to DataFrame
    new_patient_df = pd.DataFrame([new_patient_data])

    # Reorder columns in new_patient_df to match the order of features used during training
    new_patient_df = new_patient_df.reindex(columns=X.columns)

    print("New Patient Data:")
    print(new_patient_df)

    # Make prediction
    Diagnosed_Heart_Disease = model.predict(new_patient_df)
    print("Predicted Disease:")
    print(Diagnosed_Heart_Disease)

    return render_template('result.html', diagnosis=Diagnosed_Heart_Disease[0])

if __name__ == '__main__':
    app.run(debug=True)
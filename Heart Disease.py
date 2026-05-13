import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the dataset
df = pd.read_csv('HDD(4).csv')  # Replace 'HDD(4).csv' with the path to your dataset file

# Step 2: Data Preprocessing
# Handle missing values if any
df.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Gender', 'Smoking_Habit', 'Diet', 'Family_History', 'Exercise_Habits', 'Heart_Condition', 'Swelling', 'Dizziness', 'Irregular_Heart_Rate'])

# Split dataset into features (X) and target variable (y)
X = df_encoded.drop('Diagnosed_Heart_Disease', axis=1)
y = df_encoded['Diagnosed_Heart_Disease']


# Step 3: Train a Machine Learning Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Prediction
# Take input from the user for a new patient
new_patient_data = {}
print("Enter patient details:")
new_patient_data['Patient_ID'] = int(input("Patient_ID: "))
new_patient_data['Age'] = int(input("Age: "))
new_patient_data['Systolic_BP'] = int(input("Systolic blood pressure: "))
new_patient_data['Diastolic_BP'] = int(input("Diastolic blood pressure: "))
new_patient_data['Gender_Female'] = int(input("Is the patient female? (1 for yes, 0 for no): "))
new_patient_data['Gender_Male'] = 1 - new_patient_data['Gender_Female']  # Calculate Gender_Male based on Gender_Female

# Repeat for other features...
new_patient_data['Smoking_Habit_Yes'] = int(input("Is the patient a smoker? (1 for yes, 0 for no): "))
new_patient_data['Smoking_Habit_No'] = 1 - new_patient_data['Smoking_Habit_Yes']  # Calculate Smoking_Habit_No based on Smoking_Habit_Yes
new_patient_data['Diet_High Fat'] = int(input("Is the patient on a high-fat diet? (1 for yes, 0 for no): "))
new_patient_data['Diet_Balanced'] = int(input("Is the patient on a balanced diet? (1 for yes, 0 for no): "))
new_patient_data['Diet_Low Fat'] = int(input("Is the patient on a low-fat diet? (1 for yes, 0 for no): "))
new_patient_data['Family_History_Yes'] = int(input("Does the patient have a family history of heart disease? (1 for yes, 0 for no): "))
new_patient_data['Family_History_No'] = 1 - new_patient_data['Family_History_Yes']  # Calculate Family_History_No based on Family_History_Yes
new_patient_data['Exercise_Habits_Active'] = int(input("Is the patient physically active? (1 for yes, 0 for no): "))
new_patient_data['Exercise_Habits_Moderate'] = int(input("Is the patient's exercise habit moderate? (1 for yes, 0 for no): "))
new_patient_data['Exercise_Habits_Sedentary'] = int(input("Is the patient sedentary? (1 for yes, 0 for no): "))
new_patient_data['Heart_Condition_Normal'] = int(input("Heart Condition of the patient? (1 for yes, 0 for no): "))
new_patient_data['Heart_Condition_Hypertension'] = int(input("Heart Condition of the patient? (1 for yes, 0 for no): "))
new_patient_data['Heart_Condition_Coronary Disease'] = int(input("Heart Condition of the patient? (1 for yes, 0 for no): "))
new_patient_data['Heart_Condition_Aortic Disease'] = int(input("Heart Condition of the patient? (1 for yes, 0 for no): "))
new_patient_data['Heart_Condition_Heart Disease'] = int(input("Heart Condition of the patient? (1 for yes, 0 for no): "))
new_patient_data['Swelling_Yes'] = int(input("Does the patient experience swelling? (1 for yes, 0 for no): "))
new_patient_data['Swelling_No'] = 1 - new_patient_data['Swelling_Yes']  # Calculate Swelling_No based on Swelling_Yes
new_patient_data['Dizziness_Yes'] = int(input("Does the patient experience dizziness? (1 for yes, 0 for no): "))
new_patient_data['Dizziness_No'] = 1 - new_patient_data['Dizziness_Yes']  # Calculate Dizziness_No based on Dizziness_Yes
new_patient_data['Irregular_Heart_Rate_Yes'] = int(input("Does the patient have an irregular heart rate? (1 for yes, 0 for no): "))
new_patient_data['Irregular_Heart_Rate_No'] = 1 - new_patient_data['Irregular_Heart_Rate_Yes']  # Calculate Irregular_Heart_Rate_No based on Irregular_Heart_Rate_Yes

# Convert new_patient_data to DataFrame
new_patient_df = pd.DataFrame([new_patient_data])

# Reorder columns in new_patient_df to match the order of features used during training
new_patient_df = new_patient_df.reindex(columns=X.columns)

# Step 5: Prediction
Diagnosed_Heart_Disease = model.predict(new_patient_df)
print("Diagnosis for the patient is:", Diagnosed_Heart_Disease)
# -------------------- IMPORT REQUIRED LIBRARIES --------------------
# Streamlit: Web app/dashboard 
import streamlit as st

# Pandas: CSV data read, process 
import pandas as pd

# Matplotlib: Graph / bar chart
import matplotlib.pyplot as plt

# Scikit-learn: Machine Learning utilities
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -------------------- APPLICATION TITLE --------------------
# Main title shown on the web app
st.title("AI-Based Student Performance Prediction System")


# -------------------- CSV FILE UPLOAD --------------------
# User can upload a CSV file from browser
st.subheader("Upload Student Dataset (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# If user uploads a file, read that CSV
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Uploaded CSV file is being analyzed ✅")
else:
    # Otherwise, use default dataset stored locally
    df = pd.read_csv("data/student_data.csv")
    st.info("Using default dataset")


# -------------------- DATA VALIDATION --------------------
# Required columns that must be present in the CSV
required_cols = [
    "attendance",
    "internal_marks",
    "assignment_score",
    "previous_result",
    "final_result"
]

# If required columns are missing, stop execution
if not all(col in df.columns for col in required_cols):
    st.error("CSV file does not have required columns ❌")
    st.stop()


# -------------------- DATA PREVIEW --------------------
# Show dataset in tabular format
st.subheader("Dataset Preview")
st.dataframe(df)


# -------------------- MODEL TRAINING --------------------
# Features (input variables)
X = df[["attendance", "internal_marks", "assignment_score", "previous_result"]]

# Target variable (Pass / Fail)
y = df["final_result"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Model Accuracy on Test Data: {accuracy*100:.2f}%")


# -------------------- FEATURE IMPORTANCE --------------------
# Coefficients show importance of each feature
st.subheader("Feature Importance")

importance = model.coef_[0]
features = X.columns

# Bar chart to visualize feature impact
fig, ax = plt.subplots()
ax.barh(features, importance)
ax.set_xlabel("Impact on Result")
st.pyplot(fig)


# -------------------- BATCH PREDICTION --------------------
# Predict results for all students in the uploaded dataset
st.subheader("Batch Prediction for Uploaded Dataset")

batch_df = df.copy()

# Select feature columns
features_df = batch_df[[
    "attendance",
    "internal_marks",
    "assignment_score",
    "previous_result"
]]

# Predict PASS / FAIL
batch_df["Predicted_Result"] = model.predict(features_df)

# Predict probability of PASS
batch_df["Pass_Probability"] = model.predict_proba(features_df)[:, 1]

# Convert numeric prediction to readable labels
batch_df["Predicted_Result"] = batch_df["Predicted_Result"].map(
    {1: "PASS", 0: "FAIL"}
)

# Function to classify risk level
def risk_level(prob):
    if prob >= 0.75:
        return "Low Risk"
    elif prob >= 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

# Apply risk classification
batch_df["Risk_Level"] = batch_df["Pass_Probability"].apply(risk_level)

st.caption("Predictions generated for all students in the uploaded dataset")
st.dataframe(batch_df)


# -------------------- SINGLE STUDENT PREDICTION --------------------
# User input using sliders
st.subheader("Student Performance Prediction")

attendance = st.slider("Attendance (%)", 0, 100, 75)
internal_marks = st.slider("Internal Marks", 0, 30, 15)
assignment_score = st.slider("Assignment Score", 0, 20, 10)
previous_result = st.selectbox("Previous Result", [0, 1])

# Combine inputs into model format
input_data = [[attendance, internal_marks, assignment_score, previous_result]]

# Predict result and probability
prediction = model.predict(input_data)
prob = model.predict_proba(input_data)[0]

# Display prediction result
if prediction[0] == 1:
    st.success("Prediction: Student will PASS ✅")
else:
    st.error("Prediction: Student may FAIL ❌")


# -------------------- DOWNLOAD REPORT --------------------
# Allow user to download prediction report as CSV
st.subheader("Download Prediction Report")

csv = batch_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Prediction Report (CSV)",
    data=csv,
    file_name="student_prediction_report.csv",
    mime="text/csv"
)


# -------------------- AUTO INSIGHTS SUMMARY --------------------
# Class-level analytics
st.subheader("Auto Insights Summary")

total_students = len(batch_df)

high_risk = (batch_df["Risk_Level"] == "High Risk").sum()
medium_risk = (batch_df["Risk_Level"] == "Medium Risk").sum()
low_risk = (batch_df["Risk_Level"] == "Low Risk").sum()

high_risk_pct = (high_risk / total_students) * 100
medium_risk_pct = (medium_risk / total_students) * 100
low_risk_pct = (low_risk / total_students) * 100

avg_attendance = batch_df["attendance"].mean()
avg_internal = batch_df["internal_marks"].mean()
avg_assignment = batch_df["assignment_score"].mean()

st.write(f"📊 Total Students Analyzed: {total_students}")
st.write(f"🔴 High Risk Students: {high_risk_pct:.1f}%")
st.write(f"🟡 Medium Risk Students: {medium_risk_pct:.1f}%")
st.write(f"🟢 Low Risk Students: {low_risk_pct:.1f}%")

st.markdown("---")

# Generate advisory message based on risk level
if high_risk_pct > 40:
    st.warning(
        "⚠️ Many students are at high academic risk. "
        "Immediate intervention is recommended."
    )
elif medium_risk_pct > 40:
    st.info(
        "ℹ️ A significant number of students are at medium risk. "
        "Focused academic support can improve outcomes."
    )
else:
    st.success(
        "✅ Overall class performance is good."
    )

st.markdown("---")

st.write("📈 Average Academic Indicators")
st.write(f"- Average Attendance: {avg_attendance:.1f}%")
st.write(f"- Average Internal Marks: {avg_internal:.1f}")
st.write(f"- Average Assignment Score: {avg_assignment:.1f}")


# -------------------- STUDENT RISK LEVEL --------------------
st.subheader("Student Risk Level")

if prediction[0] == 1 and prob[1] > 0.75:
    st.success("Low Risk Student 🟢")
elif prediction[0] == 1:
    st.warning("Medium Risk Student 🟡")
else:
    st.error("High Risk Student 🔴")


# -------------------- AI-BASED RECOMMENDATION --------------------
st.subheader("AI-Based Recommendation")

if prediction[0] == 0:
    message = (
        "The student is at high academic risk. Improve attendance, assignments, "
        "and consider academic counseling."
    )
elif prob[1] < 0.75:
    message = (
        "The student is likely to pass but needs improvement in academic performance."
    )
else:
    message = (
        "The student is performing well. Maintain current study habits."
    )

st.info(message)


# -------------------- FOOTER --------------------
st.write("""
Limitations:
- Small dataset used
- Only academic factors considered

Future Scope:
- Larger real-world datasets
- Attendance automation
- Mobile app integration
- Advanced ML models
""")

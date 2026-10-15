import pandas as pd
import numpy as np
from functions import calculate_total, calculate_average, calculate_grade, calculate_result, get_performance_status

data = pd.read_csv("students.csv")

subjects = ["Python", "Mathematics", "AI"]

if data.isnull().values.any():
    print("Dataset contains missing values.")
    data = data.dropna()

for subject in subjects:
    data[subject] = pd.to_numeric(data[subject], errors="coerce")

data = data.dropna(subset=subjects + ["Attendance"])

valid_marks = data[subjects].apply(lambda column: column.between(0, 100).all()).all()
valid_attendance = data["Attendance"].between(0, 100).all()

if not valid_marks or not valid_attendance:
    raise ValueError("Marks and attendance must be between 0 and 100.")

data["Total"] = data[subjects].apply(lambda row: calculate_total(row), axis=1)
data["Average"] = data[subjects].apply(lambda row: calculate_average(row), axis=1).round(2)
data["Grade"] = data["Average"].apply(calculate_grade)
data["Result"] = data[subjects].apply(lambda row: calculate_result(row), axis=1)
data["Performance"] = data.apply(
    lambda row: get_performance_status(row["Average"], row["Attendance"]),
    axis=1
)

print("\n" + "=" * 65)
print("          STUDENT PERFORMANCE ANALYTICS SYSTEM")
print("=" * 65)

print("\nSTUDENT PERFORMANCE REPORT\n")
print(
    data[
        ["Student_ID", "Name", "Total", "Average", "Grade",
         "Result", "Attendance", "Performance"]
    ].to_string(index=False)
)

print("\n" + "=" * 65)
print("                    CLASS SUMMARY")
print("=" * 65)

print(f"\nTotal Students       : {len(data)}")
print(f"Class Average        : {data['Average'].mean():.2f}")
print(f"Highest Average      : {data['Average'].max():.2f}")
print(f"Lowest Average       : {data['Average'].min():.2f}")
print(f"Students Passed      : {(data['Result'] == 'Pass').sum()}")
print(f"Students Failed      : {(data['Result'] == 'Fail').sum()}")
print(f"Excellent Performers : {(data['Performance'] == 'Excellent').sum()}")
print(f"Needs Improvement    : {(data['Performance'] == 'Needs Improvement').sum()}")

print("\nDEPARTMENT-WISE ANALYSIS\n")

department_summary = data.groupby("Department").agg(
    Students=("Student_ID", "count"),
    Average_Marks=("Average", "mean"),
    Passed=("Result", lambda results: (results == "Pass").sum()),
    Failed=("Result", lambda results: (results == "Fail").sum())
).round(2)

print(department_summary.to_string())

print("\nSUBJECT-WISE AVERAGE\n")

subject_averages = data[subjects].mean()

for subject, average in subject_averages.items():
    print(f"{subject:15}: {average:.2f}")

print(f"\nBest Subject         : {subject_averages.idxmax()}")
print(f"Weakest Subject      : {subject_averages.idxmin()}")

print("\nTOP 5 STUDENTS\n")

top_students = data.nlargest(5, "Average")
print(top_students[["Name", "Average", "Grade"]].to_string(index=False))

print("\nSTUDENTS NEEDING ATTENTION\n")

attention_list = data[data["Performance"] == "Needs Improvement"]

if attention_list.empty:
    print("No students flagged.")
else:
    print(
        attention_list[
            ["Name", "Average", "Attendance", "Performance"]
        ].to_string(index=False)
    )

normalized_marks = data[subjects].to_numpy(dtype=float) / 100

weights = np.array([0.4, 0.3, 0.3])
bias = 0.0

neuron_scores = np.maximum(0, np.dot(normalized_marks, weights) + bias)

data["Neuron_Score"] = np.round(neuron_scores, 3)

print("\nARTIFICIAL NEURON DEMO\n")
print(data[["Name", "Neuron_Score"]].head().to_string(index=False))

print("\nAnalysis completed successfully.")
import numpy as np


def calculate_total(marks):
    return np.sum(marks)


def calculate_average(marks):
    return np.mean(marks)


def calculate_grade(average):
    if average >= 90:
        return "A+"
    elif average >= 80:
        return "A"
    elif average >= 70:
        return "B"
    elif average >= 60:
        return "C"
    elif average >= 40:
        return "D"
    else:
        return "F"


def calculate_result(marks):
    if np.any(np.array(marks) < 40):
        return "Fail"
    return "Pass"


def get_performance_status(average, attendance):
    if average < 50 or attendance < 75:
        return "Needs Improvement"
    elif average >= 85 and attendance >= 90:
        return "Excellent"
    return "Satisfactory"
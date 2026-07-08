"""
Module 1A: Synthetic Dataset Generator
========================================
Generates a realistic synthetic student performance dataset with 1,000 records.
Features realistic correlations between study habits, attendance, and performance.
"""

import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_students=1000, random_seed=42, output_path=None):
    """
    Generate a synthetic student performance dataset.
    
    Parameters:
        n_students (int): Number of student records to generate.
        random_seed (int): Seed for reproducibility.
        output_path (str): Path to save the CSV file. If None, uses default.
    
    Returns:
        pd.DataFrame: Generated dataset.
    """
    np.random.seed(random_seed)
    
    # ---------- Student IDs ----------
    student_ids = np.arange(1, n_students + 1)
    
    # ---------- Demographic Features ----------
    gender = np.random.choice(['Male', 'Female'], size=n_students, p=[0.52, 0.48])
    age = np.random.randint(17, 26, size=n_students)
    
    parent_education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'],
        size=n_students,
        p=[0.35, 0.35, 0.20, 0.10]
    )
    
    family_income = np.random.choice(
        ['Low', 'Medium', 'High'],
        size=n_students,
        p=[0.30, 0.45, 0.25]
    )
    
    # ---------- Academic & Behavioral Features ----------
    # Attendance (influenced by family income)
    income_attendance_boost = np.where(
        np.array(family_income) == 'High', 8,
        np.where(np.array(family_income) == 'Medium', 3, 0)
    )
    attendance_percentage = np.clip(
        np.random.normal(72, 12, n_students) + income_attendance_boost,
        40, 100
    ).round(1)
    
    # Study hours (influenced by parent education)
    edu_study_boost = np.where(
        np.array(parent_education) == 'PhD', 5,
        np.where(np.array(parent_education) == 'Master', 3,
        np.where(np.array(parent_education) == 'Bachelor', 1, 0))
    )
    study_hours_per_week = np.clip(
        np.random.normal(12, 6, n_students) + edu_study_boost,
        0, 40
    ).round(1)
    
    # Previous CGPA (correlated with study hours)
    previous_cgpa = np.clip(
        4.0 + (study_hours_per_week / 40) * 5.0 + np.random.normal(0, 0.8, n_students),
        2.0, 10.0
    ).round(2)
    
    # Participation score
    participation_score = np.clip(
        np.random.normal(5, 2.5, n_students) + (attendance_percentage - 70) / 20,
        0, 10
    ).round(0).astype(int)
    
    # Sleep hours
    sleep_hours = np.clip(
        np.random.normal(7, 1.2, n_students),
        4, 10
    ).round(1)
    
    # Binary features
    extracurricular = np.random.choice(['Yes', 'No'], size=n_students, p=[0.40, 0.60])
    internet_access = np.random.choice(['Yes', 'No'], size=n_students, p=[0.75, 0.25])
    test_preparation = np.random.choice(
        ['Completed', 'Not Completed'],
        size=n_students,
        p=[0.45, 0.55]
    )
    
    # ---------- Internal Marks (0-50) ----------
    # Correlated with study hours, attendance, and participation
    internal_marks = np.clip(
        10 +
        (study_hours_per_week / 40) * 15 +
        (attendance_percentage / 100) * 12 +
        (participation_score / 10) * 5 +
        np.random.normal(0, 5, n_students),
        0, 50
    ).round(0).astype(int)
    
    # ---------- Final Exam Score (0-100) — TARGET ----------
    # Build score from multiple factors with realistic weights
    base_score = 15.0
    
    # Study hours contribution (major factor)
    study_contrib = (study_hours_per_week / 40) * 25
    
    # Attendance contribution
    attendance_contrib = (attendance_percentage / 100) * 20
    
    # Previous CGPA contribution
    cgpa_contrib = (previous_cgpa / 10) * 15
    
    # Internal marks contribution
    internal_contrib = (internal_marks / 50) * 10
    
    # Participation contribution
    participation_contrib = (participation_score / 10) * 5
    
    # Test preparation bonus
    test_prep_bonus = np.where(np.array(test_preparation) == 'Completed', 4, 0)
    
    # Internet access bonus
    internet_bonus = np.where(np.array(internet_access) == 'Yes', 2, 0)
    
    # Sleep quality bonus (optimal around 7-8 hours)
    sleep_bonus = np.where(
        (sleep_hours >= 6.5) & (sleep_hours <= 8.5), 3,
        np.where((sleep_hours >= 5.5) & (sleep_hours <= 9.5), 1, -1)
    )
    
    # Parent education bonus
    parent_edu_bonus = np.where(
        np.array(parent_education) == 'PhD', 3,
        np.where(np.array(parent_education) == 'Master', 2,
        np.where(np.array(parent_education) == 'Bachelor', 1, 0))
    ).astype(float)
    
    # Add noise for realism
    noise = np.random.normal(0, 5, n_students)
    
    final_exam_score = np.clip(
        base_score + study_contrib + attendance_contrib + cgpa_contrib +
        internal_contrib + participation_contrib + test_prep_bonus +
        internet_bonus + sleep_bonus + parent_edu_bonus + noise,
        0, 100
    ).round(1)
    
    # ---------- Performance Level (derived from final score) ----------
    performance_level = np.where(
        final_exam_score >= 75, 'High',
        np.where(final_exam_score >= 50, 'Average', 'Low')
    )
    
    # ---------- Assemble DataFrame ----------
    df = pd.DataFrame({
        'student_id': student_ids,
        'gender': gender,
        'age': age,
        'attendance_percentage': attendance_percentage,
        'study_hours_per_week': study_hours_per_week,
        'previous_cgpa': previous_cgpa,
        'internal_marks': internal_marks,
        'participation_score': participation_score,
        'sleep_hours': sleep_hours,
        'extracurricular': extracurricular,
        'internet_access': internet_access,
        'parent_education': parent_education,
        'family_income': family_income,
        'test_preparation': test_preparation,
        'final_exam_score': final_exam_score,
        'performance_level': performance_level
    })
    
    # ---------- Introduce ~5% missing values for realism ----------
    missing_cols = ['attendance_percentage', 'study_hours_per_week', 'sleep_hours',
                    'participation_score', 'internal_marks', 'previous_cgpa']
    for col in missing_cols:
        mask = np.random.random(n_students) < 0.05
        df.loc[mask, col] = np.nan
    
    # ---------- Add ~2% duplicate rows for preprocessing practice ----------
    n_duplicates = int(n_students * 0.02)
    duplicate_rows = df.sample(n=n_duplicates, random_state=random_seed)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # ---------- Save ----------
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'student_data.csv')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[OK] Dataset generated: {len(df)} records, {df.shape[1]} features")
    print(f"   Saved to: {output_path}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicate rows: {df.duplicated().sum()}")
    print(f"\n   Performance Distribution:")
    print(f"   {df['performance_level'].value_counts().to_string()}")
    
    return df


if __name__ == '__main__':
    df = generate_student_dataset()
    print(f"\n[INFO] Dataset preview:")
    print(df.head(10).to_string())

"""
Module 5: Recommendation System
==================================
Generates personalized performance improvement recommendations
based on student data and feature importance analysis.
"""

import numpy as np


# ============================
# RECOMMENDATION RULES
# ============================

RECOMMENDATION_RULES = [
    {
        'field': 'attendance_percentage',
        'condition': lambda val: val < 60,
        'priority': 'Critical',
        'icon': '[ALERT]',
        'recommendation': 'Your attendance is critically low ({value:.0f}%). Aim for at least 75% attendance immediately. Regular class attendance is one of the strongest predictors of academic success.',
        'impact': 9
    },
    {
        'field': 'attendance_percentage',
        'condition': lambda val: 60 <= val < 75,
        'priority': 'High',
        'icon': '[WARN]',
        'recommendation': 'Your attendance ({value:.0f}%) is below the recommended 75% threshold. Try to attend all classes regularly to improve understanding and scores.',
        'impact': 8
    },
    {
        'field': 'study_hours_per_week',
        'condition': lambda val: val < 5,
        'priority': 'Critical',
        'icon': '[ALERT]',
        'recommendation': 'You are studying only {value:.1f} hours/week. This is very low. Increase to at least 10-15 hours/week with a structured study schedule.',
        'impact': 9
    },
    {
        'field': 'study_hours_per_week',
        'condition': lambda val: 5 <= val < 10,
        'priority': 'High',
        'icon': '[STUDY]',
        'recommendation': 'Your study hours ({value:.1f} hrs/week) could be improved. Aim for 15+ hours/week. Try the Pomodoro technique for better focus.',
        'impact': 7
    },
    {
        'field': 'internal_marks',
        'condition': lambda val: val < 20,
        'priority': 'High',
        'icon': '[NOTE]',
        'recommendation': 'Your internal marks ({value:.0f}/50) need significant improvement. Focus on assignments, class tests, and internal assessments.',
        'impact': 8
    },
    {
        'field': 'internal_marks',
        'condition': lambda val: 20 <= val < 30,
        'priority': 'Medium',
        'icon': '[NOTE]',
        'recommendation': 'Your internal marks ({value:.0f}/50) are average. Put extra effort into assignments and participate in class discussions.',
        'impact': 6
    },
    {
        'field': 'participation_score',
        'condition': lambda val: val < 3,
        'priority': 'High',
        'icon': '[TALK]?',
        'recommendation': 'Your class participation ({value:.0f}/10) is very low. Actively engage in discussions, ask questions, and participate in group activities.',
        'impact': 7
    },
    {
        'field': 'participation_score',
        'condition': lambda val: 3 <= val < 5,
        'priority': 'Medium',
        'icon': '[TALK]?',
        'recommendation': 'Your participation ({value:.0f}/10) can be improved. Try to contribute at least once per class session.',
        'impact': 5
    },
    {
        'field': 'test_preparation',
        'condition': lambda val: val == 'Not Completed',
        'priority': 'High',
        'icon': '[BOOK]',
        'recommendation': 'You have not completed the test preparation course. Completing it can significantly boost your exam scores.',
        'impact': 7
    },
    {
        'field': 'sleep_hours',
        'condition': lambda val: val < 6,
        'priority': 'Medium',
        'icon': '[SLEEP]',
        'recommendation': 'You are sleeping only {value:.1f} hours/night. Aim for 7-8 hours for optimal cognitive performance and memory consolidation.',
        'impact': 6
    },
    {
        'field': 'sleep_hours',
        'condition': lambda val: val > 9,
        'priority': 'Low',
        'icon': '[SLEEP]',
        'recommendation': 'You are sleeping {value:.1f} hours/night, which is above optimal. Consider using some of this time for study or activities.',
        'impact': 3
    },
    {
        'field': 'previous_cgpa',
        'condition': lambda val: val < 5.0,
        'priority': 'High',
        'icon': '[DOWN]',
        'recommendation': 'Your previous CGPA ({value:.2f}) indicates past struggles. Consider joining a study group or seeking tutoring to build stronger foundations.',
        'impact': 7
    },
    {
        'field': 'internet_access',
        'condition': lambda val: val == 'No',
        'priority': 'Medium',
        'icon': '[WEB]',
        'recommendation': 'You lack internet access, which limits access to online learning resources. Visit your institution\'s library or computer lab for supplementary learning.',
        'impact': 5
    },
    {
        'field': 'extracurricular',
        'condition': lambda val: val == 'No',
        'priority': 'Low',
        'icon': '[RUN]',
        'recommendation': 'Consider joining extracurricular activities. They help develop soft skills and can reduce academic stress.',
        'impact': 3
    },
]


def generate_recommendations(student_data, predicted_score=None, predicted_level=None):
    """
    Generate personalized recommendations for a student.
    
    Parameters:
        student_data (dict): Dictionary of student features (raw, unencoded values).
        predicted_score (float): Predicted exam score (optional).
        predicted_level (str): Predicted performance level (optional).
    
    Returns:
        list of recommendation dicts sorted by impact score.
    """
    recommendations = []
    
    for rule in RECOMMENDATION_RULES:
        field = rule['field']
        if field not in student_data:
            continue
        
        value = student_data[field]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        
        try:
            if rule['condition'](value):
                rec_text = rule['recommendation'].format(value=value)
                recommendations.append({
                    'priority': rule['priority'],
                    'icon': rule['icon'],
                    'recommendation': rec_text,
                    'field': field,
                    'impact': rule['impact']
                })
        except (TypeError, ValueError):
            continue
    
    # Add overall performance recommendation
    if predicted_level == 'Low':
        recommendations.append({
            'priority': 'Critical',
            'icon': '[TARGET]',
            'recommendation': f'Your predicted performance level is LOW (predicted score: {predicted_score:.1f}). Immediate action is needed. Focus on the high-priority recommendations above.',
            'field': 'overall',
            'impact': 10
        })
    elif predicted_level == 'Average':
        recommendations.append({
            'priority': 'Medium',
            'icon': '[TARGET]',
            'recommendation': f'Your predicted performance level is AVERAGE (predicted score: {predicted_score:.1f}). With focused effort on the areas above, you can reach HIGH performance.',
            'field': 'overall',
            'impact': 5
        })
    elif predicted_level == 'High':
        recommendations.append({
            'priority': 'Low',
            'icon': '[STAR]',
            'recommendation': f'Great job! Your predicted performance level is HIGH (predicted score: {predicted_score:.1f}). Keep up the excellent work and consider mentoring peers.',
            'field': 'overall',
            'impact': 1
        })
    
    # Sort by impact (highest first)
    recommendations.sort(key=lambda x: x['impact'], reverse=True)
    
    return recommendations


def get_strengths(student_data):
    """
    Identify student strengths based on their data.
    
    Returns:
        list of strength descriptions.
    """
    strengths = []
    
    if student_data.get('attendance_percentage', 0) >= 85:
        strengths.append(('[TARGET]', 'Excellent attendance record'))
    
    if student_data.get('study_hours_per_week', 0) >= 20:
        strengths.append(('[STUDY]', 'Strong study commitment'))
    elif student_data.get('study_hours_per_week', 0) >= 15:
        strengths.append(('[STUDY]', 'Good study habits'))
    
    if student_data.get('internal_marks', 0) >= 35:
        strengths.append(('[NOTE]', 'Strong internal assessment performance'))
    
    if student_data.get('participation_score', 0) >= 7:
        strengths.append(('[TALK]?', 'Active class participation'))
    
    if student_data.get('previous_cgpa', 0) >= 7.5:
        strengths.append(('[CHART]', 'Strong academic history'))
    
    if student_data.get('test_preparation') == 'Completed':
        strengths.append(('[BOOK]', 'Completed test preparation'))
    
    sleep = student_data.get('sleep_hours', 0)
    if 6.5 <= sleep <= 8.5:
        strengths.append((':)', 'Healthy sleep schedule'))
    
    if student_data.get('extracurricular') == 'Yes':
        strengths.append(('[RUN]', 'Well-rounded with extracurricular activities'))
    
    if student_data.get('internet_access') == 'Yes':
        strengths.append(('[WEB]', 'Has access to online learning resources'))
    
    return strengths


def get_summary_stats(recommendations):
    """
    Get summary statistics for recommendations.
    
    Returns:
        dict with priority counts and average impact.
    """
    if not recommendations:
        return {'total': 0, 'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'avg_impact': 0}
    
    priority_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
    for rec in recommendations:
        priority_counts[rec['priority']] = priority_counts.get(rec['priority'], 0) + 1
    
    avg_impact = sum(r['impact'] for r in recommendations) / len(recommendations)
    
    return {
        'total': len(recommendations),
        'critical': priority_counts.get('Critical', 0),
        'high': priority_counts.get('High', 0),
        'medium': priority_counts.get('Medium', 0),
        'low': priority_counts.get('Low', 0),
        'avg_impact': round(avg_impact, 1)
    }


if __name__ == '__main__':
    # Example usage
    sample_student = {
        'attendance_percentage': 62,
        'study_hours_per_week': 7,
        'previous_cgpa': 4.5,
        'internal_marks': 18,
        'participation_score': 3,
        'sleep_hours': 5.5,
        'extracurricular': 'No',
        'internet_access': 'Yes',
        'test_preparation': 'Not Completed'
    }
    
    recs = generate_recommendations(sample_student, predicted_score=42.5, predicted_level='Low')
    
    print("=" * 60)
    print("[TIP] RECOMMENDATIONS FOR SAMPLE STUDENT")
    print("=" * 60)
    for r in recs:
        print(f"\n{r['icon']} [{r['priority']}] {r['recommendation']}")
    
    print("\n\n[STAR] STRENGTHS:")
    strengths = get_strengths(sample_student)
    for icon, s in strengths:
        print(f"  {icon} {s}")
    
    stats = get_summary_stats(recs)
    print(f"\n[CHART] Summary: {stats}")

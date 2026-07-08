"""
Module 2: Exploratory Data Analysis
======================================
Generates comprehensive visualizations for student performance analysis.
All charts are saved to the plots/ directory and can be used in the dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set global style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')


def ensure_plots_dir():
    """Create the plots directory if it doesn't exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_score_distribution(df):
    """Plot the distribution of final exam scores."""
    ensure_plots_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['final_exam_score'].dropna(), bins=30, color='#6C5CE7', edgecolor='white', alpha=0.85)
    axes[0].set_title('Distribution of Final Exam Scores', fontweight='bold')
    axes[0].set_xlabel('Final Exam Score')
    axes[0].set_ylabel('Number of Students')
    axes[0].axvline(df['final_exam_score'].mean(), color='#E17055', linestyle='--', linewidth=2, label=f"Mean: {df['final_exam_score'].mean():.1f}")
    axes[0].legend()
    
    # KDE Plot
    sns.kdeplot(data=df, x='final_exam_score', hue='performance_level', fill=True,
                palette={'High': '#00B894', 'Average': '#FDCB6E', 'Low': '#E17055'},
                ax=axes[1], alpha=0.5)
    axes[1].set_title('Score Distribution by Performance Level', fontweight='bold')
    axes[1].set_xlabel('Final Exam Score')
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'score_distribution.png')
    plt.savefig(filepath)
    plt.close()
    print(f"[OK] Saved: score_distribution.png")
    return filepath


def plot_attendance_distribution(df):
    """Plot attendance percentage distribution."""
    ensure_plots_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['attendance_percentage'].dropna(), bins=25, color='#0984E3', edgecolor='white', alpha=0.85)
    axes[0].set_title('Distribution of Attendance Percentage', fontweight='bold')
    axes[0].set_xlabel('Attendance (%)')
    axes[0].set_ylabel('Number of Students')
    axes[0].axvline(75, color='#D63031', linestyle='--', linewidth=2, label='75% Threshold')
    axes[0].legend()
    
    # Box plot by performance level
    sns.boxplot(data=df, x='performance_level', y='attendance_percentage',
                order=['Low', 'Average', 'High'],
                palette={'High': '#00B894', 'Average': '#FDCB6E', 'Low': '#E17055'},
                ax=axes[1])
    axes[1].set_title('Attendance by Performance Level', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'attendance_distribution.png')
    plt.savefig(filepath)
    plt.close()
    print(f"[OK] Saved: attendance_distribution.png")
    return filepath


def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical features."""
    ensure_plots_dir()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude encoded columns and student_id for cleaner heatmap
    numerical_cols = [c for c in numerical_cols if not c.endswith('_encoded') and c != 'student_id']
    
    corr_matrix = df[numerical_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Heatmap of Numerical Features', fontweight='bold', pad=20)
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'correlation_heatmap.png')
    plt.savefig(filepath)
    plt.close()
    print(f"[OK] Saved: correlation_heatmap.png")
    return filepath


def plot_performance_level_counts(df):
    """Plot performance level distribution."""
    ensure_plots_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'High': '#00B894', 'Average': '#FDCB6E', 'Low': '#E17055'}
    order = ['Low', 'Average', 'High']
    counts = df['performance_level'].value_counts().reindex(order)
    
    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=[colors[x] for x in counts.index],
                       edgecolor='white', linewidth=2)
    axes[0].set_title('Student Count by Performance Level', fontweight='bold')
    axes[0].set_ylabel('Number of Students')
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(val), ha='center', fontweight='bold', fontsize=13)
    
    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=[colors[x] for x in counts.index],
                startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Performance Level Distribution', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'performance_level_counts.png')
    plt.savefig(filepath)
    plt.close()
    print(f"[OK] Saved: performance_level_counts.png")
    return filepath


def plot_study_hours_vs_score(df):
    """Scatter plot: Study hours vs Final exam score."""
    ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'High': '#00B894', 'Average': '#FDCB6E', 'Low': '#E17055'}
    for level in ['Low', 'Average', 'High']:
        mask = df['performance_level'] == level
        ax.scatter(df.loc[mask, 'study_hours_per_week'], df.loc[mask, 'final_exam_score'],
                   c=colors[level], label=level, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    
    # Trend line
    valid = df[['study_hours_per_week', 'final_exam_score']].dropna()
    z = np.polyfit(valid['study_hours_per_week'], valid['final_exam_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['study_hours_per_week'].min(), valid['study_hours_per_week'].max(), 100)
    ax.plot(x_line, p(x_line), '--', color='#2D3436', linewidth=2, label='Trend Line')
    
    ax.set_title('Study Hours vs Final Exam Score', fontweight='bold')
    ax.set_xlabel('Study Hours per Week')
    ax.set_ylabel('Final Exam Score')
    ax.legend()
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'study_hours_vs_score.png')
    plt.savefig(filepath)
    plt.close()
    print(f"[OK] Saved: study_hours_vs_score.png")
    return filepath


def plot_attendance_vs_score(df):
    """Scatter plot: Attendance vs Final exam score."""
    ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'High': '#00B894', 'Average': '#FDCB6E', 'Low': '#E17055'}
    for level in ['Low', 'Average', 'High']:
        mask = df['performance_level'] == level
        ax.scatter(df.loc[mask, 'attendance_percentage'], df.loc[mask, 'final_exam_score'],
                   c=colors[level], label=level, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    
    # Trend line
    valid = df[['attendance_percentage', 'final_exam_score']].dropna()
    z = np.polyfit(valid['attendance_percentage'], valid['final_exam_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['attendance_percentage'].min(), valid['attendance_percentage'].max(), 100)
    ax.plot(x_line, p(x_line), '--', color='#2D3436', linewidth=2, label='Trend Line')
    
    ax.set_title('Attendance vs Final Exam Score', fontweight='bold')
    ax.set_xlabel('Attendance Percentage')
    ax.set_ylabel('Final Exam Score')
    ax.legend()
    ax.axvline(75, color='#D63031', linestyle=':', linewidth=1.5, alpha=0.7, label='75% line')
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'attendance_vs_score.png')
    plt.savefig(filepath)
    plt.close()
    print(f"[OK] Saved: attendance_vs_score.png")
    return filepath


def plot_boxplots_by_category(df):
    """Box plots: Score by gender, parent education, family income."""
    ensure_plots_dir()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # By Gender
    sns.boxplot(data=df, x='gender', y='final_exam_score', ax=axes[0],
                palette=['#6C5CE7', '#FD79A8'])
    axes[0].set_title('Score by Gender', fontweight='bold')
    
    # By Parent Education
    order_edu = ['High School', 'Bachelor', 'Master', 'PhD']
    sns.boxplot(data=df, x='parent_education', y='final_exam_score', ax=axes[1],
                order=order_edu, palette='viridis')
    axes[1].set_title('Score by Parent Education', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    
    # By Family Income
    order_inc = ['Low', 'Medium', 'High']
    sns.boxplot(data=df, x='family_income', y='final_exam_score', ax=axes[2],
                order=order_inc, palette=['#E17055', '#FDCB6E', '#00B894'])
    axes[2].set_title('Score by Family Income', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'boxplots_by_category.png')
    plt.savefig(filepath)
    plt.close()
    print(f"[OK] Saved: boxplots_by_category.png")
    return filepath


def plot_mean_scores_by_category(df):
    """Bar charts showing mean scores by various categories."""
    ensure_plots_dir()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    categories = [
        ('test_preparation', 'Test Preparation'),
        ('internet_access', 'Internet Access'),
        ('extracurricular', 'Extracurricular Activities'),
        ('gender', 'Gender')
    ]
    
    palette = ['#6C5CE7', '#00B894', '#E17055', '#0984E3']
    
    for idx, (col, title) in enumerate(categories):
        ax = axes[idx // 2][idx % 2]
        means = df.groupby(col)['final_exam_score'].mean().sort_values(ascending=False)
        bars = ax.bar(means.index, means.values, color=palette[idx], edgecolor='white',
                      linewidth=2, alpha=0.85)
        ax.set_title(f'Mean Score by {title}', fontweight='bold')
        ax.set_ylabel('Mean Final Score')
        for bar, val in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'mean_scores_by_category.png')
    plt.savefig(filepath)
    plt.close()
    print(f"[OK] Saved: mean_scores_by_category.png")
    return filepath


def plot_pairplot(df):
    """Pair plot of key numerical features colored by performance level."""
    ensure_plots_dir()
    key_features = ['study_hours_per_week', 'attendance_percentage', 'previous_cgpa',
                    'internal_marks', 'final_exam_score', 'performance_level']
    df_subset = df[key_features].dropna()
    
    g = sns.pairplot(df_subset, hue='performance_level',
                     palette={'High': '#00B894', 'Average': '#FDCB6E', 'Low': '#E17055'},
                     diag_kind='kde', plot_kws={'alpha': 0.5, 's': 30},
                     height=2.2, aspect=1.1)
    g.figure.suptitle('Pair Plot of Key Features', fontweight='bold', y=1.02, fontsize=14)
    
    filepath = os.path.join(PLOTS_DIR, 'pairplot.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: pairplot.png")
    return filepath


def plot_study_hours_distribution(df):
    """Plot study hours per week distribution."""
    ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.histplot(data=df, x='study_hours_per_week', hue='performance_level',
                 kde=True, palette={'High': '#00B894', 'Average': '#FDCB6E', 'Low': '#E17055'},
                 alpha=0.5, ax=ax)
    ax.set_title('Study Hours Distribution by Performance Level', fontweight='bold')
    ax.set_xlabel('Study Hours per Week')
    ax.set_ylabel('Count')
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'study_hours_distribution.png')
    plt.savefig(filepath)
    plt.close()
    print(f"[OK] Saved: study_hours_distribution.png")
    return filepath


def generate_all_plots(df):
    """Generate all EDA plots and return their file paths."""
    print("=" * 60)
    print("[CHART] GENERATING EDA VISUALIZATIONS")
    print("=" * 60)
    
    plots = {}
    plots['score_distribution'] = plot_score_distribution(df)
    plots['attendance_distribution'] = plot_attendance_distribution(df)
    plots['correlation_heatmap'] = plot_correlation_heatmap(df)
    plots['performance_level_counts'] = plot_performance_level_counts(df)
    plots['study_hours_vs_score'] = plot_study_hours_vs_score(df)
    plots['attendance_vs_score'] = plot_attendance_vs_score(df)
    plots['boxplots_by_category'] = plot_boxplots_by_category(df)
    plots['mean_scores_by_category'] = plot_mean_scores_by_category(df)
    plots['pairplot'] = plot_pairplot(df)
    plots['study_hours_distribution'] = plot_study_hours_distribution(df)
    
    print(f"\n[OK] All {len(plots)} plots generated and saved to: {PLOTS_DIR}")
    return plots


if __name__ == '__main__':
    from data_preprocessing import load_data, handle_missing_values, remove_duplicates
    
    df = load_data()
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    plots = generate_all_plots(df)

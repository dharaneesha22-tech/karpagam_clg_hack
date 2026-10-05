"""
analytics.py - All Plotly charts and summary statistics for dashboards.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import get_grade, get_performance


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_df(students: list) -> pd.DataFrame:
    """Convert list of student dicts to DataFrame, adding grade/performance cols."""
    if not students:
        return pd.DataFrame()
    df = pd.DataFrame(students)
    if "final_mark" in df.columns:
        df["grade"]       = df["final_mark"].apply(get_grade)
        df["performance"] = df["final_mark"].apply(get_performance)
    return df


def _empty_fig(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        height=350,
    )
    return fig


# ── Summary Statistics ────────────────────────────────────────────────────────

def compute_summary_stats(students: list) -> dict:
    """Return key academic summary stats as a dict."""
    if not students:
        return {
            "total":          0,
            "avg_mark":       0.0,
            "highest_mark":   0.0,
            "lowest_mark":    0.0,
            "avg_attendance": 0.0,
            "pass_pct":       0.0,
            "fail_pct":       0.0,
        }
    df = pd.DataFrame(students)
    marks = df["final_mark"].astype(float)
    att   = df["attendance"].astype(float)
    total = len(df)
    pass_count = int((marks >= 50).sum())
    return {
        "total":          total,
        "avg_mark":       round(float(marks.mean()), 2),
        "highest_mark":   round(float(marks.max()), 2),
        "lowest_mark":    round(float(marks.min()), 2),
        "avg_attendance": round(float(att.mean()), 2),
        "pass_pct":       round(pass_count / total * 100, 2),
        "fail_pct":       round((total - pass_count) / total * 100, 2),
    }


# ── Chart 1: Marks Distribution ───────────────────────────────────────────────

def chart_marks_distribution(students: list) -> go.Figure:
    df = _to_df(students)
    if df.empty or "final_mark" not in df.columns:
        return _empty_fig("No marks data available")
    fig = px.histogram(
        df, x="final_mark", nbins=15,
        title="Marks Distribution",
        labels={"final_mark": "Final Mark", "count": "Number of Students"},
        color_discrete_sequence=["#1f77b4"],
        template="plotly_white",
    )
    fig.update_layout(bargap=0.05, height=380)
    return fig


# ── Chart 2: Attendance vs Marks ──────────────────────────────────────────────

def chart_attendance_vs_marks(students: list) -> go.Figure:
    df = _to_df(students)
    if df.empty or "attendance" not in df.columns:
        return _empty_fig("No attendance data available")
    fig = px.scatter(
        df, x="attendance", y="final_mark",
        color="grade" if "grade" in df.columns else None,
        hover_data=["name", "roll_number"] if "name" in df.columns else None,
        title="Attendance vs Final Marks",
        labels={"attendance": "Attendance (%)", "final_mark": "Final Mark"},
        template="plotly_white",
    )
    fig.update_layout(height=380)
    return fig


# ── Chart 3: Study Hours vs Marks ─────────────────────────────────────────────

def chart_study_hours_vs_marks(students: list) -> go.Figure:
    df = _to_df(students)
    if df.empty or "study_hours" not in df.columns:
        return _empty_fig("No study hours data available")
    fig = px.scatter(
        df, x="study_hours", y="final_mark",
        color="performance" if "performance" in df.columns else None,
        hover_data=["name", "roll_number"] if "name" in df.columns else None,
        title="Study Hours vs Final Marks",
        labels={"study_hours": "Study Hours/Day", "final_mark": "Final Mark"},
        template="plotly_white",
    )
    fig.update_layout(height=380)
    return fig


# ── Chart 4: Grade Distribution ───────────────────────────────────────────────

def chart_grade_distribution(students: list) -> go.Figure:
    df = _to_df(students)
    if df.empty or "grade" not in df.columns:
        return _empty_fig("No grade data available")
    grade_order = ["A+", "A", "B", "C", "D", "F"]
    grade_counts = df["grade"].value_counts().reindex(grade_order, fill_value=0).reset_index()
    grade_counts.columns = ["Grade", "Count"]
    color_map = {
        "A+": "#00c851", "A": "#28a745", "B": "#17a2b8",
        "C": "#ffc107",  "D": "#fd7e14", "F": "#dc3545",
    }
    fig = px.bar(
        grade_counts, x="Grade", y="Count",
        title="Grade Distribution",
        color="Grade",
        color_discrete_map=color_map,
        template="plotly_white",
    )
    fig.update_layout(height=380, showlegend=False)
    return fig


# ── Chart 5: Department Performance ──────────────────────────────────────────

def chart_department_performance(students: list) -> go.Figure:
    df = _to_df(students)
    if df.empty or "department" not in df.columns:
        return _empty_fig("No department data available")
    dept_stats = (
        df.groupby("department")["final_mark"]
        .agg(["mean", "max", "min", "count"])
        .reset_index()
        .rename(columns={"mean": "Average", "max": "Highest",
                         "min": "Lowest", "count": "Students"})
    )
    fig = px.bar(
        dept_stats, x="department", y="Average",
        error_y=None,
        title="Department-wise Average Performance",
        labels={"department": "Department", "Average": "Average Mark"},
        color="Average",
        color_continuous_scale="Blues",
        template="plotly_white",
        hover_data=["Highest", "Lowest", "Students"],
    )
    fig.update_layout(height=380)
    return fig


# ── Chart 6: Year-wise Performance ────────────────────────────────────────────

def chart_year_performance(students: list) -> go.Figure:
    df = _to_df(students)
    if df.empty or "year" not in df.columns:
        return _empty_fig("No year data available")
    year_stats = (
        df.groupby("year")["final_mark"]
        .mean()
        .reset_index()
        .rename(columns={"final_mark": "Average Mark", "year": "Year"})
    )
    fig = px.line(
        year_stats, x="Year", y="Average Mark",
        markers=True,
        title="Year-wise Average Performance",
        labels={"Year": "Year of Study", "Average Mark": "Average Mark"},
        template="plotly_white",
        line_shape="spline",
    )
    fig.update_traces(
        marker=dict(size=10, color="#1f77b4"),
        line=dict(color="#1f77b4", width=3),
    )
    fig.update_layout(height=380)
    return fig


# ── Chart 7: Prediction History ───────────────────────────────────────────────

def chart_prediction_history(predictions: list, roll_number: str = None) -> go.Figure:
    if not predictions:
        return _empty_fig("No prediction history available")
    df = pd.DataFrame(predictions)
    if df.empty or "predicted_mark" not in df.columns:
        return _empty_fig("No prediction data available")

    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df = df.sort_values("prediction_date")

    title = f"Prediction History — {roll_number}" if roll_number else "Prediction History"

    if roll_number:
        fig = px.line(
            df, x="prediction_date", y="predicted_mark",
            markers=True,
            title=title,
            labels={"prediction_date": "Date", "predicted_mark": "Predicted Mark"},
            template="plotly_white",
            line_shape="spline",
        )
        fig.add_hline(y=50, line_dash="dash", line_color="red",
                      annotation_text="Pass Line (50)")
    else:
        fig = px.scatter(
            df, x="prediction_date", y="predicted_mark",
            color="roll_number" if "roll_number" in df.columns else None,
            title=title,
            labels={"prediction_date": "Date", "predicted_mark": "Predicted Mark"},
            template="plotly_white",
        )
    fig.update_layout(height=380)
    return fig


# ── Chart 8: Student Radar (individual performance) ──────────────────────────

def chart_student_radar(student: dict) -> go.Figure:
    """Spider/radar chart for a single student's academic metrics."""
    categories = [
        "Attendance", "Internal Mark", "Assignment Mark",
        "Previous Mark", "Study Hours (×10)", "Assign. Completion",
    ]
    values = [
        float(student.get("attendance", 0)),
        float(student.get("internal_mark", 0)),
        float(student.get("assignment_mark", 0)),
        float(student.get("previous_mark", 0)),
        min(float(student.get("study_hours", 0)) * 10, 100),
        float(student.get("assignment_completion", 0)),
    ]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        name="Student Metrics",
        line_color="#1f77b4",
        fillcolor="rgba(31,119,180,0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Academic Performance Radar",
        height=400,
        template="plotly_white",
        showlegend=False,
    )
    return fig


# ── Chart 9: Pass/Fail Pie ────────────────────────────────────────────────────

def chart_pass_fail_pie(students: list) -> go.Figure:
    df = _to_df(students)
    if df.empty or "final_mark" not in df.columns:
        return _empty_fig("No data available")
    pass_count = int((df["final_mark"].astype(float) >= 50).sum())
    fail_count = len(df) - pass_count
    fig = px.pie(
        names=["Pass", "Fail"],
        values=[pass_count, fail_count],
        title="Pass / Fail Distribution",
        color_discrete_sequence=["#28a745", "#dc3545"],
        template="plotly_white",
        hole=0.4,
    )
    fig.update_layout(height=350)
    return fig


# ── Chart 10: Internal vs Assignment Mark ─────────────────────────────────────

def chart_internal_vs_assignment(students: list) -> go.Figure:
    df = _to_df(students)
    if df.empty:
        return _empty_fig("No data available")
    fig = px.scatter(
        df, x="internal_mark", y="assignment_mark",
        color="grade" if "grade" in df.columns else None,
        size="final_mark" if "final_mark" in df.columns else None,
        hover_data=["name", "roll_number"] if "name" in df.columns else None,
        title="Internal Mark vs Assignment Mark",
        labels={"internal_mark": "Internal Mark", "assignment_mark": "Assignment Mark"},
        template="plotly_white",
    )
    fig.update_layout(height=380)
    return fig

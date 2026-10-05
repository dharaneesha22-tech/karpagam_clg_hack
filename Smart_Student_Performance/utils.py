"""
utils.py - Grade calculation, performance labels, recommendations, validators.
"""


# ── Grade & Performance ───────────────────────────────────────────────────────

def get_grade(mark: float) -> str:
    """Return letter grade for a numeric mark (0-100)."""
    mark = float(mark)
    if mark >= 90:
        return "A+"
    elif mark >= 80:
        return "A"
    elif mark >= 70:
        return "B"
    elif mark >= 60:
        return "C"
    elif mark >= 50:
        return "D"
    else:
        return "F"


def get_performance(mark: float) -> str:
    """Return performance category for a numeric mark."""
    mark = float(mark)
    if mark >= 85:
        return "Excellent"
    elif mark >= 70:
        return "Good"
    elif mark >= 50:
        return "Average"
    else:
        return "Needs Improvement"


def get_grade_color(grade: str) -> str:
    """Return a CSS hex colour for a grade label."""
    colors = {
        "A+": "#00c851",
        "A":  "#28a745",
        "B":  "#17a2b8",
        "C":  "#ffc107",
        "D":  "#fd7e14",
        "F":  "#dc3545",
    }
    return colors.get(grade, "#6c757d")


def get_performance_color(performance: str) -> str:
    colors = {
        "Excellent":         "#00c851",
        "Good":              "#17a2b8",
        "Average":           "#ffc107",
        "Needs Improvement": "#dc3545",
    }
    return colors.get(performance, "#6c757d")


# ── Recommendations ───────────────────────────────────────────────────────────

def get_recommendations(student: dict, predicted_mark: float = None) -> list:
    """
    Return a list of recommendation strings based on student data.
    student dict keys: attendance, internal_mark, assignment_completion,
                       study_hours, assignment_mark, previous_mark.
    """
    recs = []

    attendance = float(student.get("attendance", 100))
    internal   = float(student.get("internal_mark", 50))
    assign_c   = float(student.get("assignment_completion", 100))
    hours      = float(student.get("study_hours", 2))

    if attendance < 75:
        recs.append(
            "⚠️ **Attendance is below 75%.** Regular attendance is critical for "
            "academic success. Try to attend all scheduled classes."
        )
    elif attendance < 85:
        recs.append(
            "📅 Attendance is slightly low. Aim for above 85% to stay on track."
        )

    if internal < 40:
        recs.append(
            "📝 **Internal mark is below 40.** Focus on internal assessments, "
            "class tests, and mid-term examinations to boost your score."
        )
    elif internal < 60:
        recs.append(
            "📚 Internal mark can be improved. Review lecture notes and practice "
            "past internal exam questions."
        )

    if assign_c < 60:
        recs.append(
            "📋 **Assignment completion is below 60%.** Complete all pending "
            "assignments on time. Assignments contribute significantly to your grade."
        )
    elif assign_c < 80:
        recs.append(
            "✏️ Try to submit all assignments promptly to maintain a higher score."
        )

    if hours < 2:
        recs.append(
            "⏰ **Study hours are very low (less than 2 hrs/day).** Increase focused "
            "daily study time. Aim for at least 3-4 hours of dedicated study."
        )
    elif hours < 3:
        recs.append(
            "📖 Consider increasing your daily study time to at least 3 hours for "
            "better retention."
        )

    if predicted_mark is not None:
        if predicted_mark < 50:
            recs.append(
                "🚨 **Predicted mark is below 50.** Immediate academic improvement is "
                "required. Consider seeking help from faculty or academic counselling."
            )
        elif predicted_mark < 70:
            recs.append(
                "💡 There is room to improve your predicted mark. Stay consistent and "
                "revise regularly."
            )
        elif predicted_mark >= 85:
            recs.append(
                "🌟 **Excellent performance!** Maintain your current study habits and "
                "keep up the great work."
            )
        else:
            recs.append(
                "👍 Good performance. A little more effort can push you to the Excellent "
                "category."
            )

    if not recs:
        recs.append("✅ Great job! Keep maintaining your current academic performance.")

    return recs


# ── Validation ────────────────────────────────────────────────────────────────

def validate_mark(value, field_name: str = "Mark") -> tuple:
    """
    Validate that value is a number in [0, 100].
    Returns (True, float_value) or (False, error_message).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, f"{field_name} must be a number."
    if v < 0 or v > 100:
        return False, f"{field_name} must be between 0 and 100."
    return True, v


def validate_attendance(value) -> tuple:
    return validate_mark(value, "Attendance")


def validate_study_hours(value) -> tuple:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, "Study hours must be a number."
    if v < 0:
        return False, "Study hours cannot be negative."
    return True, v


def validate_year(value) -> tuple:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return False, "Year must be a positive integer."
    if v < 1:
        return False, "Year must be a positive integer."
    return True, v


def validate_semester(value) -> tuple:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return False, "Semester must be a positive integer."
    if v < 1:
        return False, "Semester must be a positive integer."
    return True, v


def validate_student_form(name, department, year, semester, attendance,
                           internal_mark, assignment_mark, previous_mark,
                           study_hours, assignment_completion, final_mark=0):
    """
    Validate all student form fields.
    Returns (True, cleaned_dict) or (False, error_message).
    """
    errors = []

    if not str(name).strip():
        errors.append("Name is required.")
    if not str(department).strip():
        errors.append("Department is required.")

    ok, yr = validate_year(year)
    if not ok:
        errors.append(yr)

    ok, sem = validate_semester(semester)
    if not ok:
        errors.append(sem)

    ok, att = validate_attendance(attendance)
    if not ok:
        errors.append(att)

    ok, im = validate_mark(internal_mark, "Internal Mark")
    if not ok:
        errors.append(im)

    ok, am = validate_mark(assignment_mark, "Assignment Mark")
    if not ok:
        errors.append(am)

    ok, pm = validate_mark(previous_mark, "Previous Mark")
    if not ok:
        errors.append(pm)

    ok, sh = validate_study_hours(study_hours)
    if not ok:
        errors.append(sh)

    ok, ac = validate_mark(assignment_completion, "Assignment Completion")
    if not ok:
        errors.append(ac)

    ok, fm = validate_mark(final_mark, "Final Mark")
    if not ok:
        errors.append(fm)

    if errors:
        return False, "; ".join(errors)

    return True, {
        "name":                 str(name).strip(),
        "department":           str(department).strip(),
        "year":                 yr,
        "semester":             sem,
        "attendance":           att,
        "internal_mark":        im,
        "assignment_mark":      am,
        "previous_mark":        pm,
        "study_hours":          sh,
        "assignment_completion": ac,
        "final_mark":           fm,
    }


# ── At-Risk Detection ─────────────────────────────────────────────────────────

def get_risk_reasons(student: dict, predicted_mark: float = None) -> list:
    """Return list of risk reason strings, or empty list if not at risk."""
    reasons = []
    if float(student.get("attendance", 100)) < 75:
        reasons.append("Low Attendance (<75%)")
    if float(student.get("internal_mark", 50)) < 40:
        reasons.append("Low Internal Mark (<40)")
    if float(student.get("assignment_completion", 100)) < 50:
        reasons.append("Low Assignment Completion (<50%)")
    if predicted_mark is not None and float(predicted_mark) < 50:
        reasons.append("Low Predicted Mark (<50)")
    return reasons


def is_at_risk(student: dict, predicted_mark: float = None) -> bool:
    return len(get_risk_reasons(student, predicted_mark)) > 0


# ── CSV Column Spec ───────────────────────────────────────────────────────────

REQUIRED_CSV_COLUMNS = [
    "roll_number", "name", "department", "year", "semester",
    "attendance", "internal_mark", "assignment_mark", "previous_mark",
    "study_hours", "assignment_completion", "final_mark",
]


def validate_csv_dataframe(df) -> tuple:
    """
    Validate an imported CSV DataFrame.
    Returns (True, cleaned_df) or (False, error_message).
    """
    import pandas as pd

    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"

    df = df[REQUIRED_CSV_COLUMNS].copy()

    # Strip whitespace from text columns
    for col in ["roll_number", "name", "department"]:
        df[col] = df[col].astype(str).str.strip()

    # Validate numeric ranges
    numeric_checks = {
        "year":                 (1, 10),
        "semester":             (1, 12),
        "attendance":           (0, 100),
        "internal_mark":        (0, 100),
        "assignment_mark":      (0, 100),
        "previous_mark":        (0, 100),
        "study_hours":          (0, 24),
        "assignment_completion":(0, 100),
        "final_mark":           (0, 100),
    }
    for col, (lo, hi) in numeric_checks.items():
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception:
            return False, f"Column '{col}' contains non-numeric values."
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if not out_of_range.empty:
            return False, (
                f"Column '{col}' has values outside [{lo}, {hi}] "
                f"in rows: {out_of_range.index.tolist()[:5]}"
            )

    if df["roll_number"].duplicated().any():
        dupes = df[df["roll_number"].duplicated()]["roll_number"].tolist()
        return False, f"Duplicate roll numbers in CSV: {dupes[:5]}"

    return True, df

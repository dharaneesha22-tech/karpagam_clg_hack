"""
streamlit_app.py - Smart Student Performance & Mark Prediction System
Main entry point. Run with: streamlit run streamlit_app.py
"""

import io
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Bootstrap: ensure folders exist before any module import ──────────────────
BASE_DIR = Path(__file__).parent
(BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "models").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "database").mkdir(parents=True, exist_ok=True)

# ── Internal imports ──────────────────────────────────────────────────────────
from database import (
    init_db,
    add_student,
    get_all_students,
    get_student_by_roll,
    update_student,
    delete_student,
    get_students_count,
    student_exists,
    get_all_faculty,
    add_faculty,
    delete_faculty,
    get_faculty_count,
    faculty_exists,
    get_faculty_by_id,
    save_prediction,
    get_predictions_by_roll,
    get_all_predictions,
    get_predictions_count,
    get_latest_prediction,
    create_user,
    user_exists,
    update_user_password,
    delete_user,
    bulk_add_students,
)
from authentication import (
    create_default_accounts,
    init_session,
    render_login_page,
    logout,
    is_logged_in,
    get_role,
    get_username,
)
from ml_model import (
    train_models,
    predict_mark,
    get_best_model_name,
    ensure_model_exists,
    load_dataset,
    get_model_metadata,
    FEATURE_COLS,
    MODEL_PATH,
    CSV_PATH,
)
from analytics import (
    compute_summary_stats,
    chart_marks_distribution,
    chart_attendance_vs_marks,
    chart_study_hours_vs_marks,
    chart_grade_distribution,
    chart_department_performance,
    chart_year_performance,
    chart_prediction_history,
    chart_student_radar,
    chart_pass_fail_pie,
    chart_internal_vs_assignment,
)
from utils import (
    get_grade,
    get_performance,
    get_recommendations,
    get_risk_reasons,
    is_at_risk,
    validate_student_form,
    validate_csv_dataframe,
    REQUIRED_CSV_COLUMNS,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Student Performance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background:#f8f9fa; border-radius:8px; padding:8px; }
    .risk-badge { background:#dc3545; color:white; border-radius:12px;
                  padding:2px 10px; font-size:0.8rem; }
    .good-badge { background:#28a745; color:white; border-radius:12px;
                  padding:2px 10px; font-size:0.8rem; }
    .sidebar-title { font-size:1.1rem; font-weight:700; color:#1f77b4; }
    div[data-testid="stSidebarNav"] { display:none; }
</style>
""", unsafe_allow_html=True)


# ── Initialization ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing system…")
def initialize_system():
    """Run once per process: DB, default accounts, sample data, model."""
    init_db()
    create_default_accounts()

    # Seed default students into DB if empty
    if get_students_count() == 0:
        _seed_default_students()

    # Ensure model exists
    ensure_model_exists()
    return True


def _seed_default_students():
    """Insert the 30 sample students from CSV into the DB."""
    try:
        df = load_dataset()
        records = df.to_dict(orient="records")
        bulk_add_students(records)
    except Exception:
        pass


# ── Sidebar Navigation ────────────────────────────────────────────────────────

def render_student_sidebar():
    with st.sidebar:
        st.markdown('<p class="sidebar-title">🎓 Student Portal</p>', unsafe_allow_html=True)
        st.write(f"**{get_username()}**")
        st.markdown("---")
        pages = {
            "dashboard":          "🏠 Dashboard",
            "my_performance":     "📊 My Performance",
            "predict_mark":       "🔮 Predict Mark",
            "prediction_history": "📜 Prediction History",
        }
        for key, label in pages.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state["page"] = key
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()


def render_faculty_sidebar():
    with st.sidebar:
        st.markdown('<p class="sidebar-title">👨‍🏫 Faculty Portal</p>', unsafe_allow_html=True)
        st.write(f"**{get_username()}**")
        st.markdown("---")
        pages = {
            "dashboard":          "🏠 Dashboard",
            "students":           "👥 Students",
            "add_student":        "➕ Add Student",
            "analytics":          "📈 Analytics",
            "at_risk":            "⚠️ At-Risk Students",
            "model_training":     "🤖 Model Training",
            "predictions":        "🔮 Predictions",
        }
        for key, label in pages.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state["page"] = key
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()


def render_admin_sidebar():
    with st.sidebar:
        st.markdown('<p class="sidebar-title">🛡️ Admin Portal</p>', unsafe_allow_html=True)
        st.write(f"**{get_username()}**")
        st.markdown("---")
        pages = {
            "dashboard":          "🏠 Dashboard",
            "students":           "👥 Students",
            "faculty_management": "👨‍💼 Faculty Management",
            "analytics":          "📈 Analytics",
            "at_risk":            "⚠️ At-Risk Students",
            "model_training":     "🤖 Model Training",
            "predictions":        "🔮 Predictions",
            "import_data":        "📥 Import Data",
            "export_data":        "📤 Export Data",
        }
        for key, label in pages.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state["page"] = key
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()


# ── Shared Components ─────────────────────────────────────────────────────────

def render_kpi_row(stats: dict):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Students",   stats["total"])
    c2.metric("Avg Mark",         f"{stats['avg_mark']:.1f}")
    c3.metric("Highest Mark",     f"{stats['highest_mark']:.1f}")
    c4.metric("Lowest Mark",      f"{stats['lowest_mark']:.1f}")
    c5.metric("Avg Attendance",   f"{stats['avg_attendance']:.1f}%")
    c6.metric("Pass %",           f"{stats['pass_pct']:.1f}%")


def _student_form(defaults: dict = None, key_prefix: str = "form"):
    """Render student data form fields. Returns raw values dict."""
    d = defaults or {}
    departments = ["Computer Science", "Electronics", "Mechanical", "Civil",
                   "Information Technology", "Electrical", "Chemical", "Biotechnology"]

    col1, col2 = st.columns(2)
    with col1:
        name       = st.text_input("Student Name *", value=d.get("name", ""), key=f"{key_prefix}_name")
        dept_list  = departments if d.get("department", "") in departments \
                     else [d.get("department", departments[0])] + departments
        department = st.selectbox("Department *",
                                  options=dept_list,
                                  index=dept_list.index(d.get("department", departments[0])) if d.get("department","") in dept_list else 0,
                                  key=f"{key_prefix}_dept")
        year       = st.number_input("Year *", min_value=1, max_value=10,
                                     value=int(d.get("year", 1)), key=f"{key_prefix}_year")
        semester   = st.number_input("Semester *", min_value=1, max_value=12,
                                     value=int(d.get("semester", 1)), key=f"{key_prefix}_sem")
        attendance = st.number_input("Attendance (%) *", min_value=0.0, max_value=100.0,
                                     value=float(d.get("attendance", 75.0)),
                                     step=0.5, key=f"{key_prefix}_att")
        study_hours = st.number_input("Study Hours/Day *", min_value=0.0, max_value=24.0,
                                      value=float(d.get("study_hours", 2.0)),
                                      step=0.5, key=f"{key_prefix}_sh")
    with col2:
        internal_mark         = st.number_input("Internal Mark *", 0.0, 100.0,
                                                value=float(d.get("internal_mark", 50.0)),
                                                step=0.5, key=f"{key_prefix}_im")
        assignment_mark       = st.number_input("Assignment Mark *", 0.0, 100.0,
                                                value=float(d.get("assignment_mark", 50.0)),
                                                step=0.5, key=f"{key_prefix}_am")
        previous_mark         = st.number_input("Previous Mark *", 0.0, 100.0,
                                                value=float(d.get("previous_mark", 50.0)),
                                                step=0.5, key=f"{key_prefix}_pm")
        assignment_completion = st.number_input("Assignment Completion (%) *", 0.0, 100.0,
                                                value=float(d.get("assignment_completion", 80.0)),
                                                step=0.5, key=f"{key_prefix}_ac")
        final_mark            = st.number_input("Final Mark", 0.0, 100.0,
                                                value=float(d.get("final_mark", 0.0)),
                                                step=0.5, key=f"{key_prefix}_fm")
    return dict(
        name=name, department=department, year=year, semester=semester,
        attendance=attendance, internal_mark=internal_mark,
        assignment_mark=assignment_mark, previous_mark=previous_mark,
        study_hours=study_hours, assignment_completion=assignment_completion,
        final_mark=final_mark,
    )


# ═════════════════════════════════════════════════════════════════════════════
# STUDENT PAGES
# ═════════════════════════════════════════════════════════════════════════════

def page_student_dashboard():
    username = get_username()
    student  = get_student_by_roll(username)

    st.title("🏠 Student Dashboard")

    if not student:
        st.warning(
            "Your student record has not been added to the system yet. "
            "Please contact your faculty."
        )
        return

    # Profile header
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.subheader(student["name"])
        st.write(f"**Roll Number:** {student['roll_number']}")
        st.write(f"**Department:** {student['department']}")
    with col2:
        st.write(f"**Year:** {student['year']}")
        st.write(f"**Semester:** {student['semester']}")
    with col3:
        latest = get_latest_prediction(username)
        if latest:
            st.write(f"**Last Prediction:** {latest['predicted_mark']:.1f}")
            st.write(f"**Grade:** {latest['grade']}")
            st.write(f"**Performance:** {latest['performance']}")

    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Attendance",            f"{student['attendance']:.1f}%")
    c2.metric("Internal Mark",         f"{student['internal_mark']:.1f}")
    c3.metric("Assignment Mark",       f"{student['assignment_mark']:.1f}")
    c4.metric("Study Hours/Day",       f"{student['study_hours']:.1f}")
    c5.metric("Assignment Completion", f"{student['assignment_completion']:.1f}%")

    if latest:
        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicted Final Mark", f"{latest['predicted_mark']:.1f}")
        k2.metric("Grade",                latest["grade"])
        k3.metric("Performance",          latest["performance"])
        k4.metric("Attendance",           f"{student['attendance']:.1f}%")

    # Recommendations
    st.markdown("---")
    st.subheader("💡 Personalised Recommendations")
    pred_mark = latest["predicted_mark"] if latest else None
    recs = get_recommendations(student, pred_mark)
    for rec in recs:
        st.markdown(rec)


def page_my_performance():
    username = get_username()
    student  = get_student_by_roll(username)

    st.title("📊 My Performance")

    if not student:
        st.warning("Student record not found. Contact faculty.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Academic Metrics")
        metrics = {
            "Attendance (%)":          student["attendance"],
            "Internal Mark":           student["internal_mark"],
            "Assignment Mark":         student["assignment_mark"],
            "Previous Mark":           student["previous_mark"],
            "Study Hours/Day":         student["study_hours"],
            "Assignment Completion (%)": student["assignment_completion"],
            "Final Mark":              student["final_mark"],
        }
        for label, val in metrics.items():
            st.metric(label, f"{val:.1f}")

    with col2:
        fig = chart_student_radar(student)
        st.plotly_chart(fig, use_container_width=True)

    # Grade context
    st.markdown("---")
    grade = get_grade(student["final_mark"])
    perf  = get_performance(student["final_mark"])
    g1, g2 = st.columns(2)
    g1.metric("Current Grade",       grade)
    g2.metric("Performance Category", perf)


def page_predict_mark():
    username = get_username()
    student  = get_student_by_roll(username)

    st.title("🔮 Predict Final Mark")

    if not student:
        st.warning("Student record not found. Contact faculty.")
        return

    st.write("Click **Predict Final Mark** to run the AI model on your current data.")

    with st.expander("📋 Current Data Used for Prediction", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Attendance",       f"{student['attendance']:.1f}%")
        col2.metric("Internal Mark",    f"{student['internal_mark']:.1f}")
        col3.metric("Assignment Mark",  f"{student['assignment_mark']:.1f}")
        col1.metric("Previous Mark",    f"{student['previous_mark']:.1f}")
        col2.metric("Study Hours/Day",  f"{student['study_hours']:.1f}")
        col3.metric("Assignment Compl.", f"{student['assignment_completion']:.1f}%")

    if st.button("🔮 Predict Final Mark", type="primary"):
        try:
            ensure_model_exists()
            predicted = predict_mark(student)
            grade     = get_grade(predicted)
            perf      = get_performance(predicted)
            model_name = get_best_model_name()

            save_prediction(username, predicted, grade, perf, model_name)

            st.success("Prediction complete!")
            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Final Mark", f"{predicted:.2f}")
            r2.metric("Grade",                grade)
            r3.metric("Performance",          perf)

            st.info(f"Model used: **{model_name}**")

            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            recs = get_recommendations(student, predicted)
            for rec in recs:
                st.markdown(rec)

        except Exception as e:
            st.error(f"Prediction failed. Please try again. ({e})")


def page_student_prediction_history():
    username = get_username()
    st.title("📜 My Prediction History")

    preds = get_predictions_by_roll(username)
    if not preds:
        st.info("No predictions yet. Go to **Predict Mark** to generate your first prediction.")
        return

    st.plotly_chart(
        chart_prediction_history(preds, username),
        use_container_width=True,
    )

    df = pd.DataFrame(preds)[["prediction_date", "predicted_mark", "grade",
                               "performance", "model_name"]]
    df.columns = ["Date", "Predicted Mark", "Grade", "Performance", "Model"]
    st.dataframe(df, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# FACULTY PAGES
# ═════════════════════════════════════════════════════════════════════════════

def page_faculty_dashboard():
    st.title("🏠 Faculty Dashboard")
    students = get_all_students()
    stats    = compute_summary_stats(students)
    render_kpi_row(stats)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_marks_distribution(students), use_container_width=True)
    with col2:
        st.plotly_chart(chart_pass_fail_pie(students), use_container_width=True)


def page_students_list():
    st.title("👥 Students")

    students = get_all_students()

    # Search & Filter
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        search = st.text_input("🔍 Search by Roll Number or Name", key="stu_search")
    with col2:
        depts = ["All"] + sorted(list({s["department"] for s in students}))
        dept_filter = st.selectbox("Department", depts, key="stu_dept")
    with col3:
        years = ["All"] + sorted(list({str(s["year"]) for s in students}))
        year_filter = st.selectbox("Year", years, key="stu_year")
    with col4:
        sems = ["All"] + sorted(list({str(s["semester"]) for s in students}))
        sem_filter = st.selectbox("Semester", sems, key="stu_sem")

    filtered = students
    if search:
        q = search.lower()
        filtered = [s for s in filtered
                    if q in s["roll_number"].lower() or q in s["name"].lower()]
    if dept_filter != "All":
        filtered = [s for s in filtered if s["department"] == dept_filter]
    if year_filter != "All":
        filtered = [s for s in filtered if str(s["year"]) == year_filter]
    if sem_filter != "All":
        filtered = [s for s in filtered if str(s["semester"]) == sem_filter]

    st.write(f"Showing **{len(filtered)}** student(s)")

    if not filtered:
        st.info("No students match the current filters.")
        return

    df = pd.DataFrame(filtered)[[
        "roll_number", "name", "department", "year", "semester",
        "attendance", "internal_mark", "assignment_mark",
        "previous_mark", "study_hours", "assignment_completion", "final_mark",
    ]]
    df.columns = [
        "Roll No", "Name", "Department", "Year", "Sem",
        "Attendance", "Internal", "Assignment",
        "Previous", "Study Hrs", "Assign. %", "Final",
    ]
    st.dataframe(df, use_container_width=True)

    # Edit / Delete
    st.markdown("---")
    st.subheader("Edit or Delete a Student")
    roll_options = [s["roll_number"] for s in filtered]
    selected_roll = st.selectbox("Select Roll Number", roll_options, key="edit_roll_sel")

    if selected_roll:
        stu = get_student_by_roll(selected_roll)
        if stu:
            tab_edit, tab_delete = st.tabs(["✏️ Edit", "🗑️ Delete"])

            with tab_edit:
                with st.form(key=f"edit_form_{selected_roll}"):
                    form_vals = _student_form(defaults=stu, key_prefix=f"edit_{selected_roll}")
                    submitted = st.form_submit_button("💾 Save Changes", type="primary")
                    if submitted:
                        ok, result = validate_student_form(**form_vals)
                        if not ok:
                            st.error(result)
                        else:
                            success = update_student(
                                selected_roll,
                                result["name"], result["department"],
                                result["year"], result["semester"],
                                result["attendance"], result["internal_mark"],
                                result["assignment_mark"], result["previous_mark"],
                                result["study_hours"], result["assignment_completion"],
                                result["final_mark"],
                            )
                            if success:
                                st.success("Student updated successfully!")
                                st.rerun()
                            else:
                                st.error("Update failed. Please try again.")

            with tab_delete:
                st.warning(
                    f"You are about to permanently delete **{stu['name']}** "
                    f"({selected_roll}). This cannot be undone."
                )
                confirm = st.checkbox(f"I confirm deletion of {selected_roll}", key="del_confirm")
                if st.button("🗑️ Delete Student", type="primary", disabled=not confirm):
                    if delete_student(selected_roll):
                        # Also remove user account if exists
                        delete_user(selected_roll)
                        st.success(f"Student {selected_roll} deleted.")
                        st.rerun()
                    else:
                        st.error("Deletion failed.")


def page_add_student():
    st.title("➕ Add Student")

    with st.form("add_student_form"):
        roll_number = st.text_input("Roll Number *", placeholder="e.g. ST031", key="add_roll")
        form_vals   = _student_form(key_prefix="add")

        create_login = st.checkbox(
            "Create student login account (password = Roll@123)", value=True,
            key="add_create_login",
        )
        submitted = st.form_submit_button("➕ Add Student", type="primary")

    if submitted:
        if not roll_number.strip():
            st.error("Roll number is required.")
            return
        roll_number = roll_number.strip().upper()
        if student_exists(roll_number):
            st.error(f"Roll number **{roll_number}** already exists.")
            return

        ok, result = validate_student_form(**form_vals)
        if not ok:
            st.error(result)
            return

        added = add_student(
            roll_number, result["name"], result["department"],
            result["year"], result["semester"],
            result["attendance"], result["internal_mark"],
            result["assignment_mark"], result["previous_mark"],
            result["study_hours"], result["assignment_completion"],
            result["final_mark"],
        )
        if added:
            if create_login and not user_exists(roll_number):
                create_user(roll_number, f"{roll_number}@123", "student")
            st.success(f"Student **{roll_number}** added successfully!")
        else:
            st.error(f"Failed to add student. Roll number may already exist.")


def page_faculty_analytics():
    st.title("📈 Analytics")
    students = get_all_students()

    if not students:
        st.info("No student data available for analytics.")
        return

    stats = compute_summary_stats(students)
    render_kpi_row(stats)

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Marks Distribution", "Attendance vs Marks",
        "Study Hours", "Grade Distribution",
        "Department", "Year-wise",
    ])
    with tab1:
        st.plotly_chart(chart_marks_distribution(students), use_container_width=True)
    with tab2:
        st.plotly_chart(chart_attendance_vs_marks(students), use_container_width=True)
    with tab3:
        st.plotly_chart(chart_study_hours_vs_marks(students), use_container_width=True)
    with tab4:
        st.plotly_chart(chart_grade_distribution(students), use_container_width=True)
    with tab5:
        st.plotly_chart(chart_department_performance(students), use_container_width=True)
    with tab6:
        st.plotly_chart(chart_year_performance(students), use_container_width=True)


def page_at_risk():
    st.title("⚠️ At-Risk Students")

    students = get_all_students()
    if not students:
        st.info("No student data available.")
        return

    at_risk_rows = []
    for s in students:
        latest = get_latest_prediction(s["roll_number"])
        pred_mark = latest["predicted_mark"] if latest else None
        reasons = get_risk_reasons(s, pred_mark)
        if reasons:
            at_risk_rows.append({
                "Roll Number":   s["roll_number"],
                "Name":          s["name"],
                "Attendance":    s["attendance"],
                "Internal Mark": s["internal_mark"],
                "Predicted Mark": pred_mark if pred_mark is not None else "N/A",
                "Risk Reasons":  ", ".join(reasons),
            })

    if not at_risk_rows:
        st.success("🎉 No at-risk students found! All students are performing well.")
        return

    st.warning(f"Found **{len(at_risk_rows)}** at-risk student(s).")
    df = pd.DataFrame(at_risk_rows)
    st.dataframe(df, use_container_width=True)

    # Download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download At-Risk Report",
        data=csv_bytes,
        file_name="at_risk_students.csv",
        mime="text/csv",
    )


def page_model_training():
    st.title("🤖 Model Training")

    meta = get_model_metadata()
    if meta:
        st.success(f"Current best model: **{meta.get('best_model_name','N/A')}** "
                   f"(R² = {meta.get('best_r2', 'N/A')})")
        with st.expander("📋 Model Metadata", expanded=False):
            st.json(meta)

    if MODEL_PATH.exists():
        st.info(f"Model file: `models/best_model.pkl`  ✅")
    else:
        st.warning("No trained model found. Click below to train.")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        use_db_data = st.checkbox(
            "Use student records from database for training",
            value=False, key="train_use_db",
        )
    with col2:
        uploaded_file = st.file_uploader(
            "Or upload a custom CSV for training",
            type=["csv"], key="train_upload",
        )

    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models… this may take a moment."):
            try:
                df = None
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                elif use_db_data:
                    students = get_all_students()
                    if students:
                        df = pd.DataFrame(students)
                    else:
                        st.warning("No student records in DB. Using default CSV.")

                results = train_models(df)

                st.success(f"Training complete! Best model: **{results['best_name']}** "
                           f"(R² = {results['best_r2']})")
                st.write(f"Training samples: {results['n_train']} | "
                         f"Test samples: {results['n_test']}")

                st.markdown("### Model Comparison")
                rows = []
                for name, m in results["results"].items():
                    rows.append({
                        "Model":  name,
                        "MAE":    m["MAE"],
                        "MSE":    m["MSE"],
                        "RMSE":   m["RMSE"],
                        "R²":     m["R2"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            except Exception as e:
                st.error(f"Training failed: {e}")


def page_predictions_view(role: str = "faculty"):
    st.title("🔮 Predictions")
    predictions = get_all_predictions()

    if not predictions:
        st.info("No predictions recorded yet.")
        return

    df = pd.DataFrame(predictions)
    st.write(f"Total predictions: **{len(df)}**")
    st.plotly_chart(chart_prediction_history(predictions), use_container_width=True)

    display_cols = ["roll_number", "predicted_mark", "grade",
                    "performance", "model_name", "prediction_date"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True)

    csv_bytes = df[available].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Predictions CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )


def page_student_ranking():
    st.title("🏆 Student Ranking")
    students = get_all_students()
    if not students:
        st.info("No students available.")
        return

    ranked = []
    for s in students:
        latest = get_latest_prediction(s["roll_number"])
        score = latest["predicted_mark"] if latest else s["final_mark"]
        grade = get_grade(score)
        ranked.append({
            "Roll Number": s["roll_number"],
            "Name":        s["name"],
            "Department":  s["department"],
            "Predicted Mark": round(score, 2),
            "Grade":       grade,
        })

    ranked.sort(key=lambda x: x["Predicted Mark"], reverse=True)
    df = pd.DataFrame(ranked)
    df.insert(0, "Rank", range(1, len(df) + 1))
    st.dataframe(df, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# ADMIN PAGES
# ═════════════════════════════════════════════════════════════════════════════

def page_admin_dashboard():
    st.title("🛡️ Admin Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students",    get_students_count())
    col2.metric("Total Faculty",     get_faculty_count())
    col3.metric("Total Predictions", get_predictions_count())

    students = get_all_students()
    stats    = compute_summary_stats(students)
    col4.metric("Avg Mark", f"{stats['avg_mark']:.1f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Attendance", f"{stats['avg_attendance']:.1f}%")
    c2.metric("Pass %",         f"{stats['pass_pct']:.1f}%")
    c3.metric("Fail %",         f"{stats['fail_pct']:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_grade_distribution(students), use_container_width=True)
    with col2:
        st.plotly_chart(chart_department_performance(students), use_container_width=True)


def page_faculty_management():
    st.title("👨‍💼 Faculty Management")

    tab_list, tab_add, tab_reset, tab_delete = st.tabs([
        "📋 Faculty List", "➕ Add Faculty",
        "🔑 Reset Password", "🗑️ Delete Faculty",
    ])

    with tab_list:
        faculty_list = get_all_faculty()
        if not faculty_list:
            st.info("No faculty records found.")
        else:
            df = pd.DataFrame(faculty_list)[["faculty_id", "name", "department", "created_at"]]
            df.columns = ["Faculty ID", "Name", "Department", "Created At"]
            st.dataframe(df, use_container_width=True)

    with tab_add:
        with st.form("add_faculty_form"):
            fac_id   = st.text_input("Faculty ID *", placeholder="e.g. FAC002")
            fac_name = st.text_input("Full Name *")
            fac_dept = st.selectbox("Department *", [
                "Computer Science", "Electronics", "Mechanical", "Civil",
                "Information Technology", "Electrical", "Chemical", "Biotechnology",
            ])
            fac_pass = st.text_input("Password *", type="password")
            fac_pass2 = st.text_input("Confirm Password *", type="password")
            submitted = st.form_submit_button("➕ Add Faculty", type="primary")

        if submitted:
            if not fac_id.strip() or not fac_name.strip() or not fac_pass:
                st.error("All fields are required.")
            elif fac_pass != fac_pass2:
                st.error("Passwords do not match.")
            elif len(fac_pass) < 6:
                st.error("Password must be at least 6 characters.")
            elif faculty_exists(fac_id.strip().upper()):
                st.error(f"Faculty ID **{fac_id.strip().upper()}** already exists.")
            elif user_exists(fac_id.strip().upper()):
                st.error(f"A user account for **{fac_id.strip().upper()}** already exists.")
            else:
                fac_id_clean = fac_id.strip().upper()
                ok = add_faculty(fac_id_clean, fac_name.strip(), fac_dept, fac_pass)
                if ok:
                    create_user(fac_id_clean, fac_pass, "faculty")
                    st.success(f"Faculty **{fac_id_clean}** added successfully!")
                else:
                    st.error("Failed to add faculty.")

    with tab_reset:
        faculty_list = get_all_faculty()
        if not faculty_list:
            st.info("No faculty to reset.")
        else:
            fac_options = [f["faculty_id"] for f in faculty_list]
            sel_fac = st.selectbox("Select Faculty ID", fac_options, key="reset_fac_sel")
            new_pass  = st.text_input("New Password", type="password", key="reset_new_pass")
            new_pass2 = st.text_input("Confirm Password", type="password", key="reset_confirm_pass")
            if st.button("🔑 Reset Password", type="primary"):
                if not new_pass:
                    st.error("Password cannot be empty.")
                elif new_pass != new_pass2:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    if update_user_password(sel_fac, new_pass):
                        st.success(f"Password for **{sel_fac}** reset successfully.")
                    else:
                        st.error("Password reset failed.")

    with tab_delete:
        faculty_list = get_all_faculty()
        if not faculty_list:
            st.info("No faculty to delete.")
        else:
            fac_options = [f["faculty_id"] for f in faculty_list]
            sel_del = st.selectbox("Select Faculty to Delete", fac_options, key="del_fac_sel")
            confirm_del = st.checkbox(f"I confirm deletion of {sel_del}", key="del_fac_confirm")
            if st.button("🗑️ Delete Faculty", type="primary", disabled=not confirm_del):
                delete_faculty(sel_del)
                delete_user(sel_del)
                st.success(f"Faculty **{sel_del}** deleted.")
                st.rerun()


def page_import_data():
    st.title("📥 Import Student Data (CSV)")

    st.markdown("**Required columns:**")
    st.code(", ".join(REQUIRED_CSV_COLUMNS))

    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="import_csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write(f"Rows detected: **{len(df)}**")
            st.dataframe(df.head(5), use_container_width=True)

            ok, result = validate_csv_dataframe(df)
            if not ok:
                st.error(f"Validation error: {result}")
                return

            if st.button("⬆️ Import Students", type="primary"):
                records = result.to_dict(orient="records")
                success, skipped = bulk_add_students(records)
                st.success(f"Import complete: **{success}** added, **{skipped}** skipped (duplicates).")

        except Exception as e:
            st.error(f"Failed to read CSV: {e}")


def page_export_data():
    st.title("📤 Export Data")

    students    = get_all_students()
    predictions = get_all_predictions()

    st.subheader("1️⃣ Student Data")
    if students:
        df_stu = pd.DataFrame(students)
        st.write(f"{len(df_stu)} student records")
        st.download_button(
            "📥 Download Student Data CSV",
            data=df_stu.to_csv(index=False).encode("utf-8"),
            file_name="student_data.csv",
            mime="text/csv",
        )
    else:
        st.info("No student data to export.")

    st.markdown("---")
    st.subheader("2️⃣ Prediction History")
    if predictions:
        df_pred = pd.DataFrame(predictions)
        st.write(f"{len(df_pred)} prediction records")
        st.download_button(
            "📥 Download Predictions CSV",
            data=df_pred.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("No prediction data to export.")

    st.markdown("---")
    st.subheader("3️⃣ At-Risk Student Report")
    if students:
        at_risk_rows = []
        for s in students:
            latest    = get_latest_prediction(s["roll_number"])
            pred_mark = latest["predicted_mark"] if latest else None
            reasons   = get_risk_reasons(s, pred_mark)
            if reasons:
                at_risk_rows.append({
                    "Roll Number":    s["roll_number"],
                    "Name":           s["name"],
                    "Department":     s["department"],
                    "Attendance":     s["attendance"],
                    "Internal Mark":  s["internal_mark"],
                    "Predicted Mark": pred_mark if pred_mark is not None else "N/A",
                    "Risk Reasons":   ", ".join(reasons),
                })
        if at_risk_rows:
            df_risk = pd.DataFrame(at_risk_rows)
            st.write(f"{len(df_risk)} at-risk students")
            st.download_button(
                "📥 Download At-Risk Report CSV",
                data=df_risk.to_csv(index=False).encode("utf-8"),
                file_name="at_risk_report.csv",
                mime="text/csv",
            )
        else:
            st.success("No at-risk students to export.")
    else:
        st.info("No student data available.")


# ═════════════════════════════════════════════════════════════════════════════
# ROUTING
# ═════════════════════════════════════════════════════════════════════════════

def route_student():
    render_student_sidebar()
    page = st.session_state.get("page", "dashboard")
    if page == "dashboard":
        page_student_dashboard()
    elif page == "my_performance":
        page_my_performance()
    elif page == "predict_mark":
        page_predict_mark()
    elif page == "prediction_history":
        page_student_prediction_history()
    else:
        page_student_dashboard()


def route_faculty():
    render_faculty_sidebar()
    page = st.session_state.get("page", "dashboard")
    if page == "dashboard":
        page_faculty_dashboard()
    elif page == "students":
        page_students_list()
    elif page == "add_student":
        page_add_student()
    elif page == "analytics":
        page_faculty_analytics()
    elif page == "at_risk":
        page_at_risk()
    elif page == "model_training":
        page_model_training()
    elif page == "predictions":
        page_predictions_view("faculty")
    else:
        page_faculty_dashboard()


def route_admin():
    render_admin_sidebar()
    page = st.session_state.get("page", "dashboard")
    if page == "dashboard":
        page_admin_dashboard()
    elif page == "students":
        page_students_list()
    elif page == "faculty_management":
        page_faculty_management()
    elif page == "analytics":
        page_faculty_analytics()
    elif page == "at_risk":
        page_at_risk()
    elif page == "model_training":
        page_model_training()
    elif page == "predictions":
        page_predictions_view("admin")
    elif page == "import_data":
        page_import_data()
    elif page == "export_data":
        page_export_data()
    else:
        page_admin_dashboard()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    initialize_system()
    init_session()

    if not is_logged_in():
        render_login_page()
        return

    role = get_role()
    if role == "student":
        route_student()
    elif role == "faculty":
        route_faculty()
    elif role == "admin":
        route_admin()
    else:
        st.error("Unknown role. Please log out and try again.")
        if st.button("Logout"):
            logout()


if __name__ == "__main__":
    main()

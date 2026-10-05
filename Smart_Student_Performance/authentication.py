"""
authentication.py - Role-based authentication using Streamlit session_state.
"""

import streamlit as st
from database import (
    get_user,
    create_user,
    verify_password,
    user_exists,
    add_faculty,
    faculty_exists,
)


# ── Default Account Setup ─────────────────────────────────────────────────────

def create_default_accounts():
    """Create default student, faculty, and admin accounts if not present."""
    defaults = [
        ("ST001",    "Student@123", "student"),
        ("FAC001",   "Faculty@123", "faculty"),
        ("ADMIN001", "Admin@123",   "admin"),
    ]
    for username, password, role in defaults:
        if not user_exists(username):
            create_user(username, password, role)

    # Default faculty record
    if not faculty_exists("FAC001"):
        add_faculty("FAC001", "Dr. Default Faculty", "Computer Science", "Faculty@123")


# ── Session State Helpers ─────────────────────────────────────────────────────

def init_session():
    """Initialise session_state keys if not already present."""
    defaults = {
        "logged_in":  False,
        "username":   None,
        "role":       None,
        "user_data":  None,
        "page":       "dashboard",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def login(username: str, password: str) -> tuple:
    """
    Attempt login.
    Returns (True, role, user_dict) on success,
            (False, None, None) on failure.
    """
    if not username or not password:
        return False, None, None

    user = get_user(username.strip())
    if user is None:
        return False, None, None

    if not verify_password(password, user["password_hash"]):
        return False, None, None

    return True, user["role"], user


def set_session(user: dict):
    """Populate session_state after successful login."""
    st.session_state["logged_in"] = True
    st.session_state["username"]  = user["username"]
    st.session_state["role"]      = user["role"]
    st.session_state["user_data"] = user
    st.session_state["page"]      = "dashboard"


def logout():
    """Clear all session_state keys and rerun."""
    for key in ["logged_in", "username", "role", "user_data", "page"]:
        st.session_state[key] = None
    st.session_state["logged_in"] = False
    st.rerun()


def is_logged_in() -> bool:
    return st.session_state.get("logged_in", False)


def get_role() -> str:
    return st.session_state.get("role", "")


def get_username() -> str:
    return st.session_state.get("username", "")


def require_role(*roles):
    """Stop execution with an error if current role is not in roles."""
    if get_role() not in roles:
        st.error("Access denied. You do not have permission to view this page.")
        st.stop()


# ── Login UI ──────────────────────────────────────────────────────────────────

def render_login_page():
    """Render the login form. Returns True if login succeeded."""
    st.markdown(
        """
        <div style='text-align:center; padding: 2rem 0 1rem 0;'>
            <h1 style='color:#1f77b4; font-size:2.2rem;'>
                🎓 Smart Student Performance &amp; Mark Prediction System
            </h1>
            <p style='color:#555; font-size:1.1rem;'>
                AI-Powered Academic Performance Analytics
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("---")
        st.subheader("🔐 Login")

        role_choice = st.selectbox(
            "Login as",
            ["Student", "Faculty", "Admin"],
            key="login_role_choice",
        )

        if role_choice == "Student":
            id_label = "Roll Number"
            placeholder = "e.g. ST001"
        elif role_choice == "Faculty":
            id_label = "Faculty ID"
            placeholder = "e.g. FAC001"
        else:
            id_label = "Admin ID"
            placeholder = "e.g. ADMIN001"

        username = st.text_input(id_label, placeholder=placeholder, key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", type="primary", use_container_width=True):
            if not username or not password:
                st.error("Please enter both ID and password.")
                return False

            ok, role, user = login(username.strip(), password)

            if not ok:
                st.error("Invalid credentials. Please try again.")
                return False

            # Role mismatch check
            expected = role_choice.lower()
            if role != expected:
                st.error(f"This account is not a {role_choice} account.")
                return False

            set_session(user)
            st.success(f"Welcome! Logged in as {role_choice}.")
            st.rerun()
            return True

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
            "Default credentials — Student: ST001 / Student@123 | "
            "Faculty: FAC001 / Faculty@123 | Admin: ADMIN001 / Admin@123"
            "</p>",
            unsafe_allow_html=True,
        )
    return False

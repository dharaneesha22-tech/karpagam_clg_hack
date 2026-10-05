# Smart Student Performance & Mark Prediction System

AI-Powered Academic Performance Analytics built with Streamlit.

## Features

- Role-based login: Student, Faculty, Admin
- AI mark prediction using ML (Linear Regression, Decision Tree, Random Forest)
- Student performance analytics with Plotly charts
- At-risk student detection
- Faculty CRUD for student records
- Admin: faculty management, CSV import/export, model training
- SQLite database with automatic initialization
- Prediction history tracking

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Default Credentials

| Role    | Username  | Password     |
|---------|-----------|--------------|
| Student | ST001     | Student@123  |
| Faculty | FAC001    | Faculty@123  |
| Admin   | ADMIN001  | Admin@123    |

## Project Structure

```
Smart_Student_Performance/
├── streamlit_app.py      # Main entry point
├── database.py           # SQLite CRUD
├── authentication.py     # Login / session
├── ml_model.py           # ML pipeline
├── analytics.py          # Plotly charts
├── utils.py              # Helpers & validators
├── requirements.txt
├── data/
│   └── student_marks.csv
├── models/
│   └── best_model.pkl    (auto-generated)
└── database/
    └── students.db       (auto-generated)
```

## Streamlit Cloud Deployment

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file path** to `streamlit_app.py`.
4. Click **Deploy**.

No environment variables are required.

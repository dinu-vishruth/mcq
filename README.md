# AI-Powered MCQ Generator

Welcome to the **AI-Powered MCQ Generator**! This is a robust Flask-based web application designed to help teachers effortlessly generate Multiple Choice Questions (MCQs) from various documents and seamlessly manage student assessments. 

By leveraging the power of Google's Gemini AI, this application automatically reads uploaded educational materials, generates tailored question sets based on selected difficulty, and provides an end-to-end testing platform for students.

## Features ✨

### For Teachers 👩‍🏫
- **Account Management**: Signup, login, and secure session handling.
- **Document Processing**: Upload materials in `.pdf`, `.docx`, or `.pptx` formats. The app intelligently extracts text from these documents.
- **AI Question Generation**: Set the desired number of questions, difficulty level (easy, medium, hard), and time limit. The AI generates high-quality MCQs based on the uploaded content.
- **Session Management**: Each generated test creates a unique **Session Key**. Share this key with your students to grant them access to the test.
- **Reports & Analytics**: View detailed student performance reports for each session. 
- **PDF Export**: Download the cleanly formatted MCQ question sets as PDF files.

### For Students 👨‍🎓
- **Easy Access**: Log into any active test using your name and the provided Session Key.
- **Interactive Test Interface**: Answer randomized questions within a configurable time limit. Options are automatically shuffled to prevent cheating.
- **Instant Results & Explanations**: After submission, get immediate scoring along with detailed, AI-driven explanations for each correct and incorrect answer.
- **Dashboard History**: Keep track of all your past test results and performance directly from your dashboard.

## Tech Stack 🛠️

- **Backend framework**: Python 3.x, Flask, Flask-Session
- **Database**: SQLite3
- **AI Integration**: Gemini AI (for question generation and explanations)
- **Document Parsing**: Support for PDF, Word (docx), and PowerPoint (pptx) extraction
- **PDF Generation**: ReportLab
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla), Jinja2 Templating
- **Security**: Werkzeug password hashing, randomized option shuffling

## Getting Started 🚀

### 1. Prerequisites
Make sure you have Python 3 installed. You will also need a Gemini API Key setup in your environment or a `.env` file.

### 2. Installation
Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/dinu-vishruth/mcq.git
cd mcq/mcq-main

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
1. Rename `.env.example` to `.env` (if provided) and add your necessary environment variables locally (e.g., your Gemini API key).
2. The database schema will be automatically initialized using `database/schema.sql` the first time you run the app.

### 4. Running the Application
Start the Flask development server:

```bash
python app.py
```

Open your web browser and go to `http://127.0.0.1:5000/`.

## Application Structure 📂
- `app.py`: Main Flask application router.
- `models/`: Logic for PDF processing, AI generation, and explanation engine.
- `utils/`: Helpers for text cleaning, session management, and classification.
- `templates/`: HTML templates for login, dashboards, uploads, and tests.
- `database/`: SQLite database storage and schema files.
- `static/`: CSS and client-side JavaScript (e.g., test timer).

## Contributing 🤝
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
*Built to simplify assessments and enhance AI-driven learning!*

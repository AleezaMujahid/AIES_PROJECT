"""Start CodeSentinel locally:   python run.py     then open http://127.0.0.1:5000"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    # debug is OFF on purpose (the debugger would show source code and allow code execution).
    app.run(host="127.0.0.1", port=5000, debug=False)

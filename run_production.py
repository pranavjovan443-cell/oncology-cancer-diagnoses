import os
from waitress import serve
from app import app
from database.init_db import init_db

if __name__ == '__main__':
    with app.app_context():
        init_db(app)

    port = int(os.environ.get('PORT', 5000))
    print("=========================================================================")
    print(" HealMind AI Quantum Oncology — Production Waitress Server Active")
    print(f" Local Web Access:     http://127.0.0.1:{port}")
    print(f" Network Access:       http://0.0.0.0:{port}")
    print("=========================================================================")
    serve(app, host='0.0.0.0', port=port, threads=4)

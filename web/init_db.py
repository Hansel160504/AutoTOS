import time, sys
from app import app
from extensions import db

for i in range(30):
    try:
        with app.app_context():
            db.create_all()
        print("DB tables created!")
        break
    except Exception as e:
        print(f"Waiting for DB ({i+1}/30)... {e}")
        time.sleep(2)
else:
    print("DB never became ready!")
    sys.exit(1)
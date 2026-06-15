import os
import sys
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))
from backend import config

if __name__ == "__main__":
    print("=" * 50)
    print("  Face Recognition Attendance System")
    print("  ArcFace + FastAPI + SQLite + Vue.js")
    print("=" * 50)
    print(f"\nStarting server at http://{config.HOST}:{config.PORT}")
    print(f"Static files: {config.STATIC_DIR}")
    print(f"Database: {config.DB_PATH}")
    print(f"Feature DB: {config.FEATURES_DIR}")
    print("\nPress Ctrl+C to stop.\n")
    uvicorn.run(
        "backend.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        reload_dirs=[os.path.dirname(__file__)]
    )

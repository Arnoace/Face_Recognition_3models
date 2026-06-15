import os, sys, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend import config
from database.db import Database
from backend.routers.employee import create_employee_router
from backend.routers.recognition import create_recognition_router, FeatureDatabase
from backend.routers.attendance import create_attendance_router
from backend.routers.model import create_model_router
from models.model_manager import ModelManager
from models.arcface import ArcFaceModel
from models.model_a import FisherfacesModel
from models.model_b import FaceNetModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"DB path = {config.DB_PATH}")
logger.info(f"Data dir = {config.DATA_DIR}")
logger.info(f"Base dir = {config.BASE_DIR}")

logger.info("Initializing database...")
db = Database(config.DB_PATH)
db_exists = os.path.exists(config.DB_PATH)
db_size = os.path.getsize(config.DB_PATH) if db_exists else 0
logger.info(f"Database ready: {config.DB_PATH} (exists={db_exists}, size={db_size})")

logger.info("Initializing model manager...")
model_manager = ModelManager()

logger.info("Loading and registering models...")
try:
    m_arcface = ArcFaceModel()
    model_manager.register("ArcFace", m_arcface)
    logger.info(f"ArcFace registered: {m_arcface.name}")
except Exception as e:
    logger.error(f"ArcFace load failed: {e}")

try:
    m_modela = FisherfacesModel()
    model_manager.register("ModelA", m_modela)
    logger.info(f"Model-A (Fisherfaces) registered: {m_modela.name}")
except Exception as e:
    logger.error(f"Model-A load failed: {e}")

try:
    m_modelb = FaceNetModel()
    model_manager.register("ModelB", m_modelb)
    logger.info(f"FaceNet registered: {m_modelb.name}")
except Exception as e:
    logger.error(f"Model-B load failed: {e}")

logger.info("Building feature database...")
feature_db = FeatureDatabase(db, model_manager)
logger.info(f"Feature DB ready: {len(feature_db.metadata)} identities")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server started"); yield; logger.info("Server stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="Face Recognition Attendance", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    async def health():
        db_exists = os.path.exists(config.DB_PATH)
        db_size = os.path.getsize(config.DB_PATH) if db_exists else 0
        current = model_manager.current
        return {
            "status": "ok",
            "model": model_manager.current_name,
            "model_name": current.name if current else "not loaded",
            "available_models": model_manager.available_models,
            "features": len(feature_db.metadata) if feature_db else 0,
            "db_path": str(config.DB_PATH),
            "db_exists": db_exists,
            "db_size": db_size,
        }

    app.include_router(create_model_router(model_manager))
    app.include_router(create_employee_router(db, model_manager, feature_db))
    app.include_router(create_recognition_router(model_manager, feature_db))
    app.include_router(create_attendance_router(db, feature_db))

    static_dir = config.STATIC_DIR
    if os.path.exists(static_dir):
        @app.get("/")
        async def index():
            return FileResponse(os.path.join(static_dir, "index.html"))

        @app.get("/css/{fn}")
        async def css(fn: str):
            fp = os.path.join(static_dir, "css", fn)
            return FileResponse(fp, media_type="text/css") if os.path.exists(fp) else HTMLResponse("", 404)

        @app.get("/js/{fn}")
        async def js(fn: str):
            fp = os.path.join(static_dir, "js", fn)
            return FileResponse(fp, media_type="application/javascript") if os.path.exists(fp) else HTMLResponse("", 404)

        @app.get("/{fn}")
        async def stfile(fn: str):
            fp = os.path.join(static_dir, fn)
            return FileResponse(fp) if os.path.exists(fp) else HTMLResponse("", 404)
        logger.info(f"Static served from {static_dir}")
    else:
        logger.warning(f"Static dir not found: {static_dir}")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host=config.HOST, port=config.PORT, reload=True)

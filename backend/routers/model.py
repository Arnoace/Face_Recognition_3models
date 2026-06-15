import logging
from fastapi import APIRouter, HTTPException
logger = logging.getLogger(__name__)
def create_model_router(model_manager):
    router = APIRouter(prefix="/api/model", tags=["Model"])
    @router.get("/status")
    async def get_status():
        m = model_manager.current
        return {"code": 200, "data": {
            "current": model_manager.current_name,
            "available": model_manager.available_models,
            "feature_dim": m.feature_dim if m else 0,
            "model_name": m.name if m else "N/A"
        }}
    @router.post("/switch")
    async def switch(model_name: str):
        try:
            model_manager.switch(model_name)
            return {"code": 200, "message": "已切换",
                    "data": {"current": model_manager.current_name}}
        except ValueError as e:
            raise HTTPException(400, str(e))
    return router

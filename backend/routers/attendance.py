import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)


def create_attendance_router(db, feature_db):
    router = APIRouter(prefix="/api/attendance", tags=["Attendance"])

    @router.get("")
    async def get_today():
        try:
            return {"code": 200, "data": db.get_today_attendance()}
        except Exception as e:
            logger.error(f"Get today error: {e}"); raise HTTPException(500, str(e))

    @router.get("/all")
    async def get_all(limit: int = 500, start: str = None, end: str = None):
        try:
            records = db.get_attendance(limit=limit, start=start, end=end)
            return {"code": 200, "data": records, "total": len(records)}
        except Exception as e:
            logger.error(f"Get all error: {e}"); raise HTTPException(500, str(e))

    @router.post("/checkin")
    async def check_in(employee_id: str, name: str):
        try:
            if db.has_checked_in_today(employee_id):
                return {"code": 200, "message": "今日已签到", "data": {"duplicate": True}}

            # 判断是否迟到: 9:00 AM 之后签到视为迟到
            now = datetime.now()
            h = now.hour
            if h < 9:
                status = "正常"
            elif h < 12:
                status = "迟到"
            else:
                status = "早退"

            record = db.add_attendance(employee_id, name, status)
            return {"code": 200, "message": "签到成功", "data": record}
        except Exception as e:
            logger.error(f"Check-in error: {e}")
            raise HTTPException(500, str(e))

    return router

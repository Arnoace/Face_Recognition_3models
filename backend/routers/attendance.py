import logging
import cv2
import numpy as np
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

logger = logging.getLogger(__name__)

# ── Haar Cascade 人脸检测器 (全局单例) ──
_face_detector = None

def _get_detector():
    global _face_detector
    if _face_detector is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_detector = cv2.CascadeClassifier(cascade_path)
    return _face_detector


def _crop_face(image_bgr: np.ndarray):
    """检测人脸并裁剪放大的人脸区域 (112x112 RGB).

    Args:
        image_bgr: BGR 图像

    Returns:
        bytes: JPEG 编码的人脸图片，无人脸时返回 None
    """
    detector = _get_detector()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    # 取最大的人脸
    best = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = best

    # 扩大 margin
    margin_w = int(w * 0.3)
    margin_h = int(h * 0.3)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(image_bgr.shape[1], x + w + margin_w)
    y2 = min(image_bgr.shape[0], y + h + margin_h)

    face_crop = image_bgr[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))  # 放大显示

    # 编码为 JPEG
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


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
    async def check_in(
        employee_id: str = Form(...),
        name: str = Form(...),
        file: UploadFile = File(None),
    ):
        try:
            if db.has_checked_in_today(employee_id):
                return {"code": 200, "message": "今日已签到", "data": {"duplicate": True}}

            # 判断是否迟到
            now = datetime.now()
            h = now.hour
            if h < 9:
                status = "正常"
            elif h < 12:
                status = "迟到"
            else:
                status = "早退"

            # 处理打卡照片：检测并裁剪人脸
            face_photo = None
            if file is not None:
                contents = await file.read()
                if contents:
                    arr = np.frombuffer(contents, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        face_photo = _crop_face(img)
                        if face_photo is None:
                            # 无人脸时回退：保存原始图片
                            face_photo = contents

            record = db.add_attendance(employee_id, name, status, photo_bytes=face_photo)
            return {"code": 200, "message": "签到成功", "data": record}
        except Exception as e:
            logger.error(f"Check-in error: {e}")
            raise HTTPException(500, str(e))

    @router.get("/{pk:int}/photo")
    async def get_attendance_photo(pk: int):
        """获取打卡时的人脸照片."""
        try:
            result = db.get_attendance_photo(pk)
            if not result:
                raise HTTPException(404, "该记录无打卡照片")
            return {"code": 200, "data": result}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get photo error: {e}")
            raise HTTPException(500, str(e))

    return router

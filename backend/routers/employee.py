import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
logger = logging.getLogger(__name__)
def create_employee_router(db, model_manager, feature_db):
    router = APIRouter(prefix="/api/employees", tags=["Employees"])
    @router.get("")
    async def list_employees():
        try:
            employees = db.get_all_employees()
            current_model = model_manager.current_name
            stats = {'total': len(employees), 'attendance_count': db.get_attendance_count(),
                     'current_model': current_model,
                     'db_features': len(feature_db.db_metadata),
                     'yale_features': len(feature_db.yale_metadata)}
            return {"code": 200, "data": employees, "stats": stats}
        except Exception as e:
            logger.error(f"List error: {e}")
            raise HTTPException(500, str(e))
    @router.post("")
    async def register_employee(name: str = Form(...), employee_id: str = Form(...),
                                department: str = Form(""), file: UploadFile = File(...)):
        try:
            contents = await file.read()
            import cv2, numpy as np
            arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            # Extract features for ALL registered models
            all_features = {}
            for mn in model_manager.available_models:
                model = model_manager._models.get(mn)
                if model:
                    feat = model.extract_feature(img)
                    all_features[mn] = feat
            # Store employee with photo + all features
            employee = db.add_employee(name, employee_id, department, all_features, photo_bytes=contents)
            employee.pop('face_feature', None)
            employee.pop('photo', None)
            # Add features to ALL model stores in FeatureDatabase
            for mn, feat in all_features.items():
                feature_db.add_employee_feature(employee['id'], name, employee_id, department, feat, model_name=mn)
            return {"code": 200, "message": "注册成功", "data": employee}
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            logger.error(f"Register error: {e}")
            raise HTTPException(500, str(e))
    @router.delete("/{pk:int}")
    async def delete_employee(pk: int):
        try:
            emp = db.get_employee_by_pk(pk)
            if not emp:
                raise HTTPException(404, "员工不存在")
            db.delete_employee(pk)
            feature_db.remove_employee_feature(pk)
            return {"code": 200, "message": f"已删除 {emp['name']}"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Delete error: {e}")
            raise HTTPException(500, str(e))

    @router.get("/{pk:int}")
    async def get_employee_detail(pk: int):
        try:
            emp = db.get_employee_by_pk(pk)
            if not emp:
                raise HTTPException(404, "员工不存在")
            return {"code": 200, "data": emp}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Detail error: {e}")
            raise HTTPException(500, str(e))

    return router

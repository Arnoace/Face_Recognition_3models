import os, sqlite3
import pickle
import base64
import numpy as np
from .models import SCHEMA
from backend import config


class Database:
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    @staticmethod
    def _pack_features(features_dict):
        """Pack a {model_name: np.array} dict into BLOB."""
        return pickle.dumps(features_dict)

    @staticmethod
    def _unpack_features(blob):
        """Unpack BLOB into {model_name: np.array} dict. Handles old single-array format."""
        if blob is None:
            return {}
        try:
            data = pickle.loads(blob)
            if isinstance(data, dict):
                return data
            return {"ArcFace": np.frombuffer(data, dtype=np.float32) if isinstance(data, bytes) else data}
        except Exception:
            return {"ArcFace": np.frombuffer(blob, dtype=np.float32)}

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init_db(self):
        conn = self._connect()
        for stmt in SCHEMA:
            try:
                conn.execute(stmt)
            except Exception:
                pass  # ALTER TABLE might fail if column already exists
        conn.commit()
        conn.close()

    def add_employee(self, name, employee_id, department, features_dict, photo_bytes=None):
        """features_dict = {model_name: np.array feature_vector}"""
        conn = self._connect()
        try:
            fb = self._pack_features(features_dict)
            conn.execute("INSERT INTO employees (name,employee_id,department,face_feature,photo) VALUES (?,?,?,?,?)",
                         (name, employee_id, department, fb, photo_bytes))
            conn.commit()
            row = conn.execute("SELECT * FROM employees WHERE employee_id=?", (employee_id,)).fetchone()
            return dict(row)
        except sqlite3.IntegrityError:
            raise ValueError(f"工号 '{employee_id}' 已存在")
        finally:
            conn.close()

    def update_employee(self, pk, name, employee_id, department):
        conn = self._connect()
        try:
            conn.execute("UPDATE employees SET name=?, employee_id=?, department=? WHERE id=?",
                         (name, employee_id, department, pk))
            conn.commit()
            row = conn.execute("SELECT id,name,employee_id,department,created_at FROM employees WHERE id=?", (pk,)).fetchone()
            conn.close()
            return dict(row) if row else None
        except sqlite3.IntegrityError:
            conn.close()
            raise ValueError(f"工号 '{employee_id}' 已被占用")

    def get_all_employees(self):
        conn = self._connect()
        rows = conn.execute("SELECT id,name,employee_id,department,created_at FROM employees ORDER BY created_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_employee_by_pk(self, pk):
        conn = self._connect()
        row = conn.execute("SELECT * FROM employees WHERE id=?", (pk,)).fetchone()
        conn.close()
        if not row:
            return None
        d = dict(row)
        if d.get('photo'):
            d['photo_base64'] = base64.b64encode(d['photo']).decode('ascii')
            d['photo_mime'] = 'image/jpeg'
        del d['photo']
        feats = self._unpack_features(d.get('face_feature'))
        d['available_models'] = list(feats.keys())
        d['features'] = {}
        for mn, feat in feats.items():
            nrm = float(np.linalg.norm(feat))
            dim = feat.shape[0]
            preview = feat.tolist()
            label = {'ArcFace': 'ArcFace模型', 'ArcFaceSelfTrained': 'ArcFace(Self-Trained)', 'ModelA': 'Fisherfaces', 'ModelB': 'FaceNet'}.get(mn, mn)
            dim_label = f'{dim}维'
            d['features'][mn] = {'dim': dim, 'dim_label': dim_label, 'norm': round(nrm, 4), 'preview': preview, 'label': label}
        del d['face_feature']
        return d

    def delete_employee(self, pk):
        conn = self._connect()
        c = conn.execute("DELETE FROM employees WHERE id=?", (pk,))
        ok = c.rowcount > 0
        conn.commit()
        conn.close()
        return ok

    def get_all_features(self, model_name=None):
        """Return (features_array, metadata) for a specific model, or all models combined."""
        conn = self._connect()
        rows = conn.execute("SELECT id,name,employee_id,department,face_feature FROM employees WHERE face_feature IS NOT NULL").fetchall()
        conn.close()
        features, meta = [], []
        for r in rows:
            feats = self._unpack_features(r['face_feature'])
            if model_name and model_name in feats:
                feat = feats[model_name]
                features.append(feat)
                meta.append({'id': r['id'], 'name': r['name'], 'employee_id': r['employee_id'], 'department': r['department']})
            elif model_name is None:
                for mn, feat in feats.items():
                    features.append(feat)
                    meta.append({'id': r['id'], 'name': r['name'], 'employee_id': r['employee_id'], 'department': r['department'], 'model': mn})
        dim = features[0].shape[0] if features else 512
        return np.array(features) if features else np.empty((0, dim)), meta

    def add_attendance(self, employee_id, name, status='正常', photo_bytes=None):
        conn = self._connect()
        conn.execute("INSERT INTO attendance (employee_id,name,status,photo) VALUES (?,?,?,?)", (employee_id, name, status, photo_bytes))
        conn.commit()
        row = conn.execute("SELECT * FROM attendance WHERE id=last_insert_rowid()").fetchone()
        conn.close()
        return dict(row)

    def get_attendance(self, limit=500, start=None, end=None):
        conn = self._connect()
        sql = "SELECT id,employee_id,name,check_in,status,photo IS NOT NULL AS has_photo FROM attendance"
        params, conds = [], []
        if start:
            conds.append("date(check_in) >= ?")
            params.append(start)
        if end:
            conds.append("date(check_in) <= ?")
            params.append(end)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY check_in DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_today_attendance(self):
        conn = self._connect()
        rows = conn.execute("SELECT id,employee_id,name,check_in,status,photo IS NOT NULL AS has_photo FROM attendance WHERE date(check_in)=date('now') ORDER BY check_in DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def has_checked_in_today(self, employee_id):
        conn = self._connect()
        r = conn.execute("SELECT id FROM attendance WHERE employee_id=? AND date(check_in)=date('now') LIMIT 1", (employee_id,)).fetchone()
        conn.close()
        return r is not None

    def get_attendance_count(self):
        conn = self._connect()
        r = conn.execute("SELECT COUNT(*) as c FROM attendance").fetchone()
        conn.close()
        return r['c']

    def get_attendance_photo(self, pk):
        """Get the check-in face photo for an attendance record."""
        conn = self._connect()
        row = conn.execute("SELECT photo FROM attendance WHERE id=?", (pk,)).fetchone()
        conn.close()
        if not row or not row['photo']:
            return None
        import base64
        return {
            'photo_base64': base64.b64encode(row['photo']).decode('ascii'),
            'photo_mime': 'image/jpeg',
        }

    def set_attendance_photo(self, pk, photo_bytes):
        """Update the photo for an existing attendance record."""
        conn = self._connect()
        conn.execute("UPDATE attendance SET photo=? WHERE id=?", (photo_bytes, pk))
        conn.commit()
        conn.close()

    def get_employee_count(self):
        conn = self._connect()
        r = conn.execute("SELECT COUNT(*) as c FROM employees").fetchone()
        conn.close()
        return r['c']

from app import db
db.create_all()
from app import Camera #,Traffic
camera1_to_add=Camera(id=1, location='Live 1', road_length=0.19611742)
camera2_to_add=Camera(id=2, location='Live 2', road_length=0.19611742)
camera3_to_add=Camera(id=3, location='Live 3', road_length=0.089)

db.session.add(camera1_to_add)
db.session.add(camera2_to_add)
db.session.add(camera3_to_add)

db.session.commit()
Camera.query.all()
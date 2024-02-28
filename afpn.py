from pathlib import Path
from site_yolo.models import YOLO


afpn_yaml = Path.cwd() / "configs/models/yolov8-afpn.yaml"
detector = YOLO(model=str(afpn_yaml))
if __name__=="__main__":
    res = detector.predict(source='data/predict/', # device="cpu", 
                           project='runs/exp', save=True)

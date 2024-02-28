from rtdet.models.rtdet.model import RTDetector
import pdb, sys

# detector = RTDetector("configs/models/rtdet.yaml")
# detector.load(weights='runs/exp/train3/weights/best.pt')
detector = RTDetector("runs/exp/train3/weights/best.pt")

if __name__=="__main__":
    # pdb.set_trace()
    res = detector.predict(source='data/predict/', project='runs/exp', save=True)
    sys.exit(0)
    res = detector.train(
        data = "configs/datasets/VOC.yaml",
        split = "train",
        batch = 16,
        save_period = 1,
        epochs = 160,
        imgsz=640,
        project="runs/exp",
        name="train",
        device = [0],
    )

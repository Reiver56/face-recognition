import cv2, argparse, json
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "aligned"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def load_arcface(model_path: Path):
    net = cv2.dnn.readNet(str(model_path))
    return net

def arcface_embed(net, face_bgr):
    # Preprocess ArcFace: 112x112, RGB, mean/std 127.5
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (112, 112))
    blob = cv2.dnn.blobFromImage(
        rgb, 1.0/127.5, (112,112), (127.5,127.5,127.5),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    feat = net.forward().reshape(-1).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-12)  # L2 normalize
    return feat

def load_split_feats(net, split="train"):
    root = ALIGNED / split
    classes = [d.name for d in sorted(root.iterdir()) if d.is_dir()]
    X, y, labels = [], [], []
    for lid, cname in enumerate(classes):
        labels.append(cname)
        for imgp in sorted((root/cname).glob("*.*")):
            img = cv2.imread(str(imgp))
            if img is None:
                continue
            X.append(arcface_embed(net, img))
            y.append(lid)
    if not X:
        raise RuntimeError(f"Nessuna immagine trovata in {root}. Hai eseguito 01_detect_align_landmarks.py?")
    X = normalize(np.vstack(X))
    y = np.array(y, dtype=np.int32)
    return X, y, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arcface", type=str, default=str(MODEL_DIR/"arcface_r100.onnx"))
    ap.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf"])
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    net = load_arcface(Path(args.arcface))
    Xtr, ytr, classes = load_split_feats(net, "train")

    clf = SVC(kernel=args.kernel, C=args.C, probability=True)
    clf.fit(Xtr, ytr)

    dump(clf, MODEL_DIR/"svm.pkl")
    with open(MODEL_DIR/"classes.json", "w") as f:
        json.dump(classes, f)

    print(f"Addestrate {len(classes)} classi su {len(Xtr)} campioni.")
    print("Salvati: models/svm.pkl e models/classes.json")

if __name__ == "__main__":
    main()
import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Query
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from src.predictor import GunDetector, Detection, annotate_detection, annotate_segmentation
from src.config import get_settings
from src.models import Gun, Person, PixelLocation

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)
detector = GunDetector()

@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def detect_uploadfile(detector: GunDetector, file, threshold) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return detector.detect_guns(img_array, threshold), img_array


@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold)

    return results


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

def read_image(file: UploadFile) -> np.ndarray:
    img_bytes = file.file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

@app.post("/annotate_people")
async def annotate_people(file: UploadFile = File(...), draw_boxes: bool = Query(True)):
    img = read_image(file)
    seg = detector.segment_people(img)
    annotated = annotate_segmentation(img, seg, draw_boxes)
    _, buf = cv2.imencode('.jpg', annotated)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

@app.post("/detect_people")
async def detect_people(file: UploadFile = File(...)):
    img = read_image(file)
    seg = detector.segment_people(img)
    return seg

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img = read_image(file)
    det = detector.detect_guns(img)
    seg = detector.segment_people(img)
    return {"detection": det, "segmentation": seg}

@app.post("/annotate")
async def annotate(file: UploadFile = File(...), draw_boxes: bool = Query(True)):
    img = read_image(file)
    det = detector.detect_guns(img)
    seg = detector.segment_people(img)
    img_det = annotate_detection(img, det)
    img_seg = annotate_segmentation(img_det, seg, draw_boxes)
    _, buf = cv2.imencode('.jpg', img_seg)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

@app.post("/guns")
async def guns(file: UploadFile = File(...)):
    img = read_image(file)
    det = detector.detect_guns(img)
    guns = []
    for label, box in zip(det.labels, det.boxes):
        x1, y1, x2, y2 = box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        guns.append(Gun(gun_type=label.lower(), location=PixelLocation(x=cx, y=cy)))
    return guns

@app.post("/people")
async def people(file: UploadFile = File(...)):
    img = read_image(file)
    seg = detector.segment_people(img)
    persons = []
    for poly, label in zip(seg.polygons, seg.labels):
        poly_np = np.array(poly, np.int32)
        M = cv2.moments(poly_np)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = poly_np[0]
        area = cv2.contourArea(poly_np)
        persons.append(Person(
            person_type=label,
            location=PixelLocation(x=cx, y=cy),
            area=int(area)
        ))
    return persons

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")

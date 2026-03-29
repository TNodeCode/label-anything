import os
import glob
import json
from typing import List

import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import supervision as sv
from detection import Detection


@st.cache_resource
def load_grounded_dino(model_id: str = "IDEA-Research/grounding-dino-tiny"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model, device


def match_label(detected_label: str, user_labels: List[str]) -> int:
    dl = detected_label.strip().lower()
    # try exact match first
    for idx, ul in enumerate(user_labels):
        if dl == ul.strip().lower():
            return idx
    # try substring matches
    for idx, ul in enumerate(user_labels):
        uln = ul.strip().lower()
        if uln in dl or dl in uln:
            return idx
    # fallback: return 0
    return 0


def to_coco(images_meta, annotations, categories, out_path: str):
    coco = {
        "images": images_meta,
        "annotations": annotations,
        "categories": categories,
    }
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    if boxes.shape[0] == 0:
        return np.array([], dtype=int)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


st.markdown("# Grounded DINO — Text to Bounding Boxes")
st.markdown(
    "Type comma-separated text labels to detect in images and export COCO annotations."
)

labels_input = st.text_input("Labels (comma separated)", "a cat, a remote control")
image_dir = st.text_input("Path to image directory")
model_id = st.text_input("Grounded DINO model id", "IDEA-Research/grounding-dino-tiny")
box_threshold = st.slider("Box confidence threshold", 0.0, 1.0, 0.35)
text_threshold = st.slider("Text threshold", 0.0, 1.0, 0.25)
nms_iou = st.slider("NMS IoU threshold", 0.0, 1.0, 0.5)

run_button = st.button("Run Grounded DINO")

col1, col2 = st.columns(2)
with col1:
    input_image_view = st.empty()
with col2:
    output_image_view = st.empty()

if run_button:
    user_labels = [l.strip() for l in labels_input.split(",") if l.strip()]
    if not user_labels:
        st.error("Provide at least one label.")
    elif not image_dir or not os.path.isdir(os.path.expanduser(image_dir)):
        st.error("Provide a valid image directory path.")
    else:
        processor, model, device = load_grounded_dino(model_id)

        images_glob = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images_glob.extend(
                glob.glob(os.path.join(os.path.expanduser(image_dir), ext))
            )
        images_glob = sorted(images_glob)
        n_images = len(images_glob)
        if n_images == 0:
            st.warning("No images found in the directory.")
        else:
            out_dir = os.path.expanduser(f"{image_dir}-gdino")
            os.makedirs(out_dir, exist_ok=True)

            # prepare COCO containers
            images_meta = []
            annotations = []
            categories = [
                {"id": idx, "name": name} for idx, name in enumerate(user_labels)
            ]

            progress = st.progress(0)
            image_counter = 1
            ann_counter = 1

            box_annotator = sv.BoxAnnotator()

            for i, image_path in enumerate(images_glob):
                progress.progress(int((i + 1) / n_images * 100) / 100)
                img = Image.open(image_path).convert("RGB")
                w, h = img.size
                images_meta.append(
                    {
                        "id": image_counter,
                        "file_name": os.path.basename(image_path),
                        "width": w,
                        "height": h,
                    }
                )

                inputs = processor(
                    images=img, text=[user_labels], return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[img.size[::-1]],
                )

                result = results[0]
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                labels_out = result.get("labels", [])

                # collect detections first, then apply NMS per class
                det_boxes = []
                det_scores = []
                det_cat_ids = []
                for box, score, lbl in zip(boxes, scores, labels_out):
                    x1, y1, x2, y2 = [float(x) for x in box.tolist()]
                    # map detected label to user label index
                    if isinstance(lbl, str):
                        cat_id = match_label(lbl, user_labels)
                    else:
                        try:
                            lbl_str = str(lbl)
                            cat_id = match_label(lbl_str, user_labels)
                        except Exception:
                            cat_id = 0

                    det_boxes.append([x1, y1, x2, y2])
                    det_scores.append(float(score))
                    det_cat_ids.append(int(cat_id))

                bboxes_xyxy = []
                class_ids = []

                if len(det_boxes) > 0:
                    det_boxes_np = np.array(det_boxes, dtype=float)
                    det_scores_np = np.array(det_scores, dtype=float)
                    det_cat_ids_np = np.array(det_cat_ids, dtype=int)

                    keep_indices = []
                    for cls in np.unique(det_cat_ids_np):
                        cls_inds = np.where(det_cat_ids_np == cls)[0]
                        cls_boxes = det_boxes_np[cls_inds]
                        cls_scores = det_scores_np[cls_inds]
                        kept = nms_numpy(cls_boxes, cls_scores, nms_iou)
                        keep_indices.extend(cls_inds[kept].tolist())

                    keep_indices = sorted(keep_indices)

                    for idx in keep_indices:
                        x1, y1, x2, y2 = det_boxes_np[idx].tolist()
                        w_box = x2 - x1
                        h_box = y2 - y1
                        cat_id = int(det_cat_ids_np[idx])

                        annotations.append(
                            {
                                "id": ann_counter,
                                "image_id": image_counter,
                                "category_id": int(cat_id),
                                "bbox": [x1, y1, w_box, h_box],
                                "score": float(det_scores_np[idx]),
                                "area": float(w_box * h_box),
                                "iscrowd": 0,
                            }
                        )
                        ann_counter += 1

                        bboxes_xyxy.append([x1, y1, x2, y2])
                        class_ids.append(cat_id)

                # show images with boxes using Detection helper + supervision
                if len(bboxes_xyxy) > 0:
                    det = Detection.from_bboxes(
                        np.array(bboxes_xyxy).astype(int), class_id=np.array(class_ids)
                    )
                else:
                    det = Detection.from_bboxes(np.zeros((0, 4)).astype(int))

                input_image_view.image(img)

                annotated = Detection.plot_detections(
                    img.copy(), det, show_masks=False, show_bboxes=True
                )
                output_image_view.image(annotated)

                image_counter += 1

            # save COCO json
            out_json = os.path.join(out_dir, "grounded_dino_annotations.json")
            to_coco(images_meta, annotations, categories, out_json)
            st.success(f"Finished. COCO annotations written to {out_json}")

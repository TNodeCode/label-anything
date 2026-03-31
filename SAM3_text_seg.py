import os
import json
from typing import List

import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import YOLO
import supervision as sv
from detection import Detection

import cv2


def masks_nms(masks: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    # masks: (N, H, W) binary uint8/bool, scores: (N,)
    if masks.shape[0] == 0:
        return []
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(float)
    order = scores.argsort()[::-1].tolist()
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        remove_idxs = []
        for j in order:
            inter = float(((masks[i] > 0) & (masks[j] > 0)).sum())
            union = areas[i] + areas[j] - inter
            iou = 0.0 if union == 0 else inter / union
            if iou > iou_threshold:
                remove_idxs.append(j)
        # remove indices
        order = [x for x in order if x not in remove_idxs]
    return keep


@st.cache_resource
def load_sam3(
    model_path: str = "sam3.pt",
):
    overrides = dict(
        task="segment",
        mode="predict",
        model=model_path,
        device="mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu",
        save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    return predictor


@st.cache_resource
def load_yolo_world(model_path: str = "yolov8s-world.pt"):
    try:
        model = YOLO(model_path)
    except Exception:
        model = YOLO(model_path)
    return model


def mask_to_polygons(mask: np.ndarray) -> List[List[float]]:
    # mask: 2D uint8 array (0/1)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys: List[List[float]] = []
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        cnt = cv2.approxPolyDP(cnt, 1.0, True)
        poly = cnt.reshape(-1, 2).astype(float).tolist()
        # flatten
        flat = [float(x) for p in poly for x in p]
        if len(flat) >= 6:
            polys.append(flat)
    return polys


def to_coco(images_meta, annotations, categories, out_path: str):
    coco = {
        "images": images_meta,
        "annotations": annotations,
        "categories": categories,
    }
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)


st.markdown("# SAM3 — Text to Segmentation Masks")
st.markdown(
    "Provide comma-separated text prompts, run SAM3 to produce segmentation masks, and export COCO segmentation annotations."
)

prompts_input = st.text_input("Text prompts (comma separated)", "a person, a dog")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model_path = st.text_input("SAM3 model path", "sam3.pt")
model_choice = st.selectbox("Model", ("SAM3", "YOLO-World", "YOLO-Seg"))
run_button = st.button("Run")

col1, col2 = st.columns(2)
with col1:
    input_image_view = st.empty()
with col2:
    output_image_view = st.empty()

if run_button:
    prompts = [p.strip() for p in prompts_input.split(",") if p.strip()]
    if not prompts:
        st.error("Provide at least one text prompt.")
    elif uploaded_image is None:
        st.error("Please upload an image.")
    else:
        # Save uploaded image temporarily
        img = Image.open(uploaded_image)
        image_path = "temp_uploaded_image.png"
        img.save(image_path)
        
        images_meta = []
        annotations = []
        categories = [{"id": idx, "name": name} for idx, name in enumerate(prompts)]

        progress = st.progress(0)
        image_counter = 1
        ann_counter = 1

        # Prepare model(s)
        if model_choice == "SAM3":
            predictor = load_sam3(model_path)
        elif model_choice == "YOLO-World":
            yolo = load_yolo_world("yolov8x-worldv2.pt")
            try:
                yolo.set_classes(prompts)
            except Exception:
                pass
        else:
            yolo = YOLO("yoloe-26x-seg.pt")
            try:
                yolo.set_classes(prompts)
            except Exception:
                pass

        progress.progress(100)
        print("LOADED IMAGE SHAPE", image_path, img.size)
        w, h = img.size
        images_meta.append(
            {
                "id": image_counter,
                "file_name": os.path.basename(image_path),
                "width": w,
                "height": h,
            }
        )

        # collect all masks/scores/classes for this image first
        all_xyxy: List[List[float]] = []
        all_masks: List[np.ndarray] = []
        all_scores: List[float] = []
        all_cls: List[int] = []

        if model_choice == "SAM3":
            predictor.set_image(image_path)
            for pid, prompt in enumerate(prompts):
                try:
                    results = predictor(text=[prompt])
                except Exception as e:
                    st.warning(f"Predictor failed for prompt '{prompt}': {e}")
                    continue

                if not results:
                    continue

                res = results[0]
                if not len(res.boxes.data):
                    continue
                all_xyxy.extend(res.boxes.xyxy.cpu().numpy().tolist())
                all_masks.extend(res.masks.data.cpu().numpy())
                all_scores.extend(res.boxes.conf.cpu().numpy())
                all_cls.extend([pid] * len(res.masks.data))
        else:
            try:
                results = yolo.predict(image_path)

            except Exception as e:
                st.warning(
                    f"YOLO prediction failed for image '{image_path}': {e}"
                )
                results = []

            if results:
                res = results[0]
                # boxes
                try:
                    if hasattr(res, "boxes") and len(res.boxes.data):
                        all_xyxy.extend(res.boxes.xyxy.cpu().numpy().tolist())
                        all_scores.extend(res.boxes.conf.cpu().numpy())
                        try:
                            cls_arr = (
                                res.boxes.cls.cpu().numpy().astype(int).tolist()
                            )
                        except Exception:
                            cls_arr = [0] * len(res.boxes.conf)
                        all_cls.extend(cls_arr)
                except Exception:
                    pass

                # masks (if model produces them)
                try:
                    if (
                        hasattr(res, "masks")
                        and hasattr(res.masks, "data")
                        and len(res.masks.data)
                    ):
                        # res.masks.data likely shape (N, H, W)
                        all_masks.extend(
                            res.masks.data.cpu().numpy().astype(bool)
                        )
                except Exception:
                    pass

        if len(all_xyxy) == 0:
            # nothing detected for this image
            input_image_view.image(img)
            image_counter += 1
        elif len(all_masks) == 0:
            detections = sv.Detections(
                xyxy=np.array(all_xyxy),
                class_id=np.array(all_cls),
            )
            annotator = sv.BoxAnnotator()
            annotated_image = annotator.annotate(
                scene=img.copy(),
                detections=detections,
            )
            input_image_view.image(annotated_image)
            image_counter += 1
        else:
            print("IMAGE SHAPE", img.size)
            print("MASK SHAPE", all_masks[0].shape)
            w, h = all_masks[0].shape[1], all_masks[0].shape[0]
            detections = sv.Detections(
                xyxy=np.array(all_xyxy),
                mask=np.array(all_masks),
                class_id=np.array(all_cls),
            )

            # show images with masks
            annotator = sv.MaskAnnotator(
                opacity=0.5,
            )
            annotated_image = annotator.annotate(
                scene=img.copy().resize((w, h)),
                detections=detections,
            )
            input_image_view.image(annotated_image)
            image_counter += 1

        # Clean up temporary file
        if os.path.exists(image_path):
            os.remove(image_path)
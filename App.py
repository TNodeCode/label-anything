import streamlit as st
import os
import re
import numpy as np
import pandas as pd
from detection import Detection
from globox import AnnotationSet, Annotation, BoundingBox
from glob import glob
from ultralytics import SAM
import supervision as sv
from PIL import Image


CSV_FILES_DIR = "."
TRAIN_IMAGE_FILES_DIR = os.path.expanduser("~/PycharmProjects/DeepLearningProjects/PersonalProjects/CVDataInspector/datasets/spine/train")
TRAIN_ANNOTATION_FILE = "df_merged_union_train.csv"
VAL_IMAGE_FILES_DIR = os.path.expanduser("~/PycharmProjects/DeepLearningProjects/PersonalProjects/CVDataInspector/datasets/spine/val")
VAL_ANNOTATION_FILE = "df_merged_union_valid.csv"
TEST_IMAGE_FILES_DIR = os.path.expanduser("~/PycharmProjects/DeepLearningProjects/PersonalProjects/CVDataInspector/datasets/spine/test")
TEST_ANNOTATION_FILE = "df_merged_union_test.csv"


def df_to_bboxes(filepath) -> list[dict]:
    df = pd.read_csv(filepath)
    filenames = df["filename"].unique()
    bboxes_list = []
    for filename in filenames:
        bboxes = df[df["filename"] == filename]
        dict_item = {"filename": filename, "bboxes": [], "classes": []}
        for i, bbox in bboxes.iterrows():
            dict_item["bboxes"].append([int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])])
            dict_item["classes"].append(int(bbox["class_index"]))
        bboxes_list.append(dict_item)
    return bboxes_list

def predict_masks_by_bboxes(model_name: str, images_path: str, annotations: AnnotationSet) -> list[dict]:
    sam_model = SAM(model_name)
    n_annotations = len(annotations)
    for i, annotation in enumerate(annotations):
        print(f"Processing file {i+1}/{n_annotations}")
        filename = annotation.image_id
        bboxes = annotation.boxes
        for j, bbox, cls in enumerate(bboxes):
            print(f"Processing box {j+1}/{len(bboxes)}")
            images_path = os.path.expanduser(images_path)
            # result of runnign image through segmentation model
            result = sam_model(f"{images_path}/{filename}", bboxes=bbox)
            # extract mask as 2D boolean numpy array from result
            mask = result[0].masks.data.detach().cpu().squeeze().numpy()
            # convert 2D numpy boolean array to image object
            mask_image = Image.fromarray(result[0].masks.data.detach().cpu().squeeze().numpy())
            # save mask as PNG image
            output_image_path = os.path.expanduser(f"{images_path}-seg-l/{filename}-mask{j}-cls{cls}.png")
            mask_image.save(output_image_path)

#annotations = df_to_bboxes(f"{CSV_FILES_DIR}/{TRAIN_ANNOTATION_FILE}")
#predict_masks_by_bboxes(model_name="sam_l.pt", images_path=TRAIN_IMAGE_FILES_DIR, annotations=annotations)
            
st.markdown("# Label Anything with SAM")
st.markdown("This app allows you to use the Segment Anything model to create segmentation masks for your datasets that have already been labelled with bounding boxes.")

image_dir = st.text_input("Path to images")
annotation_format = st.selectbox("Annotation Format", ["COCO"])
annotation_source = st.text_input("Annotation files")
model_name = st.selectbox("Select Segmentatioon Model", ["sam_b.pt", "sam_l.pt", "mobile_sam.pt"])
btn_execute = st.button("Generate Binary Masks")

progress_bar_images = st.empty()
progress_bar_images_text = st.empty()
progress_bar_bboxes = st.empty()
progress_bar_bboxes_text = st.empty() #

col1, col2 = st.columns(2)
with col1:
    input_image_view = st.empty()
with col2:
    output_image_view = st.empty()
 
if annotation_format == "COCO" and annotation_source.strip() and btn_execute:
    annotations = list(AnnotationSet.from_coco(file_path=os.path.expanduser(annotation_source)))

    # load the model
    sam_model = SAM(model_name)
    n_annotations = len(annotations)

    # Create output directory if it does not exist
    image_path = os.path.expanduser(image_dir)
    output_dir = os.path.expanduser(f"{image_path}-{os.path.basename(model_name)}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over images
    for i, annotation in enumerate(annotations):
        progress_bar_images.progress((i+1) / n_annotations)
        progress_bar_images_text.text(f"Image {i+1} / {n_annotations}")

        n_bboxes = len(annotation.boxes)

        # Iterate over bounding boxes
        for j, bbox in enumerate(annotation.boxes):
            progress_bar_bboxes.progress((j+1) / n_bboxes)
            progress_bar_bboxes_text.text(f"Box {j+1} / {n_bboxes}")

            # load input image
            input_image_path = f"{os.path.expanduser(image_path)}/{annotation.image_id}"
            input_image = Image.open(input_image_path).copy()

            # Create detection object for bounding boxes
            detection_box = Detection.from_bboxes(
                np.array([[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]]).astype(int)
            )

            # result of runnign image through segmentation model
            result = sam_model(
                input_image_path,
                bboxes=[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
            )

            # extract mask as 2D boolean numpy array from result
            mask = result[0].masks.data.detach().cpu().squeeze().numpy()
            # convert 2D numpy boolean array to image object
            mask_image = Image.fromarray(result[0].masks.data.detach().cpu().squeeze().numpy().astype(int))
            
            # save mask as PNG image
            output_image_path = os.path.expanduser(f"{output_dir}/{os.path.basename(annotation.image_id)}-mask-{j}-cls-{bbox.label}.png")
            mask_image.save(output_image_path)

            detection_mask = Detection.from_masks(np.array([mask]))
 
            input_image_view.image(Detection.plot_detections(input_image.copy(), detection_box))
            output_image_view.image(Detection.plot_detections(input_image.copy(), detection_mask, show_bboxes=False))
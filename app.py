# ============================================================
# PNEUMOVISION AI
# One-File Explainable Pneumonia Detection System
#
# Dataset:
# Chest X-Ray Images (Pneumonia)
# Kaggle:
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
#
# Features:
# - Dataset explorer
# - Image preprocessing
# - Stratified train/validation split
# - Data augmentation
# - EfficientNetB0 transfer learning
# - Fine tuning
# - Class weighting
# - Threshold optimization
# - Accuracy / Precision / Recall / F1 / Specificity
# - ROC-AUC
# - Confusion matrix
# - ROC curve
# - Training history
# - X-ray prediction
# - Grad-CAM Explainable AI
# - Misclassification visualization
# - Streamlit dashboard
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

DATA_DIR = Path(
    "data/chest_xray"
)

ARTIFACT_DIR = Path(
    "artifacts"
)

MODEL_PATH = (
    ARTIFACT_DIR /
    "pneumovision_model.keras"
)

METRICS_PATH = (
    ARTIFACT_DIR /
    "metrics.json"
)

THRESHOLD_PATH = (
    ARTIFACT_DIR /
    "threshold.json"
)

HISTORY_PATH = (
    ARTIFACT_DIR /
    "history.json"
)

CM_PATH = (
    ARTIFACT_DIR /
    "confusion_matrix.png"
)

ROC_PATH = (
    ARTIFACT_DIR /
    "roc_curve.png"
)

ERRORS_PATH = (
    ARTIFACT_DIR /
    "misclassifications.csv"
)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(

    page_title="PneumoVision AI",

    page_icon="🫁",

    layout="wide",

    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

def add_custom_css():

    st.markdown(

        """
        <style>

        .stApp {

            background:
            linear-gradient(
                135deg,
                #06111f 0%,
                #0b1729 45%,
                #101d31 100%
            );

        }

        .block-container {

            max-width: 1450px;

            padding-top: 2rem;

            padding-bottom: 3rem;

        }

        .hero {

            padding: 35px;

            border-radius: 28px;

            background:
            linear-gradient(
                135deg,
                rgba(14,165,233,0.18),
                rgba(20,184,166,0.18),
                rgba(139,92,246,0.18)
            );

            border:
            1px solid
            rgba(255,255,255,0.12);

            box-shadow:
            0 10px 40px
            rgba(0,0,0,0.20);

            margin-bottom: 25px;

        }

        .hero h1 {

            font-size: 3.2rem;

            font-weight: 800;

            margin-bottom: 5px;

        }

        .hero p {

            font-size: 1.1rem;

            color: #cbd5e1;

        }

        .glass {

            padding: 22px;

            border-radius: 20px;

            background:
            rgba(255,255,255,0.05);

            border:
            1px solid
            rgba(255,255,255,0.08);

        }

        .metric-card {

            padding: 20px;

            border-radius: 18px;

            background:
            rgba(255,255,255,0.05);

            border:
            1px solid
            rgba(255,255,255,0.08);

            text-align: center;

        }

        .metric-title {

            color: #94a3b8;

            font-size: 0.85rem;

        }

        .metric-value {

            font-size: 1.8rem;

            font-weight: 800;

            margin-top: 5px;

        }

        .warning {

            padding: 18px;

            border-radius: 16px;

            background:
            rgba(245,158,11,0.10);

            border:
            1px solid
            rgba(245,158,11,0.35);

        }

        .success-box {

            padding: 20px;

            border-radius: 18px;

            background:
            rgba(16,185,129,0.12);

            border:
            1px solid
            rgba(16,185,129,0.35);

        }

        .danger-box {

            padding: 20px;

            border-radius: 18px;

            background:
            rgba(239,68,68,0.12);

            border:
            1px solid
            rgba(239,68,68,0.35);

        }

        </style>
        """,

        unsafe_allow_html=True
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_artifact_directory():

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def save_json(
    path,
    data
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


def load_json(
    path,
    default=None
):

    if not path.exists():

        return default

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# DATASET SCANNER
# ============================================================

def scan_dataset():

    rows = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        split_dir = (
            DATA_DIR /
            split
        )

        if not split_dir.exists():

            continue

        for class_name in [
            "NORMAL",
            "PNEUMONIA"
        ]:

            class_dir = (
                split_dir /
                class_name
            )

            if not class_dir.exists():

                continue

            label = (
                0
                if class_name == "NORMAL"
                else 1
            )

            for extension in [
                "*.jpeg",
                "*.jpg",
                "*.png"
            ]:

                for image_path in (
                    class_dir.glob(
                        extension
                    )
                ):

                    rows.append(

                        {

                            "path":
                                str(image_path),

                            "split":
                                split,

                            "class":
                                class_name,

                            "label":
                                label

                        }
                    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image
):

    if not isinstance(
        image,
        Image.Image
    ):

        image = Image.open(
            image
        )

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        IMG_SIZE
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


# ============================================================
# LOAD IMAGE FOR TF DATASET
# ============================================================

def load_image_tf(
    path,
    label
):

    image = tf.io.read_file(
        path
    )

    image = tf.io.decode_image(

        image,

        channels=3,

        expand_animations=False
    )

    image.set_shape(
        [
            None,
            None,
            3
        ]
    )

    image = tf.image.resize(

        image,

        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    label = tf.cast(
        label,
        tf.float32
    )

    return (
        image,
        label
    )


# ============================================================
# CREATE TF DATASET
# ============================================================

def create_tf_dataset(

    dataframe,

    shuffle=False,

    augment=False

):

    paths = dataframe[
        "path"
    ].values

    labels = dataframe[
        "label"
    ].values

    dataset = tf.data.Dataset.from_tensor_slices(

        (
            paths,
            labels
        )
    )

    if shuffle:

        dataset = dataset.shuffle(

            len(dataframe),

            seed=SEED
        )

    dataset = dataset.map(

        load_image_tf,

        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment:

        augmentation = tf.keras.Sequential(

            [

                tf.keras.layers.RandomFlip(
                    "horizontal"
                ),

                tf.keras.layers.RandomRotation(
                    0.05
                ),

                tf.keras.layers.RandomZoom(
                    0.10
                ),

                tf.keras.layers.RandomContrast(
                    0.10
                )

            ]
        )

        dataset = dataset.map(

            lambda x, y:
            (
                augmentation(
                    x,
                    training=True
                ),
                y
            ),

            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# BUILD EFFICIENTNET MODEL
# ============================================================

def build_model():

    inputs = tf.keras.Input(

        shape=(
            224,
            224,
            3
        ),

        name="chest_xray"
    )

    base_model = (
        tf.keras.applications.EfficientNetB0(

            include_top=False,

            weights="imagenet",

            input_shape=(
                224,
                224,
                3
            )
        )
    )

    base_model.trainable = False

    x = base_model(

        inputs,

        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(
        x
    )

    x = tf.keras.layers.BatchNormalization()(
        x
    )

    x = tf.keras.layers.Dropout(
        0.35
    )(
        x
    )

    x = tf.keras.layers.Dense(

        128,

        activation="relu"
    )(
        x
    )

    x = tf.keras.layers.Dropout(
        0.25
    )(
        x
    )

    outputs = tf.keras.layers.Dense(

        1,

        activation="sigmoid",

        name="pneumonia_probability"
    )(
        x
    )

    model = tf.keras.Model(

        inputs,

        outputs,

        name="PneumoVision_EfficientNetB0"
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=1e-3
        ),

        loss="binary_crossentropy",

        metrics=[

            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),

            tf.keras.metrics.AUC(
                name="auc"
            )

        ]
    )

    return (
        model,
        base_model
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    dataframe
):

    labels = dataframe[
        "label"
    ].values

    classes = np.unique(
        labels
    )

    weights = compute_class_weight(

        class_weight="balanced",

        classes=classes,

        y=labels
    )

    return {

        int(class_id):
        float(weight)

        for class_id, weight
        in zip(
            classes,
            weights
        )
    }


# ============================================================
# PREDICTIONS
# ============================================================

def get_probabilities(

    model,

    dataset

):

    return model.predict(

        dataset,

        verbose=0
    ).ravel()


# ============================================================
# FIND OPTIMAL THRESHOLD
# ============================================================

def find_best_threshold(

    y_true,

    probabilities

):

    best_threshold = 0.5

    best_f1 = 0

    for threshold in np.arange(

        0.20,

        0.81,

        0.01
    ):

        predictions = (

            probabilities
            >= threshold
        ).astype(int)

        score = f1_score(

            y_true,

            predictions,

            zero_division=0
        )

        if score > best_f1:

            best_f1 = score

            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_f1
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(

    model,

    test_dataframe,

    threshold

):

    test_dataset = create_tf_dataset(

        test_dataframe,

        shuffle=False,

        augment=False
    )

    probabilities = get_probabilities(

        model,

        test_dataset
    )

    y_true = test_dataframe[
        "label"
    ].values

    predictions = (

        probabilities
        >= threshold
    ).astype(int)

    matrix = confusion_matrix(

        y_true,

        predictions,

        labels=[
            0,
            1
        ]
    )

    tn, fp, fn, tp = (
        matrix.ravel()
    )

    specificity = (

        tn / (
            tn + fp
        )

        if (
            tn + fp
        ) > 0

        else 0
    )

    metrics = {

        "accuracy":
        float(
            accuracy_score(
                y_true,
                predictions
            )
        ),

        "precision":
        float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "recall":
        float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f1":
        float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "specificity":
        float(
            specificity
        ),

        "roc_auc":
        float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),

        "threshold":
        float(
            threshold
        ),

        "true_negative":
        int(tn),

        "false_positive":
        int(fp),

        "false_negative":
        int(fn),

        "true_positive":
        int(tp)

    }

    report = classification_report(

        y_true,

        predictions,

        target_names=[

            "NORMAL",

            "PNEUMONIA"

        ],

        output_dict=True,

        zero_division=0
    )

    return (

        metrics,

        report,

        matrix,

        probabilities,

        predictions
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(

    matrix

):

    figure, axis = plt.subplots(

        figsize=(
            6,
            5
        )
    )

    axis.imshow(
        matrix
    )

    axis.set_title(
        "Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted"
    )

    axis.set_ylabel(
        "Actual"
    )

    axis.set_xticks(

        [
            0,
            1
        ],

        [
            "NORMAL",
            "PNEUMONIA"
        ]
    )

    axis.set_yticks(

        [
            0,
            1
        ],

        [
            "NORMAL",
            "PNEUMONIA"
        ]
    )

    for i in range(2):

        for j in range(2):

            axis.text(

                j,

                i,

                str(
                    matrix[i, j]
                ),

                ha="center",

                va="center"
            )

    figure.tight_layout()

    figure.savefig(

        CM_PATH,

        dpi=180,

        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# SAVE ROC CURVE
# ============================================================

def save_roc_curve(

    y_true,

    probabilities

):

    fpr, tpr, _ = roc_curve(

        y_true,

        probabilities
    )

    auc_value = roc_auc_score(

        y_true,

        probabilities
    )

    figure, axis = plt.subplots(

        figsize=(
            7,
            5
        )
    )

    axis.plot(

        fpr,

        tpr,

        label=f"ROC-AUC = {auc_value:.3f}"
    )

    axis.plot(

        [
            0,
            1
        ],

        [
            0,
            1
        ],

        linestyle="--"
    )

    axis.set_xlabel(
        "False Positive Rate"
    )

    axis.set_ylabel(
        "True Positive Rate"
    )

    axis.set_title(
        "ROC Curve"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(

        ROC_PATH,

        dpi=180,

        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# SAVE MISCLASSIFICATIONS
# ============================================================

def save_misclassifications(

    dataframe,

    probabilities,

    predictions

):

    result = dataframe.copy()

    result[
        "probability_pneumonia"
    ] = probabilities

    result[
        "prediction"
    ] = np.where(

        predictions == 1,

        "PNEUMONIA",

        "NORMAL"
    )

    result[
        "correct"
    ] = (

        result["label"]
        == predictions
    )

    errors = result[
        result["correct"] == False
    ]

    errors.to_csv(

        ERRORS_PATH,

        index=False
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(

    dataframe,

    epochs,

    fine_tune_epochs

):

    ensure_artifact_directory()

    # --------------------------------------------------------
    # COMBINE TRAIN + VAL
    # --------------------------------------------------------

    development_data = dataframe[

        dataframe[
            "split"
        ].isin(
            [
                "train",
                "val"
            ]
        )

    ].copy()

    test_data = dataframe[

        dataframe[
            "split"
        ] == "test"

    ].copy()

    # --------------------------------------------------------
    # STRATIFIED SPLIT
    # --------------------------------------------------------

    train_data, validation_data = (
        train_test_split(

            development_data,

            test_size=0.15,

            random_state=SEED,

            stratify=development_data[
                "label"
            ]
        )
    )

    st.info(

        f"""
Training images: {len(train_data):,}

Validation images: {len(validation_data):,}

Test images: {len(test_data):,}
"""
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = create_tf_dataset(

        train_data,

        shuffle=True,

        augment=True
    )

    validation_dataset = create_tf_dataset(

        validation_data,

        shuffle=False,

        augment=False
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model, base_model = build_model()

    class_weights = calculate_class_weights(
        train_data
    )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            MODEL_PATH,

            monitor="val_auc",

            mode="max",

            save_best_only=True
        ),

        tf.keras.callbacks.EarlyStopping(

            monitor="val_auc",

            mode="max",

            patience=4,

            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.3,

            patience=2,

            min_lr=1e-7
        )

    ]

    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    st.subheader(
        "Stage 1 — Transfer Learning"
    )

    progress = st.progress(
        0
    )

    history1 = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=epochs,

        class_weight=class_weights,

        callbacks=callbacks
    )

    progress.progress(
        50
    )

    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------

    st.subheader(
        "Stage 2 — Fine Tuning"
    )

    base_model.trainable = True

    for layer in base_model.layers[:-40]:

        layer.trainable = False

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=1e-5
        ),

        loss="binary_crossentropy",

        metrics=[

            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),

            tf.keras.metrics.AUC(
                name="auc"
            )

        ]
    )

    history2 = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=fine_tune_epochs,

        class_weight=class_weights,

        callbacks=callbacks
    )

    progress.progress(
        100
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # FIND BEST THRESHOLD
    # --------------------------------------------------------

    validation_probabilities = (
        get_probabilities(

            model,

            validation_dataset
        )
    )

    validation_labels = (
        validation_data[
            "label"
        ].values
    )

    threshold, validation_f1 = (
        find_best_threshold(

            validation_labels,

            validation_probabilities
        )
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    (
        metrics,
        report,
        matrix,
        probabilities,
        predictions
    ) = evaluate_model(

        model,

        test_data,

        threshold
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_json(

        METRICS_PATH,

        metrics
    )

    save_json(

        THRESHOLD_PATH,

        {

            "threshold":
                threshold,

            "validation_f1":
                validation_f1

        }
    )

    history = {}

    for history_object in [
        history1,
        history2
    ]:

        for key, values in (
            history_object.history.items()
        ):

            history.setdefault(
                key,
                []
            )

            history[key].extend(

                [
                    float(value)
                    for value in values
                ]
            )

    save_json(

        HISTORY_PATH,

        history
    )

    save_confusion_matrix(
        matrix
    )

    save_roc_curve(

        test_data[
            "label"
        ].values,

        probabilities
    )

    save_misclassifications(

        test_data,

        probabilities,

        predictions
    )

    return (

        model,

        metrics,

        report
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_saved_model():

    if not MODEL_PATH.exists():

        return None

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# LOAD SAVED METRICS
# ============================================================

def get_saved_metrics():

    return load_json(

        METRICS_PATH,

        {}
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_xray(

    model,

    image,

    threshold

):

    array = preprocess_image(
        image
    )

    probability = float(

        model.predict(

            array,

            verbose=0
        )[0][0]
    )

    if probability >= threshold:

        label = "PNEUMONIA"

        confidence = probability

    else:

        label = "NORMAL"

        confidence = (
            1 -
            probability
        )

    return {

        "label":
            label,

        "pneumonia_probability":
            probability,

        "normal_probability":
            1 - probability,

        "confidence":
            confidence

    }


# ============================================================
# GRAD-CAM
# ============================================================

def find_last_conv_layer(
    model
):

    for layer in reversed(
        model.layers
    ):

        if isinstance(

            layer,

            tf.keras.layers.Conv2D

        ):

            return layer.name

        if hasattr(
            layer,
            "layers"
        ):

            for sublayer in reversed(
                layer.layers
            ):

                if isinstance(

                    sublayer,

                    tf.keras.layers.Conv2D

                ):

                    return sublayer.name

    raise ValueError(
        "No convolutional layer found."
    )


def generate_gradcam(

    image,

    model

):

    image_array = preprocess_image(
        image
    )

    last_conv_layer_name = (
        find_last_conv_layer(
            model
        )
    )

    last_conv_layer = (
        model.get_layer(
            last_conv_layer_name
        )
    )

    gradient_model = tf.keras.models.Model(

        inputs=model.inputs,

        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            gradient_model(
                image_array
            )
        )

        pneumonia_score = (
            predictions[:, 0]
        )

    gradients = tape.gradient(

        pneumonia_score,

        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(

        gradients,

        axis=(
            1,
            2
        )
    )

    conv_outputs = (
        conv_outputs[0]
    )

    heatmap = (

        conv_outputs
        @
        pooled_gradients[
            0
        ][..., tf.newaxis]

    )

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(
        heatmap
    )

    if float(maximum) > 0:

        heatmap /= maximum

    return (

        heatmap.numpy(),

        last_conv_layer_name
    )


# ============================================================
# OVERLAY GRAD-CAM
# ============================================================

def overlay_gradcam(

    image,

    heatmap

):

    import cv2

    original = np.array(

        image.convert(
            "RGB"
        )
    )

    original = cv2.resize(

        original,

        IMG_SIZE
    )

    heatmap = np.uint8(

        255 *
        heatmap
    )

    heatmap = cv2.applyColorMap(

        heatmap,

        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(

        heatmap,

        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(

        original,

        0.58,

        heatmap,

        0.42,

        0
    )

    return overlay


# ============================================================
# METRIC CARD
# ============================================================

def display_metric(

    title,

    value

):

    st.markdown(

        f"""
        <div class="metric-card">

            <div class="metric-title">
                {title}
            </div>

            <div class="metric-value">
                {value}
            </div>

        </div>
        """,

        unsafe_allow_html=True
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

def overview_page(
    dataframe
):

    st.markdown(

        """
        <div class="hero">

            <h1>
                🫁 PneumoVision AI
            </h1>

            <p>
                Explainable Deep Learning Framework
                for Pneumonia Detection from Chest X-Rays
            </p>

        </div>
        """,

        unsafe_allow_html=True
    )

    if dataframe.empty:

        st.error(
            "Dataset not found."
        )

        st.info(

            """
            Put the Kaggle dataset inside:

            data/chest_xray/

            Expected folders:

            train/NORMAL
            train/PNEUMONIA
            val/NORMAL
            val/PNEUMONIA
            test/NORMAL
            test/PNEUMONIA
            """
        )

        return

    total_images = len(
        dataframe
    )

    pneumonia_count = len(

        dataframe[
            dataframe[
                "class"
            ] == "PNEUMONIA"
        ]
    )

    normal_count = len(

        dataframe[
            dataframe[
                "class"
            ] == "NORMAL"
        ]
    )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:

        display_metric(

            "Total Images",

            f"{total_images:,}"
        )

    with c2:

        display_metric(

            "Normal",

            f"{normal_count:,}"
        )

    with c3:

        display_metric(

            "Pneumonia",

            f"{pneumonia_count:,}"
        )

    st.markdown(
        "### 📊 Dataset Distribution"
    )

    distribution = (

        dataframe
        .groupby(
            [
                "split",
                "class"
            ]
        )
        .size()
        .reset_index(
            name="Images"
        )
    )

    figure = px.bar(

        distribution,

        x="split",

        y="Images",

        color="class",

        barmode="group",

        text_auto=True,

        title="Dataset Distribution"
    )

    st.plotly_chart(

        figure,

        use_container_width=True
    )

    st.markdown(
        "### 🧠 Complete AI Pipeline"
    )

    st.code(

        """
Chest X-Ray
      ↓
Image Preprocessing
      ↓
224 × 224 Resize
      ↓
Data Augmentation
      ↓
EfficientNetB0
      ↓
Transfer Learning
      ↓
Global Average Pooling
      ↓
Dense + Dropout
      ↓
Sigmoid Probability
      ↓
Optimized Threshold
      ↓
NORMAL / PNEUMONIA
      ↓
Grad-CAM Explanation
        """
    )

    st.markdown(
        "### ✨ Why this project is stronger"
    )

    features = pd.DataFrame(

        {

            "Component": [

                "Transfer Learning",

                "Data Augmentation",

                "Class Weighting",

                "Fine Tuning",

                "Threshold Optimization",

                "Grad-CAM",

                "Interactive Dashboard"

            ],

            "Benefit": [

                "Uses pretrained visual representations",

                "Improves generalization",

                "Handles class imbalance",

                "Adapts pretrained features to X-rays",

                "Optimizes classification decision",

                "Provides visual explanation",

                "Makes the project interactive"

            ]

        }
    )

    st.dataframe(

        features,

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# TRAINING PAGE
# ============================================================

def training_page(
    dataframe
):

    st.title(
        "🚀 Train PneumoVision AI"
    )

    if dataframe.empty:

        st.error(
            "Dataset not found."
        )

        return

    st.warning(

        """
        Training a deep-learning model can take significant
        time depending on your CPU/GPU.
        """
    )

    c1, c2 = st.columns(
        2
    )

    with c1:

        epochs = st.slider(

            "Transfer Learning Epochs",

            1,

            30,

            10
        )

    with c2:

        fine_tune_epochs = st.slider(

            "Fine-Tuning Epochs",

            1,

            20,

            5
        )

    if st.button(

        "🔥 START TRAINING",

        type="primary",

        use_container_width=True
    ):

        with st.spinner(

            "Training model... Please wait."
        ):

            model, metrics, report = (
                train_model(

                    dataframe,

                    epochs,

                    fine_tune_epochs
                )
            )

        st.success(
            "Training completed!"
        )

        st.balloons()

        st.json(
            metrics
        )

        load_saved_model.clear()


# ============================================================
# PREDICTION PAGE
# ============================================================

def prediction_page():

    st.title(
        "🔬 AI X-Ray Prediction"
    )

    model = load_saved_model()

    if model is None:

        st.warning(

            """
            No trained model found.

            Go to the Training page first.
            """
        )

        return

    threshold_data = load_json(

        THRESHOLD_PATH,

        {
            "threshold": 0.5
        }
    )

    threshold = float(

        threshold_data.get(
            "threshold",
            0.5
        )
    )

    uploaded_file = st.file_uploader(

        "📤 Upload Chest X-Ray",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is None:

        st.info(
            "Upload an X-ray to begin."
        )

        return

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    result = predict_xray(

        model,

        image,

        threshold
    )

    left, right = st.columns(
        2
    )

    with left:

        st.image(

            image,

            caption="Uploaded Chest X-Ray",

            use_container_width=True
        )

    with right:

        st.subheader(
            "AI Prediction"
        )

        if result["label"] == "PNEUMONIA":

            st.markdown(

                """
                <div class="danger-box">

                <h2>
                🫁 PNEUMONIA
                </h2>

                </div>
                """,

                unsafe_allow_html=True
            )

        else:

            st.markdown(

                """
                <div class="success-box">

                <h2>
                ✅ NORMAL
                </h2>

                </div>
                """,

                unsafe_allow_html=True
            )

        st.metric(

            "Pneumonia Probability",

            f"""
            {result["pneumonia_probability"] * 100:.2f}%
            """
        )

        st.metric(

            "Normal Probability",

            f"""
            {result["normal_probability"] * 100:.2f}%
            """
        )

        st.metric(

            "Model Confidence",

            f"""
            {result["confidence"] * 100:.2f}%
            """
        )

        st.progress(

            result[
                "pneumonia_probability"
            ]
        )

        st.caption(

            f"""
            Decision threshold:
            {threshold:.2f}
            """
        )

    st.divider()

    st.subheader(
        "📊 Prediction Probability"
    )

    probability_df = pd.DataFrame(

        {

            "Class": [

                "NORMAL",

                "PNEUMONIA"

            ],

            "Probability": [

                result[
                    "normal_probability"
                ],

                result[
                    "pneumonia_probability"
                ]

            ]

        }
    )

    figure = px.bar(

        probability_df,

        x="Class",

        y="Probability",

        range_y=[
            0,
            1
        ],

        text_auto=".2f",

        title="AI Probability Distribution"
    )

    st.plotly_chart(

        figure,

        use_container_width=True
    )

    if result["confidence"] < 0.65:

        st.warning(

            """
            ⚠️ The model is relatively uncertain about
            this image. This result should not be interpreted
            as a definitive medical diagnosis.
            """
        )


# ============================================================
# EXPLAINABILITY PAGE
# ============================================================

def explainability_page():

    st.title(
        "🔥 Explainable AI — Grad-CAM"
    )

    st.write(

        """
        Grad-CAM highlights image regions that contribute
        to the model's pneumonia probability.
        """
    )

    model = load_saved_model()

    if model is None:

        st.warning(
            "Train the model first."
        )

        return

    threshold_data = load_json(

        THRESHOLD_PATH,

        {
            "threshold": 0.5
        }
    )

    threshold = float(

        threshold_data.get(
            "threshold",
            0.5
        )
    )

    uploaded_file = st.file_uploader(

        "Upload X-Ray",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        key="explainable_xray"
    )

    if uploaded_file is None:

        return

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    result = predict_xray(

        model,

        image,

        threshold
    )

    heatmap, layer = generate_gradcam(

        image,

        model
    )

    overlay = overlay_gradcam(

        image,

        heatmap
    )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:

        st.image(

            image,

            caption="Original X-Ray",

            use_container_width=True
        )

    with c2:

        st.image(

            overlay,

            caption="Grad-CAM Explanation",

            use_container_width=True
        )

    with c3:

        st.subheader(
            "AI Analysis"
        )

        st.metric(

            "Prediction",

            result["label"]
        )

        st.metric(

            "Pneumonia Probability",

            f"""
            {result["pneumonia_probability"] * 100:.2f}%
            """
        )

        st.caption(

            f"""
            Last convolutional layer:

            {layer}
            """
        )

    st.divider()

    st.info(

        """
        🔥 The colored regions represent areas that
        contributed more strongly to the model's pneumonia
        prediction. Grad-CAM explains model behavior but
        does not identify a medically validated lesion.
        """
    )


# ============================================================
# PERFORMANCE PAGE
# ============================================================

def performance_page():

    st.title(
        "📊 Model Performance"
    )

    metrics = get_saved_metrics()

    if not metrics:

        st.warning(
            "Train the model first."
        )

        return

    columns = st.columns(
        6
    )

    values = [

        (
            "Accuracy",
            metrics.get(
                "accuracy",
                0
            )
        ),

        (
            "Precision",
            metrics.get(
                "precision",
                0
            )
        ),

        (
            "Recall",
            metrics.get(
                "recall",
                0
            )
        ),

        (
            "F1",
            metrics.get(
                "f1",
                0
            )
        ),

        (
            "Specificity",
            metrics.get(
                "specificity",
                0
            )
        ),

        (
            "ROC-AUC",
            metrics.get(
                "roc_auc",
                0
            )
        )

    ]

    for column, (
        title,
        value
    ) in zip(
        columns,
        values
    ):

        with column:

            display_metric(

                title,

                f"{value:.3f}"
            )

    st.divider()

    left, right = st.columns(
        2
    )

    with left:

        if CM_PATH.exists():

            st.image(

                str(CM_PATH),

                caption="Confusion Matrix"
            )

    with right:

        if ROC_PATH.exists():

            st.image(

                str(ROC_PATH),

                caption="ROC Curve"
            )

    # --------------------------------------------------------
    # CONFUSION MATRIX AS INTERACTIVE CHART
    # --------------------------------------------------------

    st.subheader(
        "🧮 Confusion Matrix"
    )

    tn = metrics.get(
        "true_negative",
        0
    )

    fp = metrics.get(
        "false_positive",
        0
    )

    fn = metrics.get(
        "false_negative",
        0
    )

    tp = metrics.get(
        "true_positive",
        0
    )

    matrix = np.array(

        [
            [
                tn,
                fp
            ],

            [
                fn,
                tp
            ]
        ]
    )

    figure = go.Figure(

        data=go.Heatmap(

            z=matrix,

            x=[
                "NORMAL",
                "PNEUMONIA"
            ],

            y=[
                "NORMAL",
                "PNEUMONIA"
            ],

            text=matrix,

            texttemplate="%{text}",

            colorscale="Blues"
        )
    )

    figure.update_layout(

        xaxis_title="Predicted",

        yaxis_title="Actual"
    )

    st.plotly_chart(

        figure,

        use_container_width=True
    )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history = load_json(

        HISTORY_PATH,

        {}
    )

    if history:

        st.subheader(
            "📈 Training History"
        )

        rows = []

        for metric_name, values in (
            history.items()
        ):

            for epoch, value in enumerate(

                values,

                start=1

            ):

                rows.append(

                    {

                        "Epoch":
                            epoch,

                        "Metric":
                            metric_name,

                        "Value":
                            value

                    }
                )

        history_df = pd.DataFrame(
            rows
        )

        figure = px.line(

            history_df,

            x="Epoch",

            y="Value",

            color="Metric",

            markers=True,

            title="Training History"
        )

        st.plotly_chart(

            figure,

            use_container_width=True
        )

    # --------------------------------------------------------
    # MISCLASSIFICATIONS
    # --------------------------------------------------------

    if ERRORS_PATH.exists():

        st.subheader(
            "❌ Misclassified Images"
        )

        errors = pd.read_csv(
            ERRORS_PATH
        )

        st.metric(

            "Misclassified Test Images",

            len(errors)
        )

        if not errors.empty:

            st.dataframe(

                errors[
                    [
                        "path",
                        "class",
                        "prediction",
                        "probability_pneumonia"
                    ]
                ],

                use_container_width=True
            )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    add_custom_css()

    dataframe = scan_dataset()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.markdown(
        "## 🫁 PneumoVision AI"
    )

    st.sidebar.caption(
        "Explainable Medical Image AI"
    )

    st.sidebar.divider()

    page = st.sidebar.radio(

        "Navigation",

        [

            "🏠 Dashboard",

            "🚀 Train Model",

            "🔬 Predict X-Ray",

            "🔥 Explainability",

            "📊 Performance"

        ]
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        "### Dataset"
    )

    if dataframe.empty:

        st.sidebar.error(
            "Dataset not found"
        )

    else:

        st.sidebar.success(
            "Dataset detected"
        )

        st.sidebar.write(

            f"""
Images: {len(dataframe):,}
"""
        )

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    if page == "🏠 Dashboard":

        overview_page(
            dataframe
        )

    elif page == "🚀 Train Model":

        training_page(
            dataframe
        )

    elif page == "🔬 Predict X-Ray":

        prediction_page()

    elif page == "🔥 Explainability":

        explainability_page()

    elif page == "📊 Performance":

        performance_page()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
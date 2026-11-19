"""Send one plant leaf image to the local Triton model and print its scores.

Usage:
  python infer_triton.py path\to\leaf.jpg
  python infer_triton.py path\to\leaf.jpg --classes "class_a,class_b,class_c"
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


MODEL_NAME = "plant_disease_model"
SERVER_URL = "localhost:8000"
IMAGE_SIZE = (128, 128)  # width, height


def main():
    parser = argparse.ArgumentParser(description="Run plant disease inference through Triton.")
    parser.add_argument("image", help="Path to a JPG or PNG image")
    parser.add_argument(
        "--classes",
        help="Optional comma-separated class names in the exact training label order",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"Could not open image: {image_path}")

    # Match the notebook: RGB, resize to 128x128, convert to float, NCHW.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    tensor = image_rgb.astype(np.float32).transpose(2, 0, 1)[None, ...]

    client = InferenceServerClient(url=SERVER_URL)
    if not client.is_model_ready(MODEL_NAME):
        raise SystemExit(f"Triton model is not ready: {MODEL_NAME} at {SERVER_URL}")

    triton_input = InferInput("image", tensor.shape, "FP32")
    triton_input.set_data_from_numpy(tensor)
    response = client.infer(
        model_name=MODEL_NAME,
        inputs=[triton_input],
        outputs=[InferRequestedOutput("probabilities")],
    )

    scores = response.as_numpy("probabilities")[0]
    predicted_index = int(np.argmax(scores))
    class_names = args.classes.split(",") if args.classes else None

    print(f"Image: {image_path.name}")
    if class_names and len(class_names) == len(scores):
        print(f"Prediction: {class_names[predicted_index]}")
    else:
        print(f"Predicted class index: {predicted_index}")
        if class_names:
            print(f"Note: supplied {len(class_names)} class names, but model returned {len(scores)} scores.")
    print(f"Scores: {scores.tolist()}")
    print(f"Top score: {scores[predicted_index]:.6f}")


if __name__ == "__main__":
    main()

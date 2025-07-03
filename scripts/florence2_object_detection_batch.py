"""
Script for performing object detection on a batch of images using the Florence-2 model.

To run this script, use the following command:
python scripts/florence2_object_detection_batch.py --input_dir "/path/to/your/input/images" --output_dir "/path/to/your/output/directory"

Arguments:
  --input_dir: Directory containing the input images (e.g., .png, .jpg).
  --output_dir: Directory to save the output (annotated images and YOLO labels).
"""

import torch
import argparse
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import snapshot_download

def setup_directories(input_dir, output_dir):
    """Creates the output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return Path(input_dir), output_path

def load_model_and_processor():
    """
    Downloads and loads the Florence-2 large model and its associated processor.

    The model and processor are downloaded from Hugging Face Hub.
    It attempts to use a CUDA-enabled GPU if available, otherwise defaults to CPU.
    The `torch_dtype` is set to float16 for CUDA for efficiency, and float32 for CPU.

    Returns:
        tuple: A tuple containing:
            - model (AutoModelForCausalLM): The loaded Florence-2 model.
            - processor (AutoProcessor): The loaded Florence-2 processor.
            - device (str): The device (e.g., "cuda:0" or "cpu") on which the model is loaded.
            - torch_dtype (torch.dtype): The torch data type used for the model (e.g., torch.float16).
    """
    print("Downloading and loading Florence-2 model...")
    local_dir = snapshot_download(repo_id="microsoft/Florence-2-large")
    
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    torch_dtype = torch.float16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
    print("✅ Model and processor loaded.")
    return model, processor, device, torch_dtype

def process_image(image_path, model, processor, device, torch_dtype):
    """
    Runs object detection on a single image using the loaded Florence-2 model.

    The image is processed with the prompt "<OD>" to perform object detection.
    The generated output from the model is then post-processed to extract
    bounding box coordinates and labels.

    Args:
        image_path (Path): The file path to the image to be processed.
        model (AutoModelForCausalLM): The loaded Florence-2 model.
        processor (AutoProcessor): The loaded Florence-2 processor.
        device (str): The device (e.g., "cuda:0" or "cpu") the model is on.
        torch_dtype (torch.dtype): The torch data type used for model inference.

    Returns:
        tuple: A tuple containing:
            - parsed_answer (dict): A dictionary containing the parsed object detection results,
                                    including bounding boxes and labels.
            - image (PIL.Image.Image): The opened PIL Image object.
    """
    print(f"\nProcessing {image_path}...")
    image = Image.open(image_path).convert("RGB")
    prompt = "<OD>"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device, dtype=torch_dtype if k == "pixel_values" else v.dtype) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False
        )
        
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text,
        task="<OD>",
        image_size=(image.width, image.height)
    ), image

def save_annotated_image(output_path, image, parsed_answer):
    """
    Saves the provided image with detected 'person' bounding boxes and labels drawn on it.

    This function creates a matplotlib plot, draws the bounding boxes and labels
    for any detected "person" objects, and saves the resulting image to the
    specified output path.

    Args:
        output_path (Path): The file path where the annotated image will be saved.
        image (PIL.Image.Image): The original PIL Image object to annotate.
        parsed_answer (dict): A dictionary containing the parsed object detection results
                              from the Florence-2 model, expected to have an "<OD>" key
                              with "bboxes" and "labels".
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    od_result = parsed_answer.get("<OD>", {})
    for bbox, label in zip(od_result.get("bboxes", []), od_result.get("labels", [])):
        if label.lower() == "person":  # Define a class to filter
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1] - 10, label, color='white',
                    bbox=dict(facecolor='red', alpha=0.5), fontsize=8)

    ax.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_yolo_labels(output_path, image_width, image_height, parsed_answer):
    """
    Saves object detection labels in YOLO format, specifically for 'person' class.

    This function iterates through detected objects and if a 'person' is identified,
    it calculates and saves their bounding box coordinates in YOLO format
    (class_id center_x center_y width height).
    The class ID for 'person' is hardcoded as '0'.

    Args:
        output_path (Path): The file path where the YOLO label file will be saved.
        image_width (int): The width of the original image in pixels.
        image_height (int): The height of the original image in pixels.
        parsed_answer (dict): A dictionary containing the parsed object detection results
                              from the Florence-2 model, expected to have an "<OD>" key
                              with "bboxes" and "labels".
    """
    od_result = parsed_answer.get("<OD>", {})

    with open(output_path, "w") as f:
        for bbox, label in zip(od_result.get("bboxes", []), od_result.get("labels", [])):
            if label.lower() == "person":  # Define a class to filter
                x_center = ((bbox[0] + bbox[2]) / 2) / image_width
                y_center = ((bbox[1] + bbox[3]) / 2) / image_height
                width = (bbox[2] - bbox[0]) / image_width
                height = (bbox[3] - bbox[1]) / image_height
                # Ensure '0' is the correct class ID for 'person' in your YOLO dataset
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main(args):
    """
    Main function to orchestrate the batch object detection process using Florence-2.

    This function performs the following steps:
    1. Sets up the input and output directories.
    2. Loads the Florence-2 model and processor.
    3. Iterates through all PNG and JPG images in the input directory.
    4. For each image, it performs object detection.
    5. If objects are detected, it saves an annotated image with bounding boxes
       and a YOLO-format label file (filtered for 'person' class).
    6. Prints status messages throughout the process.

    Args:
        args (argparse.Namespace): An object containing command-line arguments,
                                   specifically `args.input_dir` and `args.output_dir`.
    """
    input_path, output_path = setup_directories(args.input_dir, args.output_dir)
    model, processor, device, torch_dtype = load_model_and_processor()

    image_paths = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))

    for image_path in image_paths:
        print(f"\nProcessing {image_path.name}...")
        
        parsed_answer, image = process_image(image_path, model, processor, device, torch_dtype)
        
        if not parsed_answer.get("<OD>", {}).get("bboxes"):
            print(f"⚠️ No objects detected in {image_path.name}. Skipping.")
            continue
            
        # Save annotated image
        annotated_image_path = output_path / image_path.name
        save_annotated_image(annotated_image_path, image, parsed_answer)

        # Save YOLO labels
        yolo_label_path = output_path / (image_path.stem + ".txt")
        save_yolo_labels(yolo_label_path, image.width, image.height, parsed_answer)

    print(f"\n✅ Done. Images and labels saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection using Florence-2 model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images and labels.")
    args = parser.parse_args()
    main(args)
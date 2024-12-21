This is a simple image inpainting tool that given a photo with a face is able to put the face in a generated context using cv2 for the face detection and HF inpainting pipelines for local inpainting.

To make it run, you need a `.env` file containing your `HUGGINGFACE_HUB_TOKEN` token since it needs download the inpainting model the first time.
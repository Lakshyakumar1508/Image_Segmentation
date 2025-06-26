import os
import uuid
from django.conf import settings
from django.shortcuts import render
from .forms import ImageUploadForm
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Function to perform KMeans segmentation
def process_image(image_path, result_name):
    result_rel_path = f"segmented/{result_name}"
    result_full_path = os.path.join(settings.MEDIA_ROOT, result_rel_path)

    # ✅ If the file already exists, skip processing
    if os.path.exists(result_full_path):
        return result_rel_path

    # Otherwise, process and segment
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize((100, 100))
    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(pixels)

    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_].reshape(100, 100, 3).astype(np.uint8)
    segmented_image = Image.fromarray(segmented_pixels)

    # Resize back to original
    segmented_image = segmented_image.resize(original_size)
    os.makedirs(os.path.dirname(result_full_path), exist_ok=True)
    segmented_image.save(result_full_path)

    return result_rel_path

  

# View for handling image upload and result rendering
def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            # Save uploaded image
            image = request.FILES['image']
            image_name = image.name
            upload_folder = os.path.join(settings.MEDIA_ROOT, 'uploaded')
            os.makedirs(upload_folder, exist_ok=True)
            uploaded_path = os.path.join(upload_folder, image_name)

            with open(uploaded_path, 'wb+') as f:
                for chunk in image.chunks():
                    f.write(chunk)

            # Process image with KMeans
            result_name = f"result_{uuid.uuid4().hex}.png"
            result_image_path = process_image(uploaded_path, result_name)

            # Check for optional ground truth image
            ground_truth_filename = f"{os.path.splitext(image_name)[0]}.png"
            ground_truth_path = os.path.join(settings.MEDIA_ROOT, 'ground_truth', ground_truth_filename)
            ground_truth_exists = os.path.exists(ground_truth_path)
            ground_truth_image = f"ground_truth/{ground_truth_filename}" if ground_truth_exists else None

            return render(request, 'clustering/index.html', {
                'form': form,
                'original_image': f"uploaded/{image_name}",
                'result_image': result_image_path,
                'ground_truth_image': ground_truth_image
            })
    else:
        form = ImageUploadForm()

    return render(request, 'clustering/index.html', {'form': form})

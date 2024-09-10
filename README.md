
# RCNN Crop and Weed Classification

This repository implements an RCNN (Region-based Convolutional Neural Network) model for detecting and classifying crops and weeds in agricultural images. The project utilizes selective search for region proposals, a pre-trained CNN for feature extraction, and an SVM classifier for final classification. Non-Maximum Suppression (NMS) is also applied to reduce overlapping bounding boxes.


## Installation

To set up the project locally, follow these steps:

```bash
git clone https://github.com/yourusername/RCNN-Crop-Weed-Detection.git
cd RCNN-Crop-Weed-Detection
```

Install the required dependencies using pip:
```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python
pip install numpy pandas matplotlib tensorflow scikit-learn tqdm pickle warnings
```



    
## Dataset

The dataset should be structured as follows:
```bash
/path/to/dataset/
    ├── train/           # Training images
    ├── test/            # Test images
    └── annotations/     # Annotations in the form of bounding boxes and labels
```

You can either create your own dataset or find publicly available datasets related to crop and weed detection.
## Usage/Examples

Once the dataset is set up and dependencies are installed, you can run the detection process on an image using the detection function in the main.py script.

Example usage
```python
detection('path/to/test/image.jpeg', confidence=0.9, iou_thresh=0.1)

```


## Parameters

- img_path: Path to the image to be processed.
- confidence: Confidence threshold for filtering out low-confidence predictions (default: 0.9).
- iou_thresh: IoU threshold for Non-Maximum Suppression (default: 0.1).
The output will visualize the image with bounding boxes around detected crops and weeds. The detection results are saved as prediction.jpeg.
## Deployment

Ensure your pre-trained RCNN model and SVM classifier are loaded:

```python
model_path = 'path/to/RCNN_model.h5'
svm_model_path = 'path/to/svm_classifier.pkl'
```
Call the detection function with your test image path.


## Results

The results include:

- Green boxes for detected crops.
- Blue boxes for detected weeds.
These results are displayed on the image and saved for further analysis.
## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests for the following:

- Bug fixes or improvements
- New feature requests
- Model enhancements (e.g., improving the detection accuracy or adding more classes)
To contribute:
    
    1. Fork the repository.
    2. Create your feature branch (git checkout -b feature/NewFeature).
    3. Commit your changes (git commit -m 'Add NewFeature').
    4. Push to the branch (git push origin feature/NewFeature).
    5. Open a pull request.
Please make sure your code follows the standard Python coding style (PEP8) and is well-documented.




## License
This project is licensed under the MIT License. For more details, see the LICENSE file in the root of the repository.

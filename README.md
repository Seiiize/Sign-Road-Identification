# Traffic Sign Identification Project

This project aims to automatically identify traffic signs using a machine learning model. The model has been trained to achieve high accuracy in recognizing traffic signs.

## Project Structure

- `sign_road_images/`  
  This folder contains the base images of the traffic signs. These images are used to train the model.

- `image_resize.py`  
  Python script to resize base images to 32x32 pixels. Resizing is crucial for preparing the input data for the model.

- `sign_road_image/`  
  This folder contains the resized images at 32x32 pixels, ready for training and validating the model.

- `sign_road_identification.py`  
  Python script containing the machine learning model used for traffic sign identification. The model achieved 98% accuracy on validation and 99% on testing.

## Prerequisites

Before you start, make sure to have the necessary dependencies installed. You can use a `requirements.txt` file to manage the required libraries. Here is an example of what this file might contain:

numpy
pandas
scikit-learn
tensorflow
opencv-python

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [REPOSITORY_URL]
cd [DIRECTORY_NAME]
pip install -r requirements.txt
```
## Usage

### Prepare the Images

Run the `image_resize.py` script to resize the base images to 32x32 pixels:

```bash
python image_resize.py
```
This script will read images from the sign_road_images folder, resize them, and save them in the sign_road_image folder.
## Train the Model

Run the `sign_road_identification.py` script to train the model with the resized images:

```bash
python sign_road_identification.py
```
This script will train the machine learning model and display the model's performance on validation and testing.
## Model Evaluation

The machine learning model achieved the following results:

- **Validation Accuracy:** 98%
- **Test Accuracy:** 99%

These results indicate that the model performs well in identifying traffic signs.

## Warnings

- Ensure that the `sign_road_images` folder contains the base images before running the resizing script.
- The resized images will be stored in the `sign_road_image` folder. Make sure this folder is empty before running the resizing script to avoid conflicts.

## Contributions

Contributions are welcome! If you would like to improve this project, please submit a pull request.

## License

This project is licensed under the [License Name]. See the `LICENSE` file for more details.

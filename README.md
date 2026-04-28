## Introduction
OpenCV-based skew correction tool for sheet music images.

## Requirements
```bash
pip install opencv-python numpy
```
## Usage
For Single Image
```bash
python deskew.py input.jpg output.jpg
```
Batch Processing
```bash
python deskew.py --batch "*.jpg" ./output_folder/
```
## Notes
Supports: .jpg .png .bmp .tiff

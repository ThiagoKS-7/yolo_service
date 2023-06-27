
## Getting started

#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov3-cpu
```

#### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt
```

### Downloading official pretrained weights
For Linux: Let's download official yolov3 weights pretrained on COCO dataset. 

# yolov3
```bash
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights
```

### Saving your yolov3 weights as a TensorFlow model.
Load the weights using `load_weights.py` script. This will convert the yolov3 weights into TensorFlow .ckpt model files!

# yolov3
```bash
python load_weights.py

```


## Acknowledgements:

- API Inspired by: https://github.com/theAIGuysCode/Object-Detection-API
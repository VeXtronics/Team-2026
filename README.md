# Computer Vision-Based Analysis of Corn Leaf Orientation for Precision Agriculture
**Team Internship at VexTronics**
Supervised by ir. E.A.J. Vermeer

Merve Şafak (2235943), Athanasios Poulis (2235309), Alecsandru Kreefft-Libiu (2371723), Miquel Suasso de Lima de Prado (1580140)
Eindhoven University of Technology

---

## What this project is

This project was carried out as a group internship in collaboration with VexTronics. The goal was to build a computer vision pipeline that takes top-down photos of corn plants, finds each individual plant, and scores how well it is aligned with the row it was planted in.

The motivation behind it is that corn leaf orientation affects how sunlight is distributed across the canopy. When leaves grow mainly along the row, neighboring plants shade each other more. When they grow across the row, light gets in more evenly. Whether seed placement influences this was one of the questions VexTronics wanted to explore, and automating the measurement was the contribution we could make.

The pipeline uses a fine-tuned YOLO11-pose model to detect each corn plant and predict three keypoints: the stem center and both leaf tips. From those points, we calculate the plant's position, the leaf axis angle, and how far both deviate from what we would expect for a well-planted row. The output is a 0-100 alignment score per plant with row-level summaries exported to CSV. There is also a live video demo notebook that runs the same logic on a camera feed in real time.

The full write-up, including the literature review, methodology, results, and discussion, is in the report/ folder.

---

## Folder structure

```
project/
├── notebooks/
│   ├── Corn_Alignment_Pipeline.ipynb    (main pipeline, start here)
│   └── Corn_Alignment_Prototype_Video.ipynb (live camera demo)
│
├── dataset/
│   └── our own greenhouse photos combined with images from two external sources
│       (see Dataset section below for details and citations)
│
├── papers/
│   └── all papers referenced in the report are saved here for convenience
│       (see the report for full context on how each was used)
│
├── presentations/
│   ├── VexTronics intro presentation
│   ├── midterm presentation
│   └── final presentation
│
└── report/
    └── team internship report (full write-up of the project)
```

---

## How to run the pipeline

### 1. Python version

We used **Python 3.10**. Stick to 3.10 or 3.11 because some of the YOLO and CUDA dependencies are picky about version compatibility.

### 2. Install the required libraries

```
pip install ultralytics==8.3.0
pip install opencv-python==4.9.0.80
pip install scikit-learn==1.4.2
pip install matplotlib==3.8.4
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install pyyaml==6.0.1
pip install Pillow==10.3.0
```

For the optional significance test in the results section:

```
pip install scipy==1.13.0
```

### 3. GPU and CUDA

Training needs a GPU. We used CUDA 12.1. If you get a CUDA error on startup, reinstall PyTorch with the right build:

```
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you only want to run inference without retraining, a CPU will work but expect it to be slow.

### 4. Set up the dataset

The dataset/ folder already contains everything we used. Copy the contents into:

```
project/raw_dataset/
```

The notebook expects train/, valid/, and optionally test/ subfolders, each with images/ and labels/ inside. If you export a fresh annotated dataset from Roboflow in YOLOv8 Pose format, unzip it into raw_dataset/ the same way.

### 5. Set your project path

Near the top of the notebook, change PROJECT_ROOT to point to your local copy:

```python
PROJECT_ROOT = Path(r"your/path/here")
```

### 6. Run the notebook

Open `notebooks/Corn_Alignment_Pipeline.ipynb` and run top to bottom. The sections are:

1. Setup and GPU check
2. Dataset preparation
3. YOLO config file
4. Model training (YOLO11n-pose and optionally YOLO11s-pose)
5. Model evaluation and keypoint pixel error
6. Inference wrapper
7. Row identity helper
8. Optimal line fitting (RANSAC)
9. Scoring logic
10. Full pipeline run and CSV export
11. Results and comparison plots

Results land in `project/results/plants.csv` and `project/results/rows.csv`.

---

## Live demo notebook

`Corn_Alignment_Prototype_Video.ipynb` runs the same detection and scoring on a live camera feed. It shares sections 1-6 and 8-9 with the main pipeline and replaces the rest with a real-time video stream loop. You need a webcam or USB camera connected. Set the camera index at the top of the notebook (0 for built-in, 1 for USB is usually right).

---

## Libraries overview

| Library | Version | What it is used for |
|---------|---------|---------------------|
| ultralytics | 8.3.0 | YOLO11-pose training and inference |
| opencv-python | 4.9.0.80 | Image loading, drawing, visualization |
| scikit-learn | 1.4.2 | RANSAC line fitting for row reference lines |
| numpy | 1.26.4 | Array operations and geometry math |
| pandas | 2.2.2 | Reading and analyzing the output CSVs |
| matplotlib | 3.8.4 | Plots and inline image display in the notebook |
| pyyaml | 6.0.1 | Writing the YOLO data.yaml config file |
| Pillow | 10.3.0 | EXIF-aware image loading so phone photos rotate correctly |
| scipy | 1.13.0 | Optional Welch t-test on alignment scores in section 11 |

---

## Dataset

The dataset/ folder has all the images and labels used for training and validation. It came from three sources.

**Our own data**: Sweet corn (suikermais) grown in a greenhouse in Heemskerk. We took top-down photos weekly from early vegetative stages (around V2-V4) up to V6 using a smartphone on a fixed mounting rig to keep the angle consistent. All images were annotated manually in Roboflow: one bounding box per plant, plus three keypoints (stem center, left leaf tip, right leaf tip). This is the main source for the leaf-orientation analysis.

**Brilhador et al. (2015)**: An external corn image dataset from a computer vision paper on inter-plant spacing measurement. It added useful field-like variation (different soil backgrounds, lighting, plant spacing) that our greenhouse setup did not cover. Because this dataset has no leaf-tip keypoint annotations, it was only used to help the detection stage, not orientation scoring.

> A. Brilhador, D. A. Serrarens, and F. M. Lopes, "A computer vision approach for automatic measurement of the inter-plant spacing," in Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications, ser. Lecture Notes in Computer Science. Springer, 2015, pp. 219-227.

**Roboflow weed dataset**: Added to help the model learn to tell corn apart from weeds. In top-down field images, early-stage corn and weeds can look similar, so having a separate weed class during training reduced false positive corn detections. This dataset was used only for that purpose and was not used in the leaf-orientation analysis.

> Detection, "weed dataset," https://universe.roboflow.com/detection-rni73/weed-xnmcb, accessed 30 May 2026.

All three sources were reformatted into YOLOv8 Pose format before training. The combined dataset as used is what you will find in dataset/.

---

## Papers

The papers/ folder contains all the papers we referenced during the project. They cover topics including precision agriculture, maize canopy structure, seed orientation research, YOLO architecture, and plant phenotyping. All 21 references are listed with full citations in the report. If you are reading the report and want to find a paper quickly, check the papers/ folder first before searching online.

---

## Notes for anyone picking this up later

A few things worth knowing that are not obvious from the code alone:

- The scoring tolerances (POS_TOLERANCE_PX and ANG_TOLERANCE_DEG in section 9) are set to reasonable defaults but really should be tuned based on what counts as acceptable planting error for the specific setup. We did not have enough calibrated ground truth to set these precisely during the internship.
- Row identity is declared by folder structure, not inferred by the algorithm. When collecting photos, walk one row at a time and keep the photos for each row in their own folder. Section 7 in the notebook explains the expected layout.
- The model was trained on a relatively small dataset. We planted 250 seeds but only 76 sprouted due to soil-borne corn rot, which was a setback we had to work around. More data across different weather conditions, growth stages, and corn varieties would improve generalizability.
- Pixel-to-centimeter conversion is not implemented. It requires a calibrated camera height. Once you have that number, add a PX_PER_CM constant and multiply wherever needed.
- The custom annotated dataset is not publicly released since it was collected as part of the VexTronics internship. The Brilhador et al. dataset is publicly available from the original authors.

---

## Contact

For questions about the project, the report in report/ is the best starting point. For anything beyond that, reach out through VexTronics via ir. E.A.J. Vermeer.

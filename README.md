# Visual Stress Detection System

## Team
Ritik Mishra 23BCE10850, Anshika Shrivastava 23BAI10033 , Yash Pratap Singh 23BAI10012 , Amritasha Gupta 23BAI10048 , Sreyas Sunmesh 23BAI10062, Manasvi Joshi 23BHI10193

## Purpose
This project is an advanced **Intelligent Vision System** that detects micro-deformations and predicts structural risk. It utilizes **Optical Flow (RAFT)** for tracking precise dense displacements across sequences, applies **Physics-Based Modeling** to compute strain and Von Mises stress mappings, and feeds the extracted features into a **Machine Learning Classifier** to assess structural failure probabilities.

## Features
- **Dense Optical Flow (RAFT):** Calculates precise micro-displacements from image sequences.
- **Physics Modeling:** Converts raw displacement data into strain and Von Mises Stress.
- **Machine Learning Integration:** Extracts temporal features and predicts failure risks.
- **Interactive Dashboard:** A Streamlit-based web app to tweak parameters and observe outcomes on-the-fly.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd cvgroupproject
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the interactive Streamlit dashboard using the following command:

```bash
streamlit run app.py
```

1. Open the provided local URL in your browser.
2. In the **Configuration** sidebar, set the paths to your local image dataset (`data` directory by default) and adjust physical constants (Young's Modulus and Poisson's ratio).
3. Click on **Run Full Pipeline** to process the sequence and view the generated displacement maps, heatmaps, and failure risk percentages.

## Reproducibility Instructions

This file provides detailed instructions to reproduce the experiments, from data generation to final analysis. The project's workflow is organized as a sequence of Jupyter notebooks.


### Prerequisites
Before beginning, ensure you have the following:

- **Git**: For cloning the project repository.  
- **Python 3.9** or higher.  
- **Jupyter Notebook or JupyterLab**: To run the `.ipynb` files.  
- **An NVIDIA GPU** with at least 16GB of VRAM and CUDA installed to run the Mistral-7B model with 4-bit quantization.  
- **API Keys** for the following services:  
  - Foursquare Places API (for initial data sourcing).  
  - Groq API (for data cleaning, label generation, and LLM-as-a-Judge evaluation).  

---

### Environment Setup

1. **Clone the Repository:**
    The repository is on my Github account: https://github.com/FaidYoussef/FamilyWall-Homework#
   ```bash
   git clone git@github.com:FaidYoussef/FamilyWall-Homework.git
   cd FamilyWall-Homework
    ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies:** Install all required Python packages, including Jupyter, from the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   Create a file named `.env` in the root directory of the project and add your API keys in the following format:

   ```bash
   FSQ_API_KEY="your_foursquare_api_key"
   GROQ_API_KEY="your_groq_api_key_for_mistral_testing"
   MISTRAL_MODEL_PATH="your_mistral_model_path"
   ```

---

### Execution Pipeline via Jupyter Notebooks

The core logic of this project is contained within the Jupyter notebooks located in the `notebooks/` directory. They are numbered and should be executed in sequential order by running all cells in each notebook. The `src/` directory contains helper Python modules (`.py` files) that are imported by these notebooks and are not intended to be run directly.

1. **Notebook 1: `1_dataset_generation.ipynb`**

   * **Purpose:** Handles the entire data creation pipeline, including sourcing data from Foursquare, scraping websites, cleaning the data with an LLM filter, and generating the final labeled dataset with Gemma.
   * **Key Outputs:** `data/all_businesses_descriptions_and_domains.json`

2. **Notebook 2: `2_mistral_testing.ipynb`**

   * **Purpose:** Loads the 4-bit quantized Mistral-7B model and runs zero-shot inference on the dataset created in the previous step.
   * **Note:** This notebook requires a local copy of the Mistral-7B-v0.1 model from Hugging Face. The first run will download the model files, which requires significant disk space.
   * **Key Outputs:** `data/parsed_mistral.json`

3. **Notebook 3: `3_mistral_evaluation.ipynb`**

   * **Purpose:** Implements the LLM-as-a-Judge framework. It systematically evaluates the outputs from both Gemma (from Notebook 1) and Mistral (from Notebook 2) against the defined criteria.
   * **Key Outputs:** `data/gemma_evaluations.json` and `data/parsed_mistral_evaluations.json`

4. **Notebook 4: `4_models_comparison.ipynb`**

   * **Purpose:** Performs the final comparative analysis using the evaluation data from the previous step. It calculates all statistics and generates the visualizations presented in this report.
   * **Key Outputs:** All plot images saved to the `figures/` directory.

5. **Notebook 5: `5_edge_cases_discovery.ipynb`**

   * **Purpose:** Contains the logic for the proactive edge case discovery. It first generates the five challenging test sets using an LLM, then processes each test case with both Mistral and Gemma to gather their responses.
   * **Key Outputs:** Creates and populates the `data/edge_case_test_sets/` directory with five JSON files containing the test descriptions and the models' outputs.


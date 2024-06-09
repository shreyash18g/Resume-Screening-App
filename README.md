# Resume Screening App

## Description

This project focuses on automating the process of resume evaluation and categorization. The app offers three primary functionalities:

- **Domain Prediction**: Classifies resumes into various domains such as HR, IT, Finance, etc., using machine learning.
- **Job Prediction**: Predicts the most suitable job role for the resume using machine learning.
- **Personal Information Extraction**: Extracts personal details such as name, email, and phone number using basic Python techniques.

## Datasets

### Job Predictor Dataset

The dataset used for job prediction consists of 1,615,940 entries with two columns: `Role` and `Features`.

Example:

| Role                     | Features                                            |
|--------------------------|-----------------------------------------------------|
| Social Media Manager     | 5 to 15 Years Digital Marketing Specialist M.Tech...|
| Frontend Web Developer   | 2 to 12 Years Web Developer BCA HTML, CSS, JavaScript|
| Quality Control Manager  | 0 to 12 Years Operations Manager PhD Quality control|
| Wireless Network Engineer| 4 to 11 Years Network Engineer PhD Wireless networks|
| Conference Manager       | 1 to 12 Years Event Manager MBA Event planning      |

### Domain Predictor Dataset

The dataset used for domain prediction consists of 2,484 entries with three columns: `ID`, `Category`, and `Feature`.

Example:

| ID        | Category | Feature                                                |
|-----------|----------|--------------------------------------------------------|
| 16852973  | HR       | hr administrator marketing associate hr administration |
| 22323967  | HR       | hr specialist hr operations summary media professional |
| 33176873  | HR       | hr director summary years experience recruiting         |
| 27018550  | HR       | hr specialist summary dedicated driven dynamic          |
| 17812897  | HR       | hr manager skill highlights hr skills hr department    |

The categories include:

- INFORMATION-TECHNOLOGY
- BUSINESS-DEVELOPMENT
- FINANCE
- ADVOCATE
- ACCOUNTANT
- ENGINEERING
- CHEF
- AVIATION
- FITNESS
- SALES
- BANKING
- HEALTHCARE
- CONSULTANT
- CONSTRUCTION
- PUBLIC-RELATIONS
- HR
- DESIGNER
- ARTS
- TEACHER
- APPAREL
- DIGITAL-MEDIA
- AGRICULTURE
- AUTOMOBILE
- BPO

## Included Files

### Models

- **models/domain_prediction_model.pkl**: Pre-trained model for domain prediction.
- **models/job_prediction_model.pkl**: Pre-trained model for job prediction.

### Code Files

- **app.py**: Main Flask application file.
- **requirements.txt**: List of dependencies required to run the application.

### Templates

- **templates/index.html**: HTML template for the web interface.

### Static Files

- **static/styles.css**: CSS file for styling the web interface.

### Data Files

- **data/sample_resume.txt**: Sample resume text file for testing.

## Usage

1. Clone this repository:

    ```bash
    git clone https://github.com/shreyash18g/Resume-Screening-App.git
    cd Resume-Screening-App
    ```

2. Set up a virtual environment:

    ```bash
    python3 -m venv venv
    source venv\Scripts\activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:

    ```bash
    flask run
    ```

5. Open your browser and navigate to:

    ```
    http://127.0.0.1:5000
    ```

## Data Exploration

### Metadata Analysis

Perform metadata analysis to understand the distribution of job roles, domain categories, and other relevant information.

## Feature Extraction

Feature extraction is a crucial step in preparing data for model training. Common features for text classification include TF-IDF, word embeddings, and n-grams. Extracting these features will help in building robust classification models.

## Model Training

### Develop and Train Classification Models

Develop classification models using popular machine learning frameworks such as Scikit-learn, TensorFlow, or PyTorch. Experiment with different architectures and hyperparameters to find the best model for your dataset.

## Evaluate Model Performance

Once the models are trained, evaluate their performance using appropriate metrics such as accuracy, precision, recall, and F1-score. Additionally, consider using techniques like cross-validation to ensure the reliability of your results.

## Contributing

Contributions to this project are welcome! Whether you want to suggest improvements, report issues, or contribute code, feel free to submit pull requests or open issues on the project's GitHub repository.

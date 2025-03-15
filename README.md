# LoanPulse: AI-Powered Loan Approval System

## Project Overview
LoanPulse is a robust, end-to-end machine learning pipeline that automates the loan approval process. Using advanced data processing, feature engineering, and model evaluation techniques, LoanPulse ensures accurate predictions for loan eligibility. The project is designed with MLOps best practices, including CI/CD, cloud storage, and model deployment.

---

## Project Workflow

### 1. Project Setup
- Create the project template by executing `template.py`
- Modify `setup.py` and `pyproject.toml` to import local packages (Refer to `crashcourse.txt` for more details)
- Create and activate a virtual environment, then install dependencies:
  ```bash
  conda create -n loanpulse python=3.10 -y
  conda activate loanpulse
  pip install -r requirements.txt
  ```
- Verify package installation with `pip list`

---

## MongoDB Setup
1. Sign up on MongoDB Atlas and create a new project.
2. Create a cluster using M0 service and set up a database user.
3. Configure network access with IP `0.0.0.0/0`.
4. Obtain the MongoDB connection string (`Python, Version: 3.6 or later`).
5. Store data in MongoDB by running `mongoDB_demo.ipynb`.
6. Verify the stored data in MongoDB Atlas.

---

## Logging, Exception Handling, and Notebooks
- Implement logging and exception handling in `logger.py` and `exception.py`, then test in `demo.py`.
- Conduct exploratory data analysis (EDA) and feature engineering.

---

## Data Ingestion
1. Define variables in `constants.__init__.py`.
2. Implement MongoDB connection in `configuration.mongo_db_connections.py`.
3. Fetch and transform data in `data_access/proj1_data.py`.
4. Define `DataIngestionConfig` and `DataIngestionArtifact` classes.
5. Implement the ingestion process in `components/data_ingestion.py`.
6. Test ingestion by running `demo.py`.
7. Set up MongoDB connection URL:
   ```bash
   export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net"
   ```

---

## Data Validation, Transformation & Model Training
- Define dataset schema in `config.schema.yaml`.
- Implement `Data Validation`, `Data Transformation`, and `Model Training` components.
- Add `estimator.py` to the entity folder.

---

## AWS Setup for Model Evaluation & Deployment
1. Create IAM user with `AdministratorAccess`.
2. Generate access keys and set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   ```
3. Configure `aws_connection.py` for S3 interaction.
4. Define constants for model storage.
5. Create an S3 bucket `loanpulse-models`.
6. Implement `s3_estimator.py` for cloud model storage.

---

## Model Evaluation & Deployment
- Implement `Model Evaluation` and `Model Pusher` components.
- Develop a prediction pipeline and `app.py`.
- Add `static` and `template` directories for UI support.

---

## CI/CD Pipeline Setup
1. Create `Dockerfile` and `.dockerignore`.
2. Configure GitHub Actions (`.github/workflows/aws.yaml`).
3. Create an ECR repository in AWS (`loanpulse-repo`).
4. Launch an EC2 instance (`loanpulse-machine`).
5. Install Docker on EC2 and configure a self-hosted GitHub runner.
6. Add GitHub secrets for AWS credentials and repository settings.
7. Enable port 5080 for web access:
   ```bash
   sudo ufw allow 5080
   ```
8. Deploy the model and launch the app via EC2’s public IP + `:5080`.

---

## Usage
- Train the model using `/training` route.
- Make predictions via the `/predict` endpoint.
- Access logs and monitor the pipeline using integrated logging.

---

## Conclusion
LoanPulse is a scalable and automated solution for loan approval predictions. With its robust MLOps pipeline, cloud integration, and CI/CD setup, it ensures a seamless ML lifecycle from development to deployment.

## License
This project is licensed under [MIT License].


FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ./

# Upgrade pip to ensure it fetches pre-compiled binaries
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your ML models and your inference script
COPY best_contrastive_arty7_64D_SGD.keras ./
COPY best_svm_model.pkl ./
COPY app.py ./

CMD ["app.lambda_handler"]
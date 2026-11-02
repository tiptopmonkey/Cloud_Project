import base64
import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import boto3
from botocore.client import Config
import cv2
from fpdf import FPDF
import joblib
import keras
import numpy as np
from sklearn.preprocessing import Normalizer

# ---------------- AWS CONFIGURATION ----------------
REGION = "eu-north-1"
S3_BUCKET = "vit-ultrasound-vault-895492487598"
DYNAMODB_TABLE = "UltrasoundPatientRecords"
SNS_TOPIC_ARN = "arn:aws:sns:eu-north-1:895492487598:UltrasoundDoctorAlerts"

# Initialize AWS Clients (Signature v4 is required for S3 Pre-Signed URLs in eu-north-1)
s3_client = boto3.client("s3", region_name=REGION, config=Config(signature_version="s3v4"))
dynamodb = boto3.resource("dynamodb", region_name=REGION)
patient_table = dynamodb.Table(DYNAMODB_TABLE)
sns_client = boto3.client("sns", region_name=REGION)

# ---------------- LOAD ML MODELS ----------------
feature_extractor = keras.models.load_model("best_contrastive_arty7_64D_SGD.keras", compile=False)
svm_classifier = joblib.load("best_svm_model.pkl")
scaler = Normalizer(norm="l2")


def preprocess_ultrasound(image_bytes):
    # Decode incoming base64 image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 1. Global Text Eraser
    _, text_mask = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    img_cleaned = cv2.inpaint(img, text_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # 2. Hardware Alignment (Resize, EqHist, Bilateral)
    resized = cv2.resize(img_cleaned, (128, 128), interpolation=cv2.INTER_AREA)
    img_eq = cv2.equalizeHist(resized)
    img_bf = cv2.bilateralFilter(img_eq, 9, 75, 75)

    # 3. Format FP16 Tensor for Keras
    normalized = np.expand_dims(img_bf.astype(np.float16) / 255.0, axis=(0, -1))
    return normalized, img, img_bf


def generate_pdf_report(scan_id, patient_id, patient_name, doctor_name, timestamp, diagnosis, confidence, probs, raw_path, filtered_path):
    pdf_path = f"/tmp/{scan_id}.pdf"
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "CLOUD AI ULTRASOUND DIAGNOSTIC REPORT", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Report ID: {scan_id}  |  Generated (IST): {timestamp}", ln=True, align="C")
    pdf.ln(6)

    # Patient & Referring Clinician Info
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Patient & Clinical Metadata", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Patient ID: {patient_id}    |    Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 7, f"Referring Specialist: {doctor_name}", ln=True)
    pdf.ln(4)

    # AI Diagnostic Output
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Serverless ML Classification Result", ln=True)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 9, f"Primary Diagnosis: {diagnosis.upper()} ({confidence}% Confidence)", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Probability Breakdown -> Benign: {probs['benign']}%  |  Malignant: {probs['malignant']}%", ln=True)
    pdf.ln(6)

    # Ultrasound Visual Comparison (Raw vs OpenCV Preprocessed)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Ultrasound Scan Analysis", ln=True)
    y_pos = pdf.get_y() + 4
    pdf.image(raw_path, x=20, y=y_pos, w=75, h=75)
    pdf.image(filtered_path, x=115, y=y_pos, w=75, h=75)
    pdf.set_y(y_pos + 78)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(95, 6, "Original Uploaded Scan", align="C")
    pdf.cell(95, 6, "AI Preprocessed (Text Erased + Bilateral Filter)", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, "Confidentiality Notice: This report is encrypted and stored in Amazon S3 (SSE-S3). Access is granted via temporary 24-hour cryptographic S3 Pre-Signed URL.")

    pdf.output(pdf_path)
    return pdf_path


def lambda_handler(event, context):
    cors_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS"
    }

    # 1. Handle Browser Preflight OPTIONS Request
    http_method = event.get("requestContext", {}).get("http", {}).get("method", "")
    if http_method == "OPTIONS" or event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": cors_headers, "body": ""}

    try:
        body = json.loads(event.get("body", "{}")) if event.get("body") else {}
        action = body.get("action", "analyze")

        # 2. ROUTE A: Fetch Patient History from DynamoDB
        if action == "history" or http_method == "GET":
            response = patient_table.scan(Limit=15)
            items = response.get("Items", [])
            items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return {
                "statusCode": 200,
                "headers": cors_headers,
                "body": json.dumps({"records": items[:10]}, default=str)
            }

        # 3. ROUTE B: Analyze Scan + S3 + DynamoDB + SNS Doctor Alert
        image_data = base64.b64decode(body["image"])
        patient_id = body.get("patient_id", "PT-UNKNOWN")
        patient_name = body.get("patient_name", "Anonymous Patient")
        doctor_name = body.get("doctor_name", "On-Call Radiologist")

        # Generate IST Timestamp & Unique Scan ID
        ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
        timestamp_str = ist_now.strftime("%Y-%m-%d %H:%M:%S")
        scan_id = f"SCAN-{ist_now.strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"

        # Run ML Pipeline
        tensor, raw_img, filtered_img = preprocess_ultrasound(image_data)
        features_64d = feature_extractor.predict(tensor, verbose=0)
        scaled_features = scaler.transform(features_64d)

        prediction = int(svm_classifier.predict(scaled_features)[0])
        probabilities = svm_classifier.predict_proba(scaled_features)[0].tolist()

        diagnosis = "Malignant" if prediction == 1 else "Benign"
        confidence = round(probabilities[prediction] * 100, 2)
        probs_dict = {
            "benign": round(probabilities[0] * 100, 2),
            "malignant": round(probabilities[1] * 100, 2)
        }

        # Save images to /tmp/ (AWS Lambda writable directory)
        raw_path = f"/tmp/{scan_id}_raw.png"
        filtered_path = f"/tmp/{scan_id}_filtered.png"
        cv2.imwrite(raw_path, raw_img)
        cv2.imwrite(filtered_path, filtered_img)

        # Generate PDF Report in /tmp/
        pdf_path = generate_pdf_report(
            scan_id, patient_id, patient_name, doctor_name,
            timestamp_str, diagnosis, confidence, probs_dict,
            raw_path, filtered_path
        )

        # Upload Raw Image & PDF Report to Encrypted Amazon S3
        s3_pdf_key = f"reports/{scan_id}.pdf"
        s3_img_key = f"scans/{scan_id}_raw.png"
        s3_client.upload_file(pdf_path, S3_BUCKET, s3_pdf_key, ExtraArgs={"ContentType": "application/pdf"})
        s3_client.upload_file(raw_path, S3_BUCKET, s3_img_key, ExtraArgs={"ContentType": "image/png"})

        # Generate 24-Hour S3 Pre-Signed URL for Doctor Access
        presigned_pdf_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": s3_pdf_key},
            ExpiresIn=86400
        )

        # Log Telemetry Record into Amazon DynamoDB
        db_item = {
            "scan_id": scan_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "doctor_name": doctor_name,
            "timestamp": timestamp_str,
            "diagnosis": diagnosis,
            "confidence": Decimal(str(confidence)),
            "s3_report_key": s3_pdf_key,
            "report_url": presigned_pdf_url
        }
        patient_table.put_item(Item=db_item)

        # Dispatch Automated Email Alert to Doctor via Amazon SNS
        email_subject = f"[{diagnosis.upper()} ALERT] Ultrasound Report: {patient_name} ({patient_id})"
        email_message = (
            f"Automated Clinical Diagnostic Notification\n"
            f"------------------------------------------\n"
            f"Scan ID: {scan_id}\n"
            f"Patient: {patient_name} (ID: {patient_id})\n"
            f"Referring Specialist: {doctor_name}\n"
            f"Timestamp (IST): {timestamp_str}\n\n"
            f"AI CLASSIFICATION: {diagnosis.upper()}\n"
            f"Model Confidence: {confidence}%\n"
            f"Probabilities: Benign ({probs_dict['benign']}%) | Malignant ({probs_dict['malignant']}%)\n\n"
            f"SECURE HEALTH DATA SHARING (24-Hour Encrypted S3 Pre-Signed Link):\n"
            f"Click below to view/download the full PDF Diagnostic Report & Scans:\n"
            f"{presigned_pdf_url}\n"
        )
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=email_subject[:100],
            Message=email_message
        )

        # Clean up /tmp/ files
        for path in [raw_path, filtered_path, pdf_path]:
            if os.path.exists(path):
                os.remove(path)

        return {
            "statusCode": 200,
            "headers": cors_headers,
            "body": json.dumps({
                "scan_id": scan_id,
                "timestamp": timestamp_str,
                "diagnosis": diagnosis,
                "confidence": confidence,
                "probabilities": probs_dict,
                "report_url": presigned_pdf_url,
                "sns_notified": True
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": cors_headers,
            "body": json.dumps({"error": str(e)})
        }
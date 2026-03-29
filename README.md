# Network Traffic Classification using CNN

This project classifies **encrypted network traffic** using **Convolutional Neural Networks (CNNs)** on **image-based flow features**.

Instead of inspecting packet payloads, the system uses **network flow statistics** and converts them into **8×8 grayscale images** for classification.

## Project Goal
- Detect anonymized traffic such as:
  - Non-Tor
  - Non-VPN
  - Tor
  - VPN
- Classify darknet traffic into application categories:
  - Browsing
  - Chat
  - Email
  - File Transfer
  - Audio Streaming
  - Video Streaming
  - VoIP
  - P2P

## Dataset
This project uses traffic data derived from:
- **ISCXVPN2016**
- **ISCXTor2017**

## Methodology
1. Data preprocessing and cleaning  
2. Feature selection using **ExtraTreesClassifier**  
3. Conversion of selected features into **8×8 grayscale images**  
4. Classification using a **CNN model**

## Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras



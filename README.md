# EdgeGaming_Latency_Cloud_Project_2026
# Project Title
Machine Learning-Based Network Latency Optimization Framework for Online Gaming Platforms using Edge Intelligence

---

### Team Members
*   **Atharva** - AI & Data Engineering 
*   **Dhyaan** - AWS Cloud Infrastructure 
*   **Sangeet** - Architecture & Backend Integration 

---

### Problem Statement
In fast-paced multiplayer architectures, unpredictable network jitter, kernel-level packet processing overhead, and burst packet loss cause severe latency spikes and visual desynchronization (rubber-banding). Traditional client-side dead reckoning algorithms fail entirely during sustained packet loss. Furthermore, standard operating system network stacks treat all incoming game data identically, meaning critical combat inputs get stuck in the same queue as non-critical cosmetic updates, adding unnecessary processing delay.

---

### Objectives
1.  **Deploy Edge Infrastructure:** Develop an ultra-low latency edge gaming framework by deploying game servers on AWS          Wavelength zones.
2.  **Optimize Network Routing:** Reduce kernel-level packet queuing delay to microseconds by implementing an Intent-Aware       QoS scheduler using eBPF/XDP.
3.  **Enhance State Prediction:** Improve client-side state prediction accuracy during network jitter by replacing standard      dead-reckoning with a machine learning model.
4.  **Secure Data Pipeline:** Store and process historical player spatial-temporal telemetry securely using Amazon S3 and        AWS IAM.
5.  **Implement Live Monitoring:** Monitor edge-to-client network health, track packet loss, and visualize latency metrics       in real-time using Amazon CloudWatch dashboards.

---

### Proposed Architecture/Framework
The proposed Edge Intelligence framework operates on a three-tier architecture to eliminate multiplayer desynchronization:
*   **The Physical Shortcut (AWS Wavelength):** Game server containers are hosted directly inside local 5G telecom towers,       vastly reducing the physical distance data must travel between the player and the server.
*   **The VIP Express Lane (eBPF / Kernel Bypass):** An eBPF program sits at the server's hardware level to scan incoming        UDP packets. High-priority packets bypass the standard Linux kernel queue and are routed instantly to the game engine.
*   **The AI Safety Net (Mamba State-Space Model):** A lightweight State-Space Model (SSM) runs on the edge node. When a         player's connection drops, the model analyzes their past trajectory and synthesizes the missing movement frames, keeping     the game smooth until the connection returns.

---

### Technology Stack
*   **AWS Cloud Services:** AWS Wavelength, Amazon EC2, Amazon S3, Amazon SageMaker, Amazon CloudWatch, AWS IAM, Amazon API      Gateway.
*   **Networking & OS:** eBPF, XDP, C, UDP Sockets, Linux Kernel optimizations.
*   **Machine Learning:** PyTorch, Mamba-SSM (State-Space Models), Pandas, NumPy.
*   **Frontend/Backend Logic:** Python, HTML/JS (for latency visualization dashboards).

---

### Dataset Details
*   **Dataset Name:** CS:GO Player Movement Telemetry Dataset
*   **Source:** Kaggle (Open-Source)
*   **Dataset Size:** ~2.5 GB
*   **Data Type:** CSV (Numerical and Categorical)
*   **Features:** Timestamp, Player ID, 3D Coordinates (X, Y, Z), Camera Angles (Pitch, Yaw), Action State (Move, Shoot).
*   **Purpose:** To train the Amazon SageMaker Mamba ML model on real human movement sequences so it can accurately predict      player trajectories during network packet loss.

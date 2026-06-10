# I. Project Overview
This experiment aims to abandon traditional heuristic rules (rule-based) and construct an end-to-end Multimodal Foundation Model (Dark-MFM). By aligning and fusing X-band maritime radar images (visual modality) with AIS sequential data (temporal modality), the system automatically learns the topology of normal maritime behavior to achieve high-precision, real-time detection of dark vessels. 
# II. Experimental Content and Steps 
## Experiment 1: Multimodal Data Representation
* **Objective:** To transform heterogeneous radar images and raw tabular AIS data into high-dimensional continuous vectors (embeddings) that can be processed by neural networks. 
* **Specific Operations:** 
	1. **Visual Feature Extraction:** Input continuous radar scan frames (e.g., 3000 x 3000 pixel trajectory mapping images) into a Vision Transformer (ViT) network to extract the spatial and dynamic feature vectors of radar trajectories. 
	2. **AIS Dual-Path Representation (Hybrid Approach):** Process tabular AIS data through two parallel pathways to capture both macro-level semantics and micro-level kinematics: 
		* **Path A (Semantic - Text Embedding):** Utilize template-based serialization (bypassing LLM generation) to efficiently format data using Python scripts (e.g., `f"Ship {mmsi} at {lat},{lon} moving {sog} knots."`). Feed these strings into a lightweight Text Embedding model to capture macro behavioral logic and reasoning capabilities. 
		* **Path B (Geometric - MLP + Time2Vec):** Standardize raw numerical values (Latitude, Longitude, SOG, COG) and project them into a high-dimensional space using a Multi-Layer Perceptron (MLP). Apply Time2Vec technology to generate continuous time encodings, preserving precise temporal, spatial, and kinematic features.
	3. **Feature Concatenation:** Concatenate the semantic vector (Path A) and the geometric/temporal vector (Path B). This combined embedding is then processed to serve as the final AIS representation for downstream Mid-Fusion with the radar visual features, successfully achieving both semantic explanatory power and mathematical accuracy.
## Experiment 2: CLIP-Style Contrastive Pre-training 
* **Objective:** To align radar features and AIS features in the latent space to establish geometric and semantic baselines for determining anomalies. 
* **Specific Operations:** 
	1. **Construct Paired Data:** Clean and extract a large amount of "normally sailing" and well-matched Radar-AIS data as positive samples. 
	2. **Dual-Tower Feature Alignment:** Use a contrastive loss function for pre-training, forcing the neural network to maximize the cosine similarity between the "radar feature vector" and "AIS feature vector" of the same vessel in the latent space. 
	3. **Unsupervised Dark Vessel Filtering:** During inference, if the entity features extracted by the radar have a cosine similarity lower than a safety threshold with all surrounding AIS features, the system will instantly flag it as a high-confidence "dark vessel candidate." 
## Experiment 3: Mid-Fusion Cross-Modal Interaction and Inference 
* **Objective:** To capture microscopic behavioral inconsistencies that single modalities or late fusion fail to identify, further improving detection sensitivity and accuracy. 
* **Specific Operations:** 
	1. **Mid-Level Feature Fusion:** Jointly input the radar spatial features and AIS temporal features processed in Experiments 1 and 2 into a Multi-Head Cross-Attention module. 
	2. **Semantic Conflict Detection:** Utilize the fusion layer to allow deep interaction between the two modalities. For example, learning to detect potential contradictions between minor irregular distortions in radar trajectories and the smooth headings broadcasted by AIS. 
	3. **End-to-End Classifier:** Input the fused high-dimensional features into a fully connected layer to output the final prediction probability (binary classification: normal vessel vs. dark vessel). 
--- 
# III. Expected Outcomes and Evaluation Metrics 
* **Expected Deliverables:** A fully trained Dark-MFM model weight and inference pipeline, possessing the capability for dark vessel identification with zero manual rule intervention. 
* **Core Evaluation Metrics:** 
	* **Detection Accuracy and Recall:** Ensure the miss rate of dark vessels is minimized. 
	* **Feature Alignment Error:** Evaluate the degree of separation between positive and negative samples during the contrastive learning pre-training phase. 
	* **Inference Latency:** Verify the computational efficiency of the model in real-time maritime monitoring scenarios.
# Developing an Image Classifier Using Synthetic Data from CAD Models
**Alessandro Morato**

**BH-PCMLAI, January 2024, Final Capstone Project**

<div align="justify">

## 1. Introduction

> <i style="color:blue;">Is it possible to develop an effective image recognizer using a collection of 'synthetic' images produced by CAD software?</i>

I work as an Electronic Engineer at a US National Laboratory (Lawrence Berkeley National Laboratory), on a world-class synchrotron particle accelerator called ALS (Advanced Light Source). This highly complex system consists of numerous custom-designed components, making it unique and requiring experienced personnel to operate safely. A tool capable of recognizing every single component in real-time, providing necessary information to operators, would be invaluable. However, training such a device would require an immense quantity of field-collected images.

To address this, I explored the possibility of using 3D models designed by CAD software to generate synthetic training data. Being an one-of-a-kind complex, every single main component is custom designed and a 3D model is generally available. This approach could be particularly appealing for many research and industrial facilities where custom-designed components are prevalent. This project applied these concepts to a specific part of the accelerator, the magnets of the ring, within my field of expertise (Power and Industrial Electronics).

## 2. Problem Understanding

The primary objective of this work is to develop a preliminary model for an image recognition system that can identify components within the accelerator, in particular, a family of magnets from one of the 12 sectors of the newly installed ALS Accumulator Ring (AR). Since SD-SF and SHD-SHF are identical, only SD and SHD are included in this analysis. The BEND magnet has been excluded as well, due to logistical difficulties in obtaining a sample for photo acquisition. The magnets investigated are: BEND, QF, QD, QFA, SD, and SHD.

![image](https://github.com/user-attachments/assets/cf4828aa-01a9-4d17-8ab8-5b92ba1f55c1)

Given the complexity and uniqueness of the system, a significant amount of training data would be needed to develop a robust image detector. The main challenges include:
- The need for a large volume of training images.
- The difficulty in collecting and annotating real-world images.
- Ensuring the classifier's robustness to varying real-world conditions.

This project aims to determine if synthetic images generated from CAD models can be used effectively for this purpose, potentially reducing the dependency on large volumes of real-world data. The scope of this work is to provide an initial insight into this topic, which is gaining growing interest in the industrial community. The project presented here is intended to be the first part of a broader effort that will be developed in the coming months.

## 3. The Need for Supercomputing
This analysis required several training sessions and attempts. The datasets for training the classifier models comprised hundreds, if not thousands, of images, which required substantial computational power. Training on such large datasets using standard commercial hardware would have taken weeks, making it impractical for timely development and iteration. Therefore, a more powerful computational platform was necessary to handle the workload efficiently. Google and AWS both offer affordable and well-reputed services that share some nodes of their servers and provide the required computational power. However, working at Lawrence Berkeley Laboratory gives me access to its state-of-the-art supercomputer, NERSC.

### 3.1 NERSC Perlmutter Supercomputer
The National Energy Research Scientific Computing Center (NERSC) operates the Perlmutter supercomputer, which is specifically designed to handle large-scale AI workloads and scientific simulations. Perlmutter's hybrid architecture, combining CPUs and NVIDIA A100 Tensor Core GPUs (A100-SXM4-80GB, 81053MiB), provides the computational power needed for deep learning tasks, enabling rapid training of complex classifier models. For this analysis, a node consisting of 4 distinct GPUs was allocated. In the first stage of the project, the easiest way to provide instructions to Perlmutter was through prompt commands; that is why I initially found it more convenient to structure my work in different .py files, depending on the particular sub-tasks, rather than in a single Jupyter file. After an initial exploratory phase and with a better understanding of the topic, I developed comprehensive Jupyter notebooks where I collected the main points of the work and refined the analysis.

## 4. Data Collection 1: CAD Images

The dataset necessary for this work requires (i) a vast collection of synthetic images generated from CAD models and (ii) a 'limited' (though generous) number of real-world pictures collected in the field.

### 4.1 From STEP to STL Files

3D models in the STEP format (.stp) have been converted into a series of 2D .png images. The conversion process has involved multiple steps, including (i) converting the .stp files to the STL format (.stl) and then (ii) generating images from the .stl files, through a series of automated Python scripts that leverage tools like FreeCAD and PyVista.

STEP (Standard for the Exchange of Product Data) is a widely used file format for representing 3D CAD models. It is standardized as ISO 10303 and is used to transfer 3D data between different CAD programs. STEP files have a .stp (or .step) file extension and contain detailed information about the geometry, topology, material properties, and assembly structures of a 3D model. The needed .stp files have been provided to me by ALS Mechanical Engineer group, which designed the real magnets, and they have been exported from the original projects files. This means that the correspondence between them and the real magnets is expected to be quite close.

Each .stp file is converted to the STL format. STL (Stereolithography) files describe only the surface geometry of a three-dimensional object without any representation of color, texture, or other attributes. The file format uses a series of linked triangles (polygons) to represent the surface of the 3D object, which is often referred to as a mesh. STL files are widely used for 3D printing, computer-aided design, and rapid prototyping and are the preferred choice for many Python libraries, since they are simpler and easier to manipulate compared to the original STEP files.

### 4.2 Images Generation with Open-Source Python Libraries

The graphic libraries chosen to manipulate the STL files are provided by FreeCAD, an open-source parametric 3D CAD modeler, and PyVista, an open-source Python library for 3D computer graphics. The choice was mainly driven by the fact that they are open-source tools, with strong support from a helpful community, which has been quite beneficial during development. However, developing reliable scripts has not been straightforward and has required a considerable amount of time and effort.

PyVista, in particular, features a powerful tool called 'plotter,' used to manage and display 3D visualizations. Various camera parameters, such as position, angle, and focal length, are available to capture images from different perspectives. This variability is essential for creating a dataset that covers multiple views of the object. The script renders the 3D model as a 2D image from the specified camera angles. Multiple images are rendered by varying the camera position and angle to simulate different viewpoints. An assumption was made at this point: an operator will always observe the magnets from the sides but never from the bottom or top (due to the base where the magnet is installed and the low ceiling of the accelerator's bunker). To address this, it was decided to focus on all the lateral views of the magnets (azimuth 0-360 degrees), moving ±40 degrees in altitude. To emulate imperfect alignment of the operator's camera, a ±15 degree tilt was also applied. The chosen rotation step was 15 degrees, resulting in a total of over 4,000 images for the 5 magnets investigated.

The 3D libraries in use provide advanced tools that can apply different backgrounds and lighting conditions to the scene. This step is certainly important for making the dataset more realistic and robust, as it simulates the variability found in real-world conditions. However, due to the complexity of implementation, it was skipped at this stage. All the generated images present similar lighting conditions and a plain white background.

One issue encountered while working with STL files is the management of colors. The only information stored in STL files is essentially the surface geometry. There are other formats that can store colors (e.g., OBJ), but their processing is not supported by FreeCAD and PyVista. Therefore, the decision was made to neglect color information and work with grayscale models.

The rendered black-and-white images are saved as .png files in the designated output directory (CAD_pics), ready for use in training the classification model.

### 4.3 Labeling of Synthetic Images

Labeling is a critical step in preparing data for training an image classifier model, particularly when using YOLO (You Only Look Once) models. Accurate labeling is essential to ensure that the model learns to identify and localize objects within images effectively.

YOLO models are designed to detect and classify objects within an image in real time, making them highly dependent on precise labeling. Each labeled object in an image provides the model with the information it needs to learn how to detect and localize similar objects in new, unseen images. Inaccurate or inconsistent labeling can lead to poor model performance, resulting in missed detections, false positives, or incorrect classifications. YOLO models require labels to be in a specific format that describes the objects within each image. Each image's label is stored in a separate text file with the same name as the image file but with a .txt extension. The label file contains one line per object, with each line following this structure:

`<class_id> <x_center> <y_center> <width> <height>`

- **class_id**: An integer representing the class of the object. This ID corresponds to a class listed in a separate file (usually classes.txt), where each class has a unique ID starting from 0.
- **x_center**: The x-coordinate of the object's center, normalized to the image width. The value ranges from 0 to 1, where 0 is the leftmost edge of the image, and 1 is the rightmost edge.
- **y_center**: The y-coordinate of the object's center, normalized to the image height. The value ranges from 0 to 1, where 0 is the topmost edge of the image, and 1 is the bottommost edge.
- **width**: The width of the object, normalized to the image width. The value ranges from 0 to 1, representing the object's width relative to the image width.
- **height**: The height of the object, normalized to the image height. The value ranges from 0 to 1, representing the object's height relative to the image height.

### 4.4 Reference Jupyter Notebook and Samples of Images Generated

- XXXXXXXX
- YYYYYYYY

![image](https://github.com/user-attachments/assets/1a54080c-b700-4a1f-84b4-7c2ac4803921)

## 5. Data Collection 2: Real-World Images

Each magnet was placed in an isolated position, without any other objects obstructing the view. A 4K 60Hz video was recorded for each magnet, capturing a 360-degree view by walking around the object. I then used the online tool 'Roboflow' to extract individual frames from these videos and prepare a labeled dataset ([LINK](https://roboflow.com/)). This process allowed me to efficiently generate a large, diverse set of images with the corresponding labels.

I discovered Roboflow after conducting online research on similar tools. Roboflow is an online platform designed to help developers and data scientists create, manage, and enhance computer vision datasets. It provides a user-friendly interface for uploading images, annotating them, and exporting them in a format ready for training machine learning models. Roboflow offers a free trial license, which I used to produce my real-world image dataset.

Using Roboflow significantly streamlined the process of creating a high-quality, annotated dataset from video footage. An average of 500 pictures per magnet was generated by sampling 2 frames per second.

As it will be explained in the next paragraphs, the real images plays a crucial role in this analysis. Their acquisition took a relevant amount of time, because this was kind of an iterative process: (i) taking a set of pictures, (ii) selection and labeling, (iii) trying them in the models, (iv) critical analysis of the outcomes, (v) if necessary, coming back to the acquision/labeling phase. I realized  that cr4eating a reliable dataset of images is not a straightforward process, and I believe that this project has been a useful experience do delelop a conscious undersatnding about this topic.

The main criticalieties I found are described in the following.

I understood pretty early the importance of having two distinct dataset for (i) training and (ii) validation. I initilally used pictures from the same photo session for both training and validating. Said pictures had been first shuffled and divided into two distinct dataset, not to have a contamination between the two groups. However, pictures in the two groups tended to be too much similar leading to results I believe too much optimistic, generating an overfitting model not-capable to abstract and generalize the inner information. Therefore I took two distinct photo sessions, in same environment but in different setups. Training picture were acquired from a steady position with the magnet hanged to a crane and spinning. Validation pictures were taken with the magnet standing on a base and operator walking around it. 

Another issue that took me time to understand how to manage it was the background. I couldn't get pictures with a neutral background, and especially without other magnets in the background. Onfortunately the operators who supported me in the shooting left several magnets around, so when I walked around each magnet to take pictures, I also collected several other magnets on the back. I initially tried to neglect this, but after some iterations I understood that you should actually consider eventual other "alien" classes in the scene and label them accordingly. As better explained in the following paragraphs, this greatly helped YOLO in understanding the ambient around.

### 5.1 Developing a training and validation



![image](https://github.com/user-attachments/assets/e15b1775-02b9-4b2e-b6cd-5a4e743d9727)

## 6. Modeling

In this project, YOLOv8 from Ultralytics was chosen as the model for image classification and object detection tasks. YOLOv8 is part of the YOLO (You Only Look Once) family of models, known for their balance between speed and accuracy, making them highly suitable for this experimental application.

YOLOv8 from Ultralytics was selected primarily because it was the preferred choice of other research groups in similar works, particularly those at the DESY particle accelerator (Hamburg, Germany). Additionally, a brief research on different releases further supported this choice. Its combination of speed, accuracy, flexibility, and ease of integration made it the ideal solution for the project’s requirements. The decision to use YOLOv8 was further reinforced by its strong community backing and the continuous improvements being made by Ultralytics, ensuring that the model could meet both current and future project needs.

### 6.1 The Use of Weights & Biases (W&B)

During the development and training of the YOLOv8 model for image classification, I integrated Weights & Biases (W&B) into the training pipeline ([LINK](https://wandb.ai/)). W&B is a powerful tool for experiment tracking, model management, and data visualization, which proved invaluable in analyzing and refining the model. Its use was recommended by other research scientists who had already appreciated its resources when exploring complex trainings. By taking advantage of the free trial and its comprehensive plotting functionalities, I was able to gain deeper insights into the model's performance and make informed adjustments throughout the training process. Since YOLO models are well-supported by the research community, integrating W&B was relatively straightforward.

### 6.2 Training and Test Configurations

As a very first step I created, trained and validated a standard YOLO model only using the whole CAD dataset. This allowed me to create an ideal model with almost perfect performance (both precision and recall close to 1). This trained model was then validated using a random selection of real images, 40 per class plus additional 40 as background. Background is a special default class in YOLO indicating a picture with no objects to be identified. The results of this test was miserable. The model basically never works: almost all the provided real images are misclassified as 'background', meaning that this trained model cannot identify any object in the validation pictures. It seemed that the model couldn't relate the gained information of the training phase to the validation process. The overall performance was so low that some metrics were misleading. The overall precision doesn't seem too bad; some classes, like QD and QF, have a precision of 1. However, the high precision for QF and QD indicates that the model is not making false positive errors for these classes, but it might not be making many predictions for these classes at all. Precision here is a misleading parameter; this is why it was crucial to consider both precision and recall together (or more holistic metrics, like F1) to get a complete picture of the model's performance (recall <2%).

Another hypotesis was that the model for some reasons wasn't able to properly work with the provided real dataset. To understand this and to have a benchmark, I generated a new model and I trained it excluvely using real images. At this point the distinction between real images for training and real images for validation bacame crucial. As anticipated, I believed that validating on images different but very similar to the ones used in training process could potentially bring to overfitting. That is what brought me to develop two distinct datased, taken in different conditions. The metrics are promising: Precision: 0.923, Recall: 0.834, F1: 0.876. I also learned about another metrich which I considered in the rest of the analysis, "mean Average Precision at IoU=0.50 (Intersection over Union)", mAP50 ([LINK](https://docs.ultralytics.com/guides/yolo-performance-metrics/). This is a common evaluation metric used in object detection tasks, especially when working with YOLO (You Only Look Once) models. It measures the accuracy of the model in detecting and correctly classifying objects in images. Therefore it covers two aspects: not only (i) the correct classification, but also (ii) the identification of the objects in the picture. In this fully real model mAP50 was 0.926.









### 6.3 Developed Scripts for Identified Configurations

- [train_yolo_0.py](https://github.com/emmetizeta/CP-Vision-Lab-Tool/blob/main/train_yolo_0.py): Model validation, full CAD dataset
- [train_yolo_1.py](https://github.com/emmetizeta/CP-Vision-Lab-Tool/blob/main/train_yolo_1.py): CAD dataset for training only, with 500 real images for validation
- [train_yolo_2.py](https://github.com/emmetizeta/CP-Vision-Lab-Tool/blob/main/train_yolo_2.py): Real images for training (1k), with 500 real images for validation 
- [train_yolo_3.py](https://github.com/emmetizeta/CP-Vision-Lab-Tool/blob/main/train_yolo_3.py): Limited selection of real images for training (40), with 500 real images for validation 
- [train_yolo_4.py](https://github.com/emmetizeta/CP-Vision-Lab-Tool/blob/main/train_yolo_4.py): Limited selection of real images for training (40) + CAD imaages (weighted), with 500 real images for validation 

## 7. Evaluation

The performance of each training configuration was evaluated using a validation dataset consisting of random 500 real-world images of the magnets. The key metric used for evaluation was mean Average Precision (mAP) at IoU threshold 0.50 (mAP50), together with a confusion matrix to understand more in detail the classification process.

### 7.1 Validation of the Model
To verify the scripts in all their parts (from the image generation and labeling, to classification) a first ideal case is considered. Both training and validation dataset are made by CAD syntetic images (ratio 80-20%). The results are close to perfection, proving the classification model is working properly. As said, this test didn't want to provide any useful insights; it was just a quick way to verify that no major errors were present. This approach helped me to identify an inefficiency in the labeling routine. I initially observed surprisingly low scores in this ideal case, which didn't make sense to me. A more careful analysis allowed me to improve the labeling file (the one here provided) and get classification scores close to 1.

![image](https://github.com/user-attachments/assets/9ed71fcb-9bbe-4612-a568-4ae4065a7ec0)

### 7.2 CAD Images Only
This was the first real test. Training with synthetic CAD images alone resulted in poor performance. The synthetic images were probably too ideal, with plain white backgrounds and consistent lighting, unlike the real-world conditions. This observation comes from the confusion matrix, where every real test image is classified as "background", meaning that the model cannot recognize any object and classify everything as a meaningless background, without objects over it.

![image](https://github.com/user-attachments/assets/611eb061-04f5-4f0c-b728-b5a52169e19d)

### 7.3 Real Images (High Population)
Training with a substantial number of real images (1k, randomly selected among classes) yielded almost perfect classification performance. This confirm the quality and robustness of YOLO: even a not-optimized model of this kind can achieve outstanding results with a sufficient number of images. How much means sufficient? Even a dataset with much less images provided good results (e.g. 100-200); the main difference is that the model needs more epochs and time to converge.

![image](https://github.com/user-attachments/assets/564c35ef-2572-4285-9f2e-d0e0564a9146)
![image](https://github.com/user-attachments/assets/6b35f019-80c6-4c15-a6bd-7968d1352e5c)

### 7.4 Real Images (Very-Limited Population)
This training dataset aims to simulate what an operator can do in the real life. 8 images are aquired per magnet: walking around the object, one photo per side, plus one photo per each angle. Training with such a small dataset of real images resulted in reduced performance (compared to 5.2) and required many epochs to get a stable model.

### 7.5 Combining Synthetic and Real Images
This test aims to understand if a population of syntetic images can help to improve the results obtained with the reduced real dataset. Initially I followed some attempts to create balanced datasets, with a comparable number of real and syntetic images. However, this approach didn't provide evident improvements (quite the opposite...). Much better performances are obtained with a weighted approach. A combined weighted strategy improved performance slightly and significantly reduced the number of epochs required for training. This suggests that synthetic data can aid in building a robust classifier when realistic variations are introduced.

![image](https://github.com/user-attachments/assets/2d6be580-a1df-46c7-9150-329d65a63085)
![image](https://github.com/user-attachments/assets/97d723e7-98a9-45aa-ad31-3cfa88af3094)

### 7.6 Analysis

The results indicate that while synthetic data alone is insufficient due to the lack of real-world imperfections, combining synthetic data with real data can enhance the model's performance and reduce training time. Key observations include:
- The synthetic images here produced provide good initial dataset but fail to capture the variability of real-world conditions.
- Real images are crucial for achieving high performance, but collecting a large dataset can be challenging.
- A mixed approach suggests that can leverage the strengths of both synthetic and real data, improving robustness and training efficiency.

## 8. Findings and Future Recommendations

The study demonstrates that synthetic datasets can provide valuable assistance in training image classifiers, but the quality and realism of synthetic data are crucial for effective generalization to real-world conditions. Introducing realistic variations in synthetic data can significantly improve the performance and robustness of the model. As a future improvement of the model I would like to gemerate the CAD pictures with random gaussian background and different light conditions. I'd like to understand if YOLO classifier would be more able to separate the central object from the backgroud. I was hoping the model was already able to do it from the shapesover ideal white backgrounds, but it's evident that this is not the case.

## 9. Deployment

The deployment phase would involve integrating the trained model into a real-time image recognition system within the particle accelerator facility. This would provide operators with real-time information about the components they are working with, enhancing safety and efficiency. The deployment process would include:
- Setting up the hardware and software infrastructure for real-time image capture and processing.
- Integrating the trained model into the system.
- Testing and fine-tuning the system in the actual operating environment.
This future scenario is possible but will require an extensive effort to be implemented, after this first analysis.

## 10. Conclusion and Answer of Initial Question

*Is it possible to develop an effective image recognizer using a collection of 'synthetic' images produced by CAD software?* Yes, but only with realistic CAD data. The project demonstrates that while synthetic datasets (even ideal) can assist in training image classifiers, the realism of synthetic data is crucial for effective generalization to real-world conditions. Introducing realistic variations in synthetic data should significantly improve the performance and robustness of the model. This work provides an initial understanding and demonstrates the potential of using CAD-generated synthetic data for training image classifiers. Further improvements are possible and could involve refining the synthetic data generation process to better mimic real-world conditions.


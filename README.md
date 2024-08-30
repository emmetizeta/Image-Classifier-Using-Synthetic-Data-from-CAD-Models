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

Jupyter notebook to generate the CAD dataset (images+labels): [LINK](https://github.com/emmetizeta/Image-Classifier-Using-Synthetic-Data-from-CAD-Models/blob/main/1_CAD_Images_Generator.ipynb)

![image](https://github.com/user-attachments/assets/e71a67fc-7f41-4e68-8c4c-83a5a262ed63)

## 5. Data Collection 2: Real-World Images

Each magnet was placed in an isolated position, without any other objects obstructing the view. A 4K 60Hz video was recorded for each magnet, capturing a 360-degree view by walking around the object. I then used the online tool 'Roboflow' to extract individual frames from these videos and prepare a labeled dataset ([LINK](https://roboflow.com/)). This process allowed me to efficiently generate a large, diverse set of images with the corresponding labels.

I discovered Roboflow after conducting online research on similar tools. Roboflow is an online platform designed to help developers and data scientists create, manage, and enhance computer vision datasets. It provides a user-friendly interface for uploading images, annotating them, and exporting them in a format ready for training machine learning models. Roboflow offers a free trial license, which I used to produce my real-world image dataset.

Using Roboflow significantly streamlined the process of creating a high-quality, annotated dataset from video footage. An average of 500 pictures per magnet was generated by sampling 2 frames per second.

As will be explained in the following paragraphs, the real images play a crucial role in this analysis. Their acquisition took a significant amount of time because it was an iterative process: (i) taking a set of pictures, (ii) selecting and labeling them, (iii) testing them in the models, (iv) critically analyzing the outcomes, and (v) if necessary, returning to the acquisition/labeling phase. I realized that creating a reliable dataset of images is not a straightforward process, and I believe this project has been a valuable experience in developing a conscious understanding of this topic.

The main challenges I encountered are described below.

- Early on, I understood the importance of having two distinct datasets for (i) training and (ii) validation. Initially, I used pictures from the same photo session for both training and validation. These pictures were first shuffled and divided into two distinct datasets to avoid contamination between the two groups. However, the pictures in the two groups tended to be too similar, leading to results that I believe were overly optimistic, generating an overfitting model that was not capable of abstracting and generalizing the underlying information. Therefore, I conducted two distinct photo sessions in the same environment but with different setups. Training pictures were acquired from a steady position with the magnet hanging from a crane and spinning. Validation pictures were taken with the magnet standing on a base while I walked around it.

- Another issue that took me some time to understand how to manage was the background. I couldn't take pictures with a neutral background, especially without other magnets present. Unfortunately, the operators who assisted me in the shoot left several magnets around, so when I walked around each magnet to take pictures, I also captured several other magnets in the background. I initially tried to ignore this, but after some iterations, I realized that I should actually consider any other "alien" classes in the scene and label them accordingly. Whether this helped the YOLO model is still unclear to me, and it is something that requires further investigation.

![image](https://github.com/user-attachments/assets/52c4e526-0870-47ca-829f-6c8f16dc4705)

![image](https://github.com/user-attachments/assets/9a77ca4b-a398-40d0-99e5-835d488e310e)

## 6. Modeling and Evaluation of the different Configurations

In this project, YOLOv8 from Ultralytics ([LINK](https://docs.ultralytics.com/)) was chosen as the model for image classification and object detection tasks. YOLOv8 is part of the YOLO (You Only Look Once) family of models, known for their balance between speed and accuracy, making them highly suitable for this experimental application.

YOLOv8 from Ultralytics was selected primarily because it was the preferred choice of other research groups in similar works, particularly those at the DESY particle accelerator (Hamburg, Germany). The decision to use YOLOv8 was further reinforced by its strong user community.

During the development and training of the YOLOv8 model for image classification, I integrated Weights & Biases (W&B) into the training pipeline ([LINK](https://wandb.ai/)). W&B is a powerful tool for experiment tracking, model management, and data visualization, which proved invaluable in analyzing and refining the model. Its use was recommended by other research scientists who had already appreciated its resources when exploring complex trainings. By taking advantage of the free trial and its comprehensive plotting functionalities, I was able to gain deeper insights into the model's performance and make informed adjustments throughout the training process. Since YOLO models are well-supported by the research community, integrating W&B was relatively straightforward.

### 6.1 A Fully Synthetic Model

As a first step, I created, trained, and validated a standard YOLO model using only the entire CAD dataset. This allowed me to create an ideal model with almost perfect performance, with both precision and recall close to 1. This trained model was then validated using a random selection of real images, 40 per class plus an additional 40 as background. Background is a special default class in YOLO indicating a picture with no objects to be identified. The results of this test were disappointing. The model essentially failed: almost all the provided real images were misclassified as 'background', meaning that this trained model couldn't identify any objects in the validation pictures. It seemed that the model couldn't apply the information gained during the training phase to the validation process. The overall performance was so poor that some metrics were misleading. The overall precision didn't seem too bad; for example, some classes, like QD and QF, had a precision of 1. However, the high precision for QF and QD indicates that the model is not making false positive errors for these classes, but it might not be making many predictions for these classes at all. Precision here is a misleading parameter; this is why it was crucial to consider both precision and recall together (or more holistic metrics, like F1) to get a complete picture of the model's performance (recall <2%).

![image](https://github.com/user-attachments/assets/cfb2e042-fab3-4efd-94d5-b71cf4f19192)

### 6.2 A Fully Real Model: How Good Could This Classifier Be?

The previous paragraph demonstrated that a complete dataset of CAD images does not perform well when applied to a collection of real pictures. Therefore, I attempted to train a similar model using a collection of real pictures. To avoid overfitting, I collected the training and validation images in different sessions, creating two distinct datasets; the two collections of pictures are independent of each other. Each class of magnets has an available collection of approximately 100 real pictures. The metrics were promising: Precision: 0.947, Recall: 0.835, F1: 0.887. I also learned about another metric which I considered in the rest of the analysis, "mean Average Precision at IoU=0.50 (Intersection over Union)", mAP50 ([LINK](https://docs.ultralytics.com/guides/yolo-performance-metrics/). This seems a common evaluation metric used in object detection tasks, especially when working with YOLO models. It measures the accuracy of the model in detecting and correctly classifying objects in images. Therefore it covers two aspects: not only (i) the correct classification, but also (ii) the identification of the objects in the picture. In this fully real model mAP50 was 0.929.

![image](https://github.com/user-attachments/assets/c40430a2-8353-4c78-b08c-970433c35039)

### 6.3 A Limited Real Dataset Reinforced with a Set of CAD Images

The model investigated here is essentially the same as in the previous paragraph but with a much more limited set of training pictures. The training dataset consists of 5 photos per class, randomly selected. Five pictures is certainly a very limited number; the idea is to provide the model with a few field-taken pictures and observe how well it generalizes the information and abstracts a model. In addition, a collection of 150 random CAD images is included. The CAD images are selected to maintain a good balance among classes. The number 150 is an arbitrary choice: the real pictures involved in the training total 30 (5 images for each of the 5 classes plus 5 background images). A factor of 5 is used to determine how many CAD images should be included.

This reinforced mixed model showed a drastic improvement over the limited 5-image dataset. Although the model still has room for improvement, there is a clear trend that a training dataset composed of CAD images, reinforced with a limited number of real pictures, can yield promising metrics.

![image](https://github.com/user-attachments/assets/871c70b3-2ce3-4e5c-8661-f2a799952b48)

### 6.4 Optimization of the Mixed Model

A deeper investigation was conducted to "optimize" the model. Grid search cross-validation (CV) is generally not feasible with YOLO models due to their different output structure compared to classification or regression models typically used with grid search CV. Besides the computational complexity, dataset splitting can also lead to issues like overfitting, especially when the number of images is limited. Additionally, the training and validation datasets cannot be mixed in this case: I don’t want CAD images in validation, nor do I want to "contaminate" the two real datasets.

YOLO offers some "tuning" tools, but I haven't had much time to fully explore them. Therefore, I decided to follow a simpler approach. The main parameters I am interested in are (i) the number of CAD images and (ii) the number of real images needed to achieve good performance. How to quantify performance? Two nested loops iterate through a series of potential values, collecting metrics, particularly mAP50. As mentioned earlier, mAP50 is a standard metric in object detection that evaluates a model's ability to accurately detect objects while balancing precision and recall. It measures the mean Average Precision (mAP) when the Intersection over Union (IoU) threshold is set at 0.5, providing a clear and consistent benchmark. Using mAP50 as the reference metric allows for direct comparison with other models and ensures that the YOLO model delivers reliable detection quality.

![image](https://github.com/user-attachments/assets/7eed7b11-c051-4fc7-a6d1-1ddaab247efb)

The graph suggests that:

- Real images greatly improve the metrics, even in small quantities.
- CAD images must be added carefully: if the number of CAD images is too large compared to real images, the effect could be counterproductive.
- According to mAP50, the targeted metric, there are three significant steps mainly related to the number of real images (5, 10, 20).

Considering the metrics displayed, a good compromise could be 500 CAD images combined with 10 real images per class. Of course, it would be easy to select 20 or more real images, but the scope of this analysis is to minimize the real image population as much as possible. Ideally, the model should be trained with as few real images as possible. On the other hand, it is also evident that too many CAD images don't help; in fact, they could have quite the opposite effect. This is particularly clear in the case of 1000 CAD and 5 real images. Considering the selected distribution (500 CAD, 10 real), the total number of real images is 10 per 5 classes + 10 background images = 60 real images. Approximately, this means a ratio of 10:1 between CAD and real images, which could be considered a good "rule of thumb" for balancing the two datasets.

This analysis provides an important insight: CAD images can truly be the "backbone" of the model, but they need a minimum number of real images (even 5 per class is sufficient for decent results). This implies that YOLO requires minimal information to understand the context and environment. Once it has that, the model can generalize the information and effectively leverage the extensive training from the CAD dataset.

### 6.5 Evaluation of Optimized Model

The optimized mixed model is 500 CAD images, plus 10 real images per class and 10 images as background.

Optimized Mixed Model Summary:
- Precision: 0.908
- Recall: 0.792
- mAP50: 0.892
- F1: 0.846

![image](https://github.com/user-attachments/assets/c3a95515-93fc-495b-b00a-fa651ae731f7)


## 7 Conclusions and Final Recommendations

The use of the YOLO framework has proven to be a highly effective tool for developing an object classifier. The primary goal of this analysis was to understand how effective it is to implement such a tool primarily using CAD-generated images, as opposed to real pictures, which are usually more challenging to collect. The results are summarized in the table and graphs below.

![image](https://github.com/user-attachments/assets/89d5ba60-ba64-4353-ac27-edf0affad0e4)

![image](https://github.com/user-attachments/assets/ab8b268d-6bb3-47c7-959a-bca6c9c1516e)

The analysis suggested that this mixed approach is quite effective. A model trained solely on CAD images is not adequate for detecting real-world objects. Conversely, a model trained with very few real images provides modest results. However, merging the two approaches leads to a tremendous improvement: with just a few real-world images in the training set, YOLO appears to generalize the main information from the dataset and apply it in an environmental context. The dataset investigated suggested a rule of thumb with a scaling factor of 10 between the number of artificial CAD images and real images. A dataset trained with a total of 60 images (10 per class + 10 background) and 500 CAD images yielded very good results, not far from the best model trained with 600 real-world images.

The analysis has addressed my initial question:

`Is it possible to develop an effective image recognizer using a collection of 'synthetic' images produced by CAD software?`

**YES**: CAD images have proven to be a very powerful tool for easily developing an image recognition tool. However, they cannot be used alone and will always require a minimum number of real-world images to allow the classifier to correlate the artificial dataset with the real world.

## 8. Suggested Next Steps

This work has been quite challenging in many aspects and required more time than expected in different areas:

- Developing an understanding of the YOLO model and its tools
- Gaining experience with the NERSC Perlmutter system
- Developing reliable tools to produce a CAD dataset
- Finding the right real-world images and learning how to use them effectively
- Understanding how to label the datasets accurately
- Integrating all components into a cohesive and effective process
- Analyzing the results, interpreting the metrics, and optimizing the outcomes

The project turned out to be a true full-stack application of machine learning. Due to time constraints, I couldn't complete the final step: deployment. The plan was to take the most promising mixed model and apply it to a new dataset of completely new images, even in different environments, with the ultimate goal of achieving a real-time video application. I made some progress, but the results weren't yet adequate to be presented here. This is something I will continue working on in the coming weeks/months to further develop this analysis.

Regarding the analysis itself, there are some open points that I couldn't address; I would recommend further development in these areas:

- Understanding YOLO’s auxiliary tools (e.g. tuning). Some of them appear very powerful and could significantly boost results.
- Extending the parameter research. I focused on parameters that interested me, but many other parameters deserve consideration (e.g., number of epochs/batches, size of images).
- Studying the effect of the selected real images. Are some images more effective than others? Is there a criterion for selecting specific images in terms of object position, lighting, distances, etc.?
- Unfortunately, the photo shooting area was not well-organized. In many pictures, there is a magnet in the foreground and many other magnets in the background. Does this affect the model, or is it helpful? I suspect this may have impacted the results: in all the confusion matrices, several backgrounds were interpreted as magnets. This suggests that the -model sometimes detects magnets where there aren’t any. This issue should be further investigated.
- The color in the images. Due to limitations in the CAD image generator, I could only obtain grayscale images. To maintain consistency, I used grayscale real-world images as well. What if I had used color images instead? I made some attempts, and it seems that color models tend to overfit. However, this is another aspect that was only slightly touched upon. More intensive and rigorous work is required.

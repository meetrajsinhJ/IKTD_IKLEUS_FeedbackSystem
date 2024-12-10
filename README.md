# IKTD_IKLEUS_FeedbackSystem
Bewertung von Technische Zeichnungen


Table of Contents
1. Dataset
2. Data Augmentation
3. Clustering for Exploratory Data Analysis
4. Computer Vision Operations
5. Image Classification
6. Image Segmentation
7. Feedbacksystem Module

1.DATASET

4 different types of Technical Drawings along with their respective Evaluation Table:-
a) Schraube
b) Baugruppen
c) Toleranz
d) Gussaufgabe


 ![Screenshot 2024-12-10 at 15 05 52](https://github.com/user-attachments/assets/81e3b8d2-3c05-41d5-9c4e-d9498f9eb04c)

2.DATA AUGMENTATION


a). Increasing the the dataset by implementating various data Augmentation Techniques:
Scale variation , Brightness and Contrast , Rotation etc.

![Screenshot 2024-12-10 at 15 15 05](https://github.com/user-attachments/assets/66110008-3132-4017-9ff9-0b737ac809be)

b). Adding noise and reconstruction using Variational AutoEncoders
![reconstructed](https://github.com/user-attachments/assets/f3aee5d1-66c6-4b4e-8912-4befa47fb6a3)
![vaeimages](https://github.com/user-attachments/assets/cb3bd2a1-8262-44f5-a261-4b256cd12ae0)

3.CLUSTERING

Appling k-means Clustering algorithm on the dataset.

a) Clustering on Corrected and Not-Corrected Images
![Screenshot 2024-12-10 at 15 21 03](https://github.com/user-attachments/assets/4f7c59a2-b25c-4191-a612-ce6299bd8e38)
b) Clustering on Entire Dataset
![Screenshot 2024-12-10 at 15 21 18](https://github.com/user-attachments/assets/470125cd-67e8-4953-8e42-ba147fa325b5)

4.COMPUTER VISION OPERATIONS:



a). Canny Edge Detection:

![Screenshot 2024-12-10 at 15 23 31](https://github.com/user-attachments/assets/564213f5-94ba-4d61-b135-516c4bcab0d2)
![Screenshot 2024-12-10 at 15 23 40](https://github.com/user-attachments/assets/b2e29034-5841-4811-87a6-6650022365cb)




b). SIFT feature Matching and Homography Matrix Calculations:

![Screenshot 2024-12-10 at 15 24 49](https://github.com/user-attachments/assets/d9b4a0ea-ad41-445f-bae6-c5865273934c)


5.IMAGE CLASSIFICATION:


-> Classification of the images based on the drawing type: 

Finetuning ResNet50 , EfficicentNet Models on this custom Dataset.
Models along with the Training Scripts are uploaded in Image Classification directory.






6.IMAGE SEGMENTATION:

-> Segementing the drawing and the text box on the Image.

   Trained a yolov8 model on custom dataset.

   ![result](https://github.com/user-attachments/assets/c3fd8783-8096-4256-acd2-eb9cb11b2e1c)

7.FEEDBACK SYSTEM MODULE:
   
<img width="655" alt="Screenshot 2024-11-17 at 18 01 01" src="https://github.com/user-attachments/assets/de31df5e-d194-42f7-9e5f-1592d4918f6a">

Dataset Preparation:

![Screenshot 2024-12-10 at 15 36 30](https://github.com/user-attachments/assets/6a1760b4-75ac-4eeb-8339-cedfcd6c5a0f)

Model Fusion:


Combining Visual Features with Text Analysis along with Domain Knowledge to generate feedback on the drawing.
![training_loss](https://github.com/user-attachments/assets/d45abec0-2d00-41d4-8243-321b25214824)
![training_metrics](https://github.com/user-attachments/assets/46a65fc2-6439-4d0f-8cbd-50a8fc93bcdf)



Output:
![inference_visualization](https://github.com/user-attachments/assets/c9814aae-ecca-4d1a-a283-f8723a1ffab0)
![inference_visualization9](https://github.com/user-attachments/assets/bd85b1e6-0112-44f8-b699-77d24d3e5372)
![inference_visualization10](https://github.com/user-attachments/assets/65dac3ad-30eb-4e38-b33e-b20d23558771)




Contact Author: meetrajsinh19.de@gmail.com


Institut für Konstruktionstechnik und Technisches Design
https://www.iktd.uni-stuttgart.de



RICE IMAGE CLASSIFICATION BASED ON DEEP LEARNING

![image](https://github.com/user-attachments/assets/b7b1e7b2-b19c-4c4f-af1f-1f1342423ac9)

Authors: Jane Martha: Yvonne Mwangi, Bethuel Maruru, Loise Mburuga, Jane Martha,Sonia Ojay,Faith Mwenda 



                             PROJECT  OVERVIEW

 The project focuses on leveraging a dataset of Rice images to develop machine learning models capable of
 accurately identifying quality rice. The dataset sourced from Kaggle, comprises of images classified into 5 categories:
 arbori,basmati,ipsala,jasmine,karacadag. This project aims to build and optimize convolutional neural network(CNN) models to achieve high quality classification accuracy.

 
![image](https://github.com/user-attachments/assets/7575658d-f419-43bf-97ae-a02f38ca373b)

                          PROBLEM STATEMENT

                          
Rice is a staple food for more than half of the world's population, making its quality and type assessment crucial for both consumers and producers. Traditional methods of rice classification are labor-intensive, time-consuming, and prone to human error. With the advent of machine learning and image processing technologies, there is a significant opportunity to automate and improve the accuracy of rice classification processes.

                           DATA UNDERSTANDING


Source: Kaggle's Rice Image Classification dataset. Structure: The dataset icontains five rice varieties namely arbori,basmati,ipsala,jasmine,karacadag with 75,000 images.
 directories: train: Contains the following Class names and their indices:
Arborio: 0
Basmati: 1
Ipsala: 2
Jasmine: 3
Karacadag: 4

Class names:
['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

                            OBJECTIVES


The primary objective of this project is to develop a machine learning model capable of accurately classifying different types of rice grains based on their images. The classification will be performed using a dataset comprising various rice types, each with distinct visual characteristics.


                            Methodology

  This project encompassed the following tasks:

Data Acquisition and Preprocessing: Collection and preprocessing of a diverse set of rice grain images to ensure the dataset is suitable for model training and evaluation.
Model Development: Design and implementation of machine learning models using techniques such as Convolutional Neural Networks (CNNs) to classify the rice images.
Model Evaluation: Evaluation of the model's performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score) to ensure its reliability and robustness.
Optimization: Fine-tuning the model to enhance its performance and reduce computational costs.
Deployment: Development of a user-friendly interface for practical applications, enabling end-users to upload rice images and receive classification results.


                           Expected Outcomes
The expected outcomes of this project include:

A trained machine learning model capable of classifying rice types with high accuracy.
A comprehensive analysis of the model's performance and potential areas for improvement.
A deployable application that facilitates easy and accurate rice classification for end-users.
By addressing these objectives and challenges, this project aims to contribute to the advancement of agricultural technologies and improve the efficiency and accuracy of rice classification processes.

                        Models Categories

1: Baseline Convolutional Neural Network Model(CNN)
2:Tuned Convolutional Neural Network Model(CNN)
3:Complex Model
4:ResNet50V2 Mode        


                         OUTPUT

![image](https://github.com/user-attachments/assets/4acf61d9-6486-4e95-85b1-6372653870a6)

We also Checked Class Balance to ensure that all classes had a similar number of instances. We confirmed that our dataset was balanced because all classes had equal number of instances


                       MODEL ARCHITECTURE

Baseline Mode

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 126, 126, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 63, 63, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 61, 61, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 30, 30, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 57600)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │       7,372,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 5)                   │             645 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 7,392,965 (28.20 MB)
 Trainable params: 7,392,965 (28.20 MB)
 Non-trainable params: 0 (0.00 B)



The baseline model architecture comprises a total of 7392965 parameters, with all parameters being trainable. The model's size is approximately 28.20 MB. There are no non-trainable parameters in this architecture, showing that all parameters are updated during the training process


          precision    recall  f1-score   support

     Arborio       0.99      0.95      0.97      2294
     Basmati       0.99      0.99      0.99      2259
      Ipsala       1.00      1.00      1.00      2186
     Jasmine       0.99      0.99      0.99      2206
   Karacadag       0.95      1.00      0.97      2305

    accuracy                           0.98     11250
   macro avg       0.98      0.98      0.98     11250
weighted avg       0.98      0.98      0.98     11250


Observations:
 We observed that execution time decreases each time a new model is executed and the accuracy becomes better each
 time hence boosting performance.
The best model is the Baeline Model, which has validation accuracy of 0.98 and took the shortest execution time of 1hr
 30min 10 secs.
 The least was Pre-trained ResNet50V2 Architecture



                           Conclusion
                           
The rice image classification project demonstrates the potential of using advanced machine learning techniques to automate and improve the accuracy of rice type identification. By leveraging convolutional neural networks (CNNs) and image processing technologies, we have developed a robust model capable of distinguishing between various types of rice grains based on their visual characteristics.

                   Recommendations


 Future enhancements to this project could include expanding the dataset to include a wider variety of rice types and further fine-tuning the model using advanced techniques such as ensemble learning. Additionally, integrating more sophisticated image preprocessing steps and exploring the use of other deep learning architectures could yield even better performance.
                           
                        

                        

                        

                            
                             

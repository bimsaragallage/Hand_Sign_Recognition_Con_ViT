# Hand_Sign_Recognition_Con_ViT

This project explores American Sign Language (ASL) hand gesture recognition using Vision Transformers (ViT). The performance of Keras' ViT implementation is compared with a custom model integrating Convolutional Additive Attention.

## Installation

### Dataset
Download the American Sign Language dataset from Kaggle using the following link:
[ASL Dataset on Kaggle](https://www.kaggle.com/datasets/ayuraj/asl-dataset)

### Requirements
Ensure you have Python 3.11 installed. Use the following command to create a virtual environment and install the required libraries:

# Create a virtual environment (optional but recommended)
python -m venv asl_env
source asl_env/bin/activate  # On Windows: .\asl_env\Scripts\activate

# Install dependencies
pip install numpy matplotlib scikit-learn seaborn tensorflow keras

## Project Description

Hand gesture recognition systems have gained significant attention in recent years due to their wide range of applications, including assistive technology for people with disabilities. While Convolutional Neural Networks (CNNs) have been highly effective in developing these systems by capturing static and dynamic hand gestures, Vision Transformers (ViTs) have emerged as a promising alternative. ViTs are known for their unique mechanism of dividing images into patches, vectorizing them, and calculating self-attention across the patches, enabling them to capture long-range dependencies within an image.

Despite their promising capabilities, ViTs face challenges related to high computational complexity and capacity requirements due to their use of dot-product similarity calculations. Recent advancements have introduced alternative approaches to reduce the complexity, such as the "Convolutional Additive Self-Attention" mechanism. This method simplifies the similarity score calculations, potentially enhancing model efficiency.

### Objectives
The primary goals of this project are:
1. To develop a Convolutional Additive Vision Transformer-based model for recognizing American Sign Language (ASL) hand gestures.
2. To compare the performance and efficiency of the Convolutional Additive Attention model against the traditional multi-head self-attention Vision Transformer, focusing on key metrics like parameter counts, floating-point operations (FLOPs), runtime, and inference times.

### Methodology
This project involves evaluating and comparing two different Vision Transformer architectures:
- **Traditional Multi-Head Self-Attention ViT**: Implemented using Keras’ Vision Transformer, which utilizes dot-product similarity calculations.
- **Convolutional Additive Self-Attention ViT**: A modified version of ViT incorporating convolutional additive self-attention, designed to streamline the similarity calculation process and improve computational efficiency.

Both models were trained and tested using the American Sign Language dataset. Their performance was measured in terms of training, validation, and testing accuracy. Additionally, the models' efficiency was evaluated by calculating the total number of parameters, FLOPs, and runtime, providing a comparative analysis through visualizations and graphs.

#### Comparative Performance Graphs

- **Training and Validation Accuracy**: A line graph showing the training and validation accuracy of both models across epochs.
- **FLOPs and Runtime Efficiency**: A bar graph comparing the FLOPs and total runtime of the two models.

![out4](https://github.com/user-attachments/assets/7586169a-7cfd-400e-af7a-5dd0deb85fc4)
*Figure 1: Comparative Training and Validation Accuracy of Traditional and Convolutional Additive Attention ViTs.*

![out5](https://github.com/user-attachments/assets/707cfedc-c45a-49ee-bb6e-ddfe694cee5d)
*Figure 2: FLOPs and Runtime Efficiency Comparison between the Two Models.*


## Conclusion

It can be mainly concluded that between the two types of models we have performed, which were traditional multi-head self-attention and convolutional additive attention, the convolutional additive attention model has achieved greater success in the hand gesture recognition task. The model yielded higher testing accuracy and lower test loss while maintaining a lower FLOP count, parameter count, and significantly reduced training time with faster inference times.

Considering the inference time and FLOP count, this modification on attention is well-suited for real-time and mobile applications, as highlighted in the original research. This advancement facilitates efficient training on specific segments of image classification tasks due to the minimal training time required.

Additionally, while training with a limited number of epochs and a fixed patch embedding size is effective, extending training to higher-dimensional patch embeddings and more epochs can result in even greater accuracy and efficiency. This approach helps prevent underfitting and enhances the model's capability to distinguish similar hand signs more accurately.

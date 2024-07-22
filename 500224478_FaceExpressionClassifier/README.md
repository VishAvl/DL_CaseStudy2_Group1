# Face Expression Classifer

This is a face expression classifer model that uses VGG16 with VGG_Face weights which is a VGG16 model train on faces dataset
that would be perfectly useful in this case. We unfreezed last layer of the base model and then passed it over a final 
classification head. 

Along with data-augmentation the model was trained for 25 epochs which resulted in ~60% of validation accuracy without any 
sign of overfitting. 

### Frontend
Have a look at the application by clicking the link 👉: [Face Expression Classifier](https://face-expression-classifier.streamlit.app/)

### Backend
### Backend
We used Huggingface's spaces to deploy the Neural Network and the link for it is 👉: [Backend](https://huggingface.co/spaces/gruhit-patel/face-expression-classifier/tree/main) <br>.
For this particular project the requirements.txt has very very specific versioning details because for this project we had made use of VGG_Face library 
that provides us with the pretrained VGG model over VGGFace dataset and hence to include it we had to tweak a lot with perfect python version as well as 
other packages versions as well.

#### App Demo
[Video.webm](https://github.com/user-attachments/assets/56ac4f57-2c42-4f14-9d19-1bbceb145060)

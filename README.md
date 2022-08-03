# Honors_Project
This was my 4th year Honors Project which explored Machine Learning with newly researched cervical cancer screenings.

The Ecole Polytechnique in France has developed a new biomedical diagnostic tool called Mueller Polarimetry which involves exploiting the polarisation of light to determine whether biological tissue is healthy or not.

A 16 Dimensional Matrix is output by this method resulting in a 600x800x16 image compared to normal 600x800x3 rgb images. A pathologist then performs a diagnoses on this tissue to mark regions of the tissue as CIN2-3 (unhealthy) and healthy tissue. 


![image](https://user-images.githubusercontent.com/27258375/182729451-8c860b58-032e-4cda-a441-cea3cc1f2653.png)

From this, I was able to extract each classified pixel with 16 features and 1 classification to train multiple machine learning pipelines to classify unknown cancer tissue.

![image](https://user-images.githubusercontent.com/27258375/182729604-c7d590ee-24d7-4f21-a2bc-c88ca6bbf376.png)

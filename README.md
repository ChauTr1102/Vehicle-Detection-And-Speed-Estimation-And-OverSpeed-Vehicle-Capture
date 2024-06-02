
# Vehicle Detection And Speed Estimation And Over Speed Vehicle Capture


The "Vehicle Detection And Speed Estimation And OverSpeed Vehicle Capture" is an advanced project developed as part of a course at FPT University. This system is designed to enhance road safety by detecting vehicles that exceed speed limits and capturing their license plate information. It is a crucial tool for law enforcement agencies and traffic management authorities to monitor and regulate vehicle speeds, reducing the incidence of speeding-related accidents and ensuring compliance with traffic regulations.
## Feature

This System has following Features:
 
 -  Using Mouse to draw the zone of working. 
 -  Vehicle classification.
 -  Vehicle counting.
 -  Speed estimation.
 -  Overspeed detection and License plate recognition with OCR.
 



## Google colab

[Colab is here](https://colab.research.google.com/drive/1tlNGIsUnlHYSgf2i7Dln3h65MIv9YDSY#scrollTo=bm34yPDHMS8H)


## System design
The system initially captures camera data and classifies vehicles. If successful, it estimates vehicle speed and checks if it exceeds the road's limit. If not speeding, it continues capturing frames. If speeding, it moves to license plate recognition. Successful recognition allows progression, otherwise, it resumes capturing frames. This iterative process ensures only compliant vehicles proceed to further stages.
![Logo](https://github.com/ChauTr1102/20-10-flowers/blob/main/img/diagram.png?raw=true)

## Zone selection breakdown

![image](https://github.com/ChauTr1102/20-10-flowers/blob/main/img/speed.png?raw=true)

To address limitations in our initial speed estimation approach, we developed an advanced algorithm incorporating perspective transformation using OpenCV. This transformation eliminates distortions by converting image coordinates to actual road coordinates. The algorithm defines a trapezoidal source region of interest based on road perspective, transformed into a rectangular target region using a calculated matrix (M). Point-tracking capabilities in computer vision libraries facilitate transforming bounding box vertices, enabling calculation of vehicle distance traveled over time. This yields speed in meters per second, convertible to kilometers per hour by multiplying by 3.6.

## Output Result

![image](https://github.com/ChauTr1102/20-10-flowers/blob/main/img/Screenshot%202024-06-02%20105101.png?raw=true)

The license plate recognition system initially performs well but requires high-resolution images to function effectively. Despite successful identification and extraction of license plate regions, the system's accuracy diminishes with lower resolution inputs. This limitation highlights the dependency on image quality for optimal performance, suggesting potential enhancements in handling lower-resolution images to improve overall system robustness and versatility.

## Classification Output
![i](https://github.com/ChauTr1102/20-10-flowers/blob/main/img/Screenshot%202024-06-02%20114039.png?raw=true)

## Speed Estimation Output
![i](https://github.com/ChauTr1102/20-10-flowers/blob/main/img/Screenshot%202024-06-02%20104451.png?raw=true)

## Lisence Plate Recognition Output

![i](https://github.com/ChauTr1102/20-10-flowers/blob/main/img/Screenshot%202024-06-02%20105206.png?raw=true)

You will need a telephoto lens to squeeze the focal length of the image so the vehicle would look bigger, or a very high resolution footage of traffic or set your camera very close to the vechicle so that the OCR could work properly. If not, we afraid due to the cropping algorithm of the YoLo when capturing the bounding box would downsize the image to size that could not be used for license plate recognition. You could use some method to tackle this problem is to use GAN to enhance the quality of the image.

## Capturing Vehicle's Bouding Box Images for futher procession

![oma](https://github.com/ChauTr1102/20-10-flowers/blob/main/img/Screenshot%202024-06-02%20104550.png?raw=true)

The system capture the Vehicle's bounding box images perfectly but like i was saying, with most of the videos on the internet we could not use the OCR for License Recognition. But i think if you have all the things i was mentioning i think this function would work properly.

## Result from The System vs Ground Truth Data 

![im2](https://github.com/ChauTr1102/20-10-flowers/blob/main/img/0307(2).png?raw=true)

![im2](https://raw.githubusercontent.com/ChauTr1102/20-10-flowers/8d0eb2ca348f324f1d6e97a278a7bb07de5a5f96/result.png)

Despite the challenges posed by the bustling and high-traffic conditions of Hoa Lac, we remain undeterred in our pursuit of evaluating the efficacy of our speed estimation program. In an adventurous approach, we decided to immerse ourselves in the dynamic environment by mounting a camera to the chest of one team member. This unconventional method allows us to record and analyze the speed estimation process in real-time, capturing the intricate details of traffic flow and vehicle movements. 


## Authors

- [@Nguyen Ngoc Hieu](https://github.com/Bojjoo)
- [@Trinh Minh Chau](https://github.com/ChauTr1102)
- [@Ngo Duc Manh](https://github.com/zerfop)
- [@Le Nguyen Thanh Binh](https://github.com/BinhLNT)
## 🛠 Skills
Deep Learning, Python, EasyOCR, YoLov8, Optimization


## License

[Apache](https://choosealicense.com/licenses/apache-2.0/)


# YOLO_v1_to_v6

## YOLO V1
### Simple contents
Object Detection 모델은 크게 1-stage detector와 2-stage detector로 나눌 수 있다.
이 둘을 구분하는 기준은 Feature Extract와 Classification하는 과정을 동시에 진행하는가 아니면 순차적으로 진행하는가이다.
- 1-stage detector로는 YOLO 시리즈와 SSD 등이 있으며 특징으로는 속도가 빠르며 정확도가 낮다는 특징이 있다.
- 2-stage detector로는 R-CNN, Fast R-CNN 등이 있으며 정확도가 높고 최적화가 어렵다는 특징이 있다.   
(이 기준을 Feature Extract, Classification을 기준으로 설명하는 사람도 있고, Localization, Classification으로 설명하는 사람도 있음)   
  
객체인식을 수행하는 모델의 구조는 크게 다음과 같이 설명할 수 있다.  
Input Image -> Backbone(feature extract) -> Head(object detection)
(이 떄 Backbone은 특징 추출이 목적이기 때문에 특징 추출에 최적화된 Classification 모델을 사용한다.)
YOLO v1에서는 Backbone 구조로 개발 당시 성능이 좋았던 VGG를 변환하여 DarkNet이라는 모델을 만들어 사용하였다.  
224 * 224 크기의 ImageNet으로 pretrain된 모델을 fine tuning하여 사용하였으며 Inference 시에는 해상도를 위해 448 * 448 크기의 이미지를 사용하였다.  
  
YOLO v1의 Head에서 진행하는 과정을 간단히 정리하면  다음과 같다.
1. Input image를 S * S Grid로 나눔. (논문에서는 S=7)
2. 각 Grid에 대해 Bounding Box, Classification을 진행한다.
    1. Bounding Box
    - 각 Grid는 Bounding Box를 계산한다. (BackBone에서 받은 Feature map을 이용하여 만드는 것으로 추측한다.)
    - 각 Bounding Box는 Box 중심의 x좌표, y좌표, w, h, Confidence score를 가진다.(Box 안에 객체가 존재할 확률)
    - output size : S * S * B * 5 (B:Bounding box의 개수, 5: x,y,w,h,confidence score)
    2. Classification
    - 각 Grid cell에 대해 Classification을 진행.
    - output size: S * S * C (C:각 class별 확률)
3. NMS(Non-maximun Suppression)을 이용하여 겹치는 Box를 처리하고 결과를 도출한다.  

학습에 사용될 Loss는 다음과 같이 간단화할 수 있고 자세한 식은 아래의 그림과 같다.  
$Loss = Localization + Confidence + Classification$

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbpD927%2FbtqRVpnLCGe%2FelD6wAkeSotSm1NYsW9jx0%2Fimg.png)

$\lambda_{coord}$ : Localization loss에 높은 가중치를 주기 위해 논문에서는 5로 설정  
$\lambda_{noobj}$ : 객체가 존재하지 않는 background의 영향을 낮추기 위해 논문에서는 0.5로 설정  
$\mathbb{1}^{obj}_i$ : if object가 존재.

(2-stage detector는 일반적으로 Background class를 따로 두고 학습하는데 YOLO는 Confidence score가 Threshold보다 작으면 Background라 판단할 수 있기 때문에 필요없다.)

YOLO v1의 특징 : 속도는 빠르지만 작은 객체를 잘 탐지하지 못한다. 
(loss를 학습하면서 두 Box 중 IoU가 큰 것만 학습하는데, 큰 객체의 경우 박스간 IoU 차이가 커서 잘 판별할 수 있지만, 작은 객체는 약간의 차이로 IoU의 차이가 뒤집힐 수 있어 잘 못찾는다고 한다. >> YOLO v3에서 해결)

### 논문 공부하다가 생긴 의문
- 각 Grid는 Backbone을 통해 계산된 Feature map을 이용하여 Bounding Box를 계산할텐데 어떠한 과정을 통해 만들어질까?
- 개발 당시 성능이 좋았던 VGG를 변형한 DarkNet을 Backbone으로 이용했다고 하는데 비교적 최근에 개발된 다른 모델을 사용하면 성능이 어떻게 나올까?

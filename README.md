# Simply-Customizing-Card
Future Finance A.I. Challenge 2020
### 주제 : 내 카드를 GAN단하게 커스터마이징하자
##### Idea : [Face2Malnyun](https://github.com/bryandlee/malnyun_faces)


## 1. 데이터 크롤링 ( 여신작가 / 복학왕 ) + 데이터 전처리 ( 얼굴만 따로 캡쳐 )
##### - 다양한 사람이면 좋음
##### - 이말년 그림체는 500장 / 다른 작가님들은 3 channel이여서 더 많이 필요할 수도 있음.
##### - 크기는 상관 없습니다 ( 어차피 다 256으로 Resize 할 예정이지만 크면 클 수록 좋음 )
##### - 만일 웹툰에서 데이터 허락을 받지 못하면, 다른 데이터셋을 찾아봐야 할지도.. ex) [Simpson1](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset), [Simpson2](https://www.kaggle.com/kostastokis/simpsons-faces) [anime](https://www.kaggle.com/splcher/animefacedataset), [Cartoon](https://google.github.io/cartoonset/)


## 2. 모델구현 
##### - 모델 1 : StyleGAN v2 < FFHQ dataset Pre-trained Model > + FreezeD/ADA
##### - 모델 2 : U-GAT-IT


## 3. PPT작성
##### - 작년 수상자(https://github.com/ukiKwon/voice-separater-with-ripTracking)
##### - PPT를 어떻게 만들것인지 구상





## 2020.08.16 Review

1. 데이터셋 정리 ( 심슨 데이터셋 / 웹툰 데이터셋 문의 )
2. 모델에 대한 이해
3. GPU서버 대여 ( Ubuntu / GPU 15TFLOPS / Keras, PyTorch / Python)

## 2020.08.17 Review

1. StyleGAN / U-GAT-IT 구현완료 ( 파라미터 튜닝은 조금 살펴봐야됨, 모델 이해 X, FFHQ Pre-trained 모델 확인 )

|Model|Paper|Code|
|:------|:---|:---|
| U-GAT-IT | [Paper](https://arxiv.org/pdf/1907.10830.pdf) | [GitHub](https://github.com/znxlwm/UGATIT-pytorch) |
| StyleGAN | [Paper](https://arxiv.org/pdf/1812.04948.pdf) | [GitHub](https://github.com/rosinality/stylegan2-pytorch) |

2. FreezeD / ADA 논문 리뷰

|Method|Paper|Review|
|:------|:---|:---|
| FreezeD (Freeze Discriminator) | [Paper](https://arxiv.org/pdf/2002.10964.pdf) | Pretrained된 Discriminator의 low-layer을 Freeze시키고 High-layer만 학습시키는 방법 |
| ADA (Adaptive Data Augmentation)| [Paper](https://research.nvidia.com/sites/default/files/pubs/2020-06_Training-Generative-Adversarial/karras2020-limited-data.pdf )| 5가지 방법으로 확률적으로 Data Augmentation하는 방법 논문 p.19-20 참조 |

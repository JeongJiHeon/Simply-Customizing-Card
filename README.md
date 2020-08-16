# Simply-Customizing-Card
Future Finance A.I. Challenge 2020
## 주제 : 내 카드를 GAN단하게 커스터마이징하자

### 1. 데이터 크롤링 ( 여신작가 / 복학왕 ) + 데이터 전처리 ( 얼굴만 따로 캡쳐 )
##### - 다양한 사람이면 좋음
##### - 이말년 그림체는 500장 / 다른 작가님들은 3 channel이여서 더 많이 필요할 수도 있음.
##### - 크기는 상관 없습니다 ( 어차피 다 Resize 할 예정이지만 크면 클 수록 좋음 )

###     < 정해야 할 것 >
#####  1. Resize할 이미지 사이즈 ( 이말년 할때 256x256으로 했다고함, 근데 메모리 터질 가능성 높음)
#####  2. 데이터 갯수 ( 많으면 많을수록 좋음 )

### 2. 모델구현 
##### - 모델 1 : StyleGAN v2 < FFHQ dataset Pre-trained Model > + FreezeD/ADA
##### - 모델 2 : U-GAT-IT

###      < 정해야 할 것 >
##### 1. 프레임워크 < TF, keras / Pytorch >
##### 2. fine-tuning
##### 3. 미세조정(FreezeD는 알겠는데 ADA는 모르곘음.. 찾아봐야 할 듯)

### 3. PPT작성
##### - 작년 수상자(https://github.com/ukiKwon/voice-separater-with-ripTracking)
##### - PPT를 어떻게 만들것인지 구상

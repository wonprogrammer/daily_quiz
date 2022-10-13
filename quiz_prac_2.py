import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
imgs = ['https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F8WEvz%2FbtrOtJaMpMF%2FXB1LNLXFCgC6h2DusXm2gK%2Fimg.jpg']  # batch of images

results = model(imgs)

print(results.xyxy[0], results.xyxy[0][0][0].item())  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)

tmp_img = cv2.imread('weather.jpeg')
cv2.rectangle(tmp_img, (int(results.xyxy[0][0][0].item()), int(results.xyxy[0][0][1].item())), (int(results.xyxy[0][0][2].item()), int(results.xyxy[0][0][3].item())), (0,255,0))
cv2.imwrite('result.png', tmp_img)
from unittest import result
import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
imgs = ['https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F8WEvz%2FbtrOtJaMpMF%2FXB1LNLXFCgC6h2DusXm2gK%2Fimg.jpg']  # batch of images

result = model(imgs)

result.save()
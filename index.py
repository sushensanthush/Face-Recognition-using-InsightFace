import cv2
from insightface.app import FaceAnalysis
import numpy


fa = FaceAnalysis(name="buffalo_l")
fa.prepare(ctx_id=0, det_size=(640, 640))

img1 = cv2.imread("person1.webp")
img2 = cv2.imread("person2.webp")

facesA = fa.get(img1)
faceA = facesA[0]

facesB = fa.get(img2)
faceB = facesB[0]

code1 = faceA.embedding
code2 = faceB.embedding

distance = numpy.linalg.norm(code1-code2)
print("Distance:", distance)
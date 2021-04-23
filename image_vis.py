import cv2
import numpy as np

for res in [16,32,64,128,256,512]:
    dir = "report_images/" + str(res) + "/"

    for img in range(1):
        input = cv2.imread("report_images/"+str(res)+".png")

        r = input[:,:,0]
        g = input[:,:,1]
        b = input[:,:,2]
   

        r = cv2.resize(r, (256, 256), interpolation=cv2.INTER_AREA )                    # Resize image
        r = cv2.applyColorMap(r, cv2.COLORMAP_HOT)
        cv2.imshow('image',r)
        cv2.waitKey(0)
        cv2.imwrite(dir + str(img) + "_density.png", r)


        g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA )                    # Resize image
        g = cv2.applyColorMap(g, cv2.COLORMAP_HOT)
        cv2.imshow('image',g)
        cv2.waitKey(0)
        cv2.imwrite(dir + str(img) + "_height.png", g)


        b = cv2.resize(b, (256, 256), interpolation=cv2.INTER_AREA)                    # Resize image
        b = cv2.applyColorMap(b, cv2.COLORMAP_HOT)
        cv2.imshow('image',b)
        cv2.waitKey(0)
        cv2.imwrite(dir + str(img) + "_heightgrad.png", b)
        print(dir + str(img) + "_heightgrad.png")




#LEGEND
w = 32
h = 256
legend = img[:,:,0]
legend = cv2.resize(legend, (32, 256))
for i in range(h):
    for j in range(w):
        legend[i,j] = 255 -i
legend = cv2.applyColorMap(legend, cv2.COLORMAP_HOT)
cv2.imwrite("report_images/legend.png", legend)
cv2.imshow('image', legend)
cv2.waitKey(0)


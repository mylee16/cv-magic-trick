import cv2
import numpy as np
from deeplab.model import Deeplabv3


# load deeplab model
deeplab_model = Deeplabv3(input_shape=(512, 512, 3))

# Using Webcam
capture = cv2.VideoCapture(0)

to_predict = True

background_img = cv2.imread('img/background.jpg')
background_img = cv2.resize(background_img, dsize=(512, 512), interpolation=cv2.INTER_AREA)
background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB).astype('uint8')

while True:

    isTrue, frame = capture.read()
    frame_resized = cv2.resize(frame, dsize=(512, 512), interpolation=cv2.INTER_AREA)

    if to_predict:
        frame_normalized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2RGB).astype('uint8')
        frame_normalized = (frame_normalized / 127.5) - 1.

        res = deeplab_model.predict(np.expand_dims(frame_normalized, 0))
        labels = np.argmax(res.squeeze(), -1)
        mask = (labels != 15).astype(np.uint8)
        mask_inverse = (labels == 15).astype(np.uint8)

        background = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)
        foreground = cv2.bitwise_and(background_img, background_img, mask=mask_inverse)
        combined = background + foreground

    if isTrue:
        if to_predict:
            cv2.imshow("Video", combined)
        else:
            cv2.imshow("Video", frame_resized)
        key = cv2.waitKey(1)

        if key == ord("q"):
            # Quit when q is pressed
            break
        elif key == ord("t"):
            # Toggle display when t is pressed
            if to_predict:
                to_predict = False
            else:
                to_predict = True

capture.release()
cv2.destroyAllWindows()

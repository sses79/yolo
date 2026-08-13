IP Webcam already runs a local server on your Android phone. For the first test, use its normal HTTP video stream:

```text
Android + IP Webcam
        │
        │ HTTP/MJPEG
        ▼
       Mac
        │
   ┌────┴─────┐
   VLC      OpenCV
              │
              ▼
          AI model
```

### Do this now

1. Make sure the Android phone and Mac are on the **same Wi-Fi**.
2. In IP Webcam, scroll to the bottom and tap **Start server**.
3. The phone should show something like:

```text
IPv4: http://192.168.1.142:8080
```

4. On your Mac, open that address in Chrome/Safari, e.g.:

```text
http://192.168.1.142:8080
```

You should get the **IP Webcam control page**.

5. On that page, under **Video renderer**, choose **Browser** to confirm the live camera works.

For OpenCV, IP Webcam normally exposes the MJPEG stream at:

```text
http://192.168.1.142:8080/video
```

So eventually our Python code can be as simple as:

```python
import cv2

url = "http://192.168.1.142:8080/video"

cap = cv2.VideoCapture(url)

while True:
    ok, frame = cap.read()

    if not ok:
        break

    cv2.imshow("Android Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break
```

**SRT/RTMP Push Streaming is for a different architecture** where the phone actively pushes video to a streaming server. We don't need that complexity for a local AI demo.

Get **`http://<phone-ip>:8080` working on your Mac first**. Once that's working, the next step is **IP Webcam → OpenCV → MediaPipe/YOLO**, which is where the project gets interesting.

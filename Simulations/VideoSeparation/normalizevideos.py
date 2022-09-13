'''
This file will normalize the videos in the Original-Videos folder to 10 second videos each
(30 FPS)
'''
import cv2
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(CURRENT_DIR, "Frames")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

def save_frames(vid, vid_name):
    current_frame = 1
    end = False
    while(current_frame<330):
        _, frame = vid.read()
        if current_frame>=30:
            name = os.path.join(data_dir, vid_name, "{}.jpg".format(current_frame-29))
            cv2.imwrite(name, frame)
        current_frame+=1

def normalize(vid_name):
    img_arr = []
    count = 1
    for files in os.listdir(os.path.join(data_dir, vid_name)):
        img = cv2.imread(os.path.join(data_dir, vid_name, "{count}.jpg".format(count)))
        height, width, layers = img.shape
        size = (width, height)
        img_arr.append(img)
        if count == 300:
            break
        count+=1
    
    out = cv2.VideoWriter(os.path.join(CURRENT_DIR, "Normalized-Videos", "{vid_name}.avi".format(vid_name)), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)

    for i in range(len(img_arr)):
        out.write(img_arr[i])
    out.release()

if __name__ == "__main__":

    v1 = cv2.VideoCapture(os.path.join(CURRENT_DIR, "Original-Videos", "SHIP.mp4"))
    v2 = cv2.VideoCapture(os.path.join(CURRENT_DIR, "Original-Videos", "GUITAR.mp4"))
    v3 = cv2.VideoCapture(os.path.join(CURRENT_DIR, "Original-Videos", "BIRD.mp4"))

    vid_names = ("SHIP", "GUITAR", "BIRD")
    vids = (v1, v2, v3)

    for i in range(3):
        if not os.path.exists(os.path.join(data_dir, vid_names[i])):
            os.mkdir(os.path.join(data_dir, vid_names[i]))
        save_frames(vids[i], vid_names[i])
        normalize(vid_names[i])

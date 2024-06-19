import dlib
import cv2
import argparse, os, random
import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from model import model_static
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from colour import Color

# parser is for adding command on the CLI and make it easier to choose option
parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, help='input video path. live cam is used when not specified')
parser.add_argument('--face', type=str, help='face detection file path. dlib face detector is used when not specified')
parser.add_argument('--model_weight', type=str, help='path to model weights file', default='data/model_weights.pkl')
parser.add_argument('--jitter', type=int, help='jitter bbox n times, and average results', default=0)
parser.add_argument('-save_vis', help='saves output as video', action='store_true')
parser.add_argument('-save_text', help='saves output as text', action='store_true')
parser.add_argument('-display_off', help='do not display frames', action='store_true')

args = parser.parse_args()
# I think I can replace this since this is just human face detector using dlib
CNN_FACE_MODEL = 'data/mmod_human_face_detector.dat' # from http://dlib.net/files/mmod_human_face_detector.dat.bz2


def bbox_jitter(bbox_left, bbox_top, bbox_right, bbox_bottom):
    cx = (bbox_right+bbox_left)/2.0
    cy = (bbox_bottom+bbox_top)/2.0
    scale = random.uniform(0.8, 1.2)
    bbox_right = (bbox_right-cx)*scale + cx
    bbox_left = (bbox_left-cx)*scale + cx
    bbox_top = (bbox_top-cy)*scale + cy
    bbox_bottom = (bbox_bottom-cy)*scale + cy
    return bbox_left, bbox_top, bbox_right, bbox_bottom


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def run(video_path, face_path, model_weight, jitter, vis, display_off, save_text):
    # set up vis settings
    red = Color("red")
    colors = list(red.range_to(Color("green"),10))
    font = ImageFont.truetype("data/arial.ttf", 40)

    # set up video source
    if video_path is None:
        cap = cv2.VideoCapture(0)
        video_path = 'live.avi'
    else:
        cap = cv2.VideoCapture(video_path)

    # set up output file
    if save_text:
        # basename basically return the last directory in a directory sequence (C://misc/projectbasically/demo.py for example basename return demo.py) and then replace it with what required
        outtext_name = os.path.basename(video_path).replace('.avi','_output.txt')
        f = open(outtext_name, "w")
    if vis:
        outvis_name = os.path.basename(video_path).replace('.avi','_output.avi')
        # cv2.VideoCapture().get() is less heavy in computation compare to image.shape since image.shape process each frame, .get get the video size from the start 
        # the number is a cv2 function but since it is enumerator, you can do it like this check this for further list of enumerator https://shorturl.at/VWall
        # 3 for width of frame in the video stream
        # 4 for height of frame in the video stream
        # 5 is for framerate
        imwidth = int(cap.get(3)); imheight = int(cap.get(4))
        # VideoWriter responsible for outputting video
        # video format have fourcc codes assigned to them by concatening four char Asscii character so you have to be specific on the quote not double quote
        # This one is motionjpeg 
        outvid = cv2.VideoWriter(outvis_name,cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(5), (imwidth,imheight))

    # set up face detection mode
    if face_path is None:
        facemode = 'DLIB'
    # else:
    #     # This code is assuming you have given it a csv file containining cooordiate for displaying the bounding box and adjust it 
    #     # Since the readme did said that the box was a bit tight
    #     facemode = 'GIVEN'
    #     column_names = ['frame', 'left', 'top', 'right', 'bottom']
    #     df = pd.read_csv(face_path, names=column_names, index_col=0)
    #     df['left'] -= (df['right']-df['left'])*0.2
    #     df['right'] += (df['right']-df['left'])*0.2
    #     df['top'] -= (df['bottom']-df['top'])*0.1
    #     df['bottom'] += (df['bottom']-df['top'])*0.1
    #     df['left'] = df['left'].astype('int')
    #     df['top'] = df['top'].astype('int')
    #     df['right'] = df['right'].astype('int')
    #     df['bottom'] = df['bottom'].astype('int')

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        exit()

    if facemode == 'DLIB':
        cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)
    frame_cnt = 0

    # set up data transformation
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load model weights
    model = model_static(model_weight)
    model_dict = model.state_dict()
    snapshot = torch.load(model_weight)
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    model.cuda()
    # model.train(False) changed here:
    model.eval()
    # video reading loop
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            height, width, channels = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_cnt += 1
            bbox = []
            if facemode == 'DLIB':
                # The 1 in the second argument indicates that we should upsample the image
                # 1 time.  This will make everything bigger and allow us to detect more
                # faces.
                dets = cnn_face_detector(frame, 1)
                """     This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
                        These objects can be accessed by simply iterating over the mmod_rectangles object
                        The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.
                        
                        It is also possible to pass a list of images to the detector.
                        - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)

                        In this case it will return a mmod_rectangless object.
                        This object behaves just like a list of lists and can be iterated over."""
                # Either way this is modifiedable though dlib returned directly the box 
                # Mediapipe do not return the coor like this but there is a way to get it
                # Welp doesn't seem like I need to care since drawing util draw detection will do it job
                for d in dets:
                    l = d.rect.left()
                    r = d.rect.right()
                    t = d.rect.top()
                    b = d.rect.bottom()
                    # expand a bit
                    l -= (r-l)*0.2
                    r += (r-l)*0.2
                    t -= (b-t)*0.2
                    b += (b-t)*0.2
                    bbox.append([l,t,r,b])
            # elif facemode == 'GIVEN':
            #     if frame_cnt in df.index:
            #         bbox.append([df.loc[frame_cnt,'left'],df.loc[frame_cnt,'top'],df.loc[frame_cnt,'right'],df.loc[frame_cnt,'bottom']])

            frame = Image.fromarray(frame)
            for b in bbox:
                face = frame.crop((b))
                img = test_transforms(face)
                img.unsqueeze_(0)
                if jitter > 0:
                    for i in range(jitter):
                        bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b[0], b[1], b[2], b[3])
                        bj = [bj_left, bj_top, bj_right, bj_bottom]
                        facej = frame.crop((bj))
                        img_jittered = test_transforms(facej)
                        img_jittered.unsqueeze_(0)
                        img = torch.cat([img, img_jittered])

                # forward pass
                output = model(img.cuda())
                if jitter > 0:
                    output = torch.mean(output, 0)
                score = torch.sigmoid(output).item()

                coloridx = min(int(round(score*10)),9)
                draw = ImageDraw.Draw(frame)
                drawrect(draw, [(b[0], b[1]), (b[2], b[3])], outline=colors[coloridx].hex, width=5)
                draw.text((b[0],b[3]), str(round(score,2)), fill=(255,255,255,128), font=font)
                if save_text:
                    f.write(f"{frame_cnt},{score}\n")

            if not display_off:
                frame = np.asarray(frame) # convert PIL image back to opencv format for faster display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow('',frame)
                if vis:
                    outvid.write(frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        else:
            break

    if vis:
        outvid.release()
    if save_text:
        f.close()
    cap.release()
    print('DONE!')


if __name__ == "__main__":
    run(args.video, args.face, args.model_weight, args.jitter, args.save_vis, args.display_off, args.save_text)

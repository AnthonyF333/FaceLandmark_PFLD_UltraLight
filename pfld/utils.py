import cv2
import numpy as np
import math
import logging
from datetime import datetime
import torch.nn as nn
import torch
import os


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def save_checkpoint(cfg, model, step=None, extra=''):
    save_path = cfg.MODEL_PATH

    if extra:
        if step is not None:
            torch.save(model.state_dict(), os.path.join(save_path, ('{}_step:{}_{}.pth'.format(cfg.MODEL_TYPE.lower(), step, extra))))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, ('{}_{}.pth'.format(cfg.MODEL_TYPE.lower(), extra))))
    else:
        if step is not None:
            torch.save(model.state_dict(), os.path.join(save_path, ('{}_step:{}.pth'.format(cfg.MODEL_TYPE.lower(), step))))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, ('{}.pth'.format(cfg.MODEL_TYPE.lower()))))


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # CRITICAL > ERROR > WARNING > INFO > DEBUG

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        stream_handler.setLevel(logging.WARNING)
        logger.addHandler(stream_handler)


def write_cfg(logging, cfg):
    for key_val in cfg:
        logging.info(str(key_val) + ': ' + str(cfg[key_val]))
    logging.info("\n")


def calculate_pitch_yaw_roll(landmarks_2D, cam_w=256, cam_h=256,
                             radians=False):
    """ Return the the pitch  yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """

    assert landmarks_2D is not None, 'landmarks_2D is None'

    # Estimated camera matrix values.
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    # dlib (68 landmark) trached points
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    # wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    # X-Y-Z with X pointing forward and Y on the left and Z up.
    # The X-Y-Z coordinates used are like the standard coordinates of ROS (robotic operative system)
    # OpenCV uses the reference usually used in computer vision:
    # X points to the right, Y down, Z to the front
    landmarks_3D = np.float32([
        [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT, 
        [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT, 
        [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
        [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
        [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
        [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
        [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
        [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
        [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
        [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
        [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
        [0.000000, -7.415691, 4.070434],  # CHIN
    ])
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)

    # Applying the PnP solver to find the 3D pose of the head from the 2D position of the landmarks.
    # retval - bool
    # rvec - Output rotation vector that, together with tvec, brings points from the world coordinate system to the camera coordinate system.
    # tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D,
                                 camera_matrix, camera_distortion)
    # Get as input the rotational vector, Return a rotational matrix

    # const double PI = 3.141592653;
    # double thetaz = atan2(r21, r11) / PI * 180;
    # double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / PI * 180;
    # double thetax = atan2(r32, r33) / PI * 180;

    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    return map(lambda k: k[0], euler_angles)  # euler_angles contain (pitch, yaw, roll)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def vis_landmark(img_path, annotation, norm, point_num):
    """
    line format: [img_name bbox_x1 bbox_y1  bbox_x2 bbox_y2 landmark_x1 landmark y1 ...]
    """
    # check point len
    assert len(line) == 1 + 4 + point_num * 2  # img_path + bbox + point_num*2

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    img_name = annotation[0]
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = annotation[1:5]
    landmark = annotation[5:]

    landmark_x = line[1 + 4::2]
    landmark_y = line[1 + 4 + 1::2]
    if norm:
        for i in range(len(landmark_x)):
            landmark_x[i] = landmark_x[i] * w
            landmark_y[i] = landmark_y[i] * h

    # draw bbox and face landmark
    cv2.rectangle(img, (int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2)), (0, 0, 255), 2)
    for i in range(len(landmark_x)):
        cv2.circle(img, (int(landmark_x[i]), int(landmark_y[i])), 2, (255, 0, 0), -1)

    cv2.imshow("image", img)
    cv2.waitKey(0)

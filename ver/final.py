import cv2
import numpy as np
import dlib
from gaze_tracking import GazeTracking
import pyautogui as pag
from math import hypot
from numpy import array
import win32com.client
import winsound

# Load sound
speaker = win32com.client.Dispatch("SAPI.SpVoice")

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

board = np.zeros((300, 1400), np.uint8)
board[:] = 255

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

first_frame = None

# Keyboard settings
keyboard = np.zeros((400, 1100, 4), np.uint8)

key_arr_1 = np.array(
    [("1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "."), ("Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "?"),
     ("A", "S", "D", "F", "G", "H", "J", "K", "L", "'"," "), ("Z", "X", "C", "V", "B", "N", "M", ",","<", "CL","")])

pts = np.array([[1020,340],[1020,360],[1040,360],[1070,390],[1070,310],[1040,340],[1020,340]],np.int32)

def direction(nose_point, anchor_point, w, h, multiple=1):
    nx = nose_point[0]
    ny = nose_point[1]
    x = anchor_point[0]
    y = anchor_point[1]

    if ny > y + multiple * h:
        return 'DOWN'
    elif ny <= y - multiple * h:
        return 'UP'

    return '-'


def letter(letter_index_i, letter_index_j, text, letter_light):
    width = 100
    height = 100
    th = 3  # thickness

    # Keys
    x = letter_index_j * width
    y = letter_index_i * height

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 5
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
        cv2.polylines(keyboard, [pts], 1, (51, 51, 51), 4)
        cv2.line(keyboard,(858,349),(888,349),(51,51,51),4)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)
        cv2.polylines(keyboard, [pts], 1, (255, 255, 255), 4)
        cv2.line(keyboard, (858, 349), (888, 349), (255,255,255), 4)

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def draw_menu():
    rows, cols, _ = keyboard.shape
    th_lines = 4  # thickness lines
    cv2.line(keyboard, (int(cols / 2) - int(th_lines / 2), 0), (int(cols / 2) - int(th_lines / 2), rows),
             (51, 51, 51), th_lines)
    cv2.putText(keyboard, "LEFT", (80, 300), font, 6, (255, 255, 255), 5)
    cv2.putText(keyboard, "RIGHT", (80 + int(cols / 2), 300), font, 6, (255, 255, 255), 5)


font = cv2.FONT_HERSHEY_PLAIN


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    if ver_line_lenght == 0:
        ver_line_lenght = 1;
    ratio = hor_line_lenght / ver_line_lenght
    return ratio


def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


# Counters
frames = 0
letter_index_i = 0
letter_index_j = 0
keyboard_selection_frames = 0
blinking_frames = 0
frames_to_blink = 6

text = ""

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    if first_frame is None:
        first_frame = frame
        left_pupil = array(gaze.pupil_left_coords())
        right_pupil = array(gaze.pupil_right_coords())
        firstpointx = (left_pupil[0] + right_pupil[0]) / 2
        firstpointy = (left_pupil[1] + right_pupil[1]) / 2
        frame_eye = array([int(firstpointx), int(firstpointy)])
        # frame_eye = array(gaze.frame_left_coords(first_frame))
        continue

    frame = gaze.annotated_frame()

    keyboard[:] = (0, 0, 0, 0)
    frames += 1

    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))

        # Gaze detection
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        drag = 12

        left_pupil = array(gaze.pupil_left_coords())
        right_pupil = array(gaze.pupil_right_coords())
        w, h = 8, 8
        dir1 = ""
        mid_point = frame_eye
        if left_pupil.size > 1 and right_pupil.size > 1:
            midpointx = (left_pupil[0] + right_pupil[0]) / 2
            midpointy = (left_pupil[1] + right_pupil[1]) / 2

            mid_point = array([int(midpointx), int(midpointy)])

        if mid_point.size > 1:
            dir1 = direction(mid_point, frame_eye, w, h)
            # cv2.line(frame, tuple(mid_point), tuple(frame_eye), (255, 0, 0), 2)
            # cv2.line(frame, (900,900), tuple(frame_eye),(255, 0, 0), 2)

        if blinking_ratio > 5.7:

            blinking_frames += 1
            frames -= 1
            active_letter = key_arr_1[letter_index_i][letter_index_j]
            keyboard_selection_frames = 0
            # Typing letter
            if blinking_frames == frames_to_blink:
                if active_letter != "<" and active_letter != "" and active_letter != "CL":
                    text += active_letter

                if active_letter == "<":
                    temp = text
                    c = text[-1:]
                    text = text[:-1]
                    cv2.putText(board, temp, (80, 100), font, 9, (255, 255, 255), 3)
                if active_letter == "CL":
                    cv2.putText(board, text, (80, 100), font, 9, (255, 255, 255), 3)
                    text = ""
                if active_letter == "":
                    speaker.Speak(text)

                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

        else:
            blinking_frames = 0
            # Show the text we're writing on the board
            cv2.putText(board, text, (80, 100), font, 9, 0, 3)

        if gaze_ratio < 0.8:
            keyboard_selection_frames += 1
            # If Kept gaze on one side more than 9 frames, move to keyboard
            if keyboard_selection_frames == 9:
                # print("Right" + str(gaze_ratio) + " " + str(blinking_ratio))
                cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
                if letter_index_j < 10 and blinking_ratio < 5:
                    letter_index_j += 1
                keyboard_selection_frames = 0

        elif gaze_ratio > 1.5:
            keyboard_selection_frames += 1
            # If Kept gaze on one side more than 9 frames, move to keyboard
            if keyboard_selection_frames == 9:
                # print("LEFT" + str(gaze_ratio) + " " + str(blinking_ratio))
                cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
                if letter_index_j > 0 and blinking_ratio < 5:
                    letter_index_j -= 1
                keyboard_selection_frames = 0
        elif gaze_ratio >= 0.8 and gaze_ratio < 1.5:

            if dir1 == 'UP':
                keyboard_selection_frames += 1
                # If Kept gaze on one side more than 9 frames, move to keyboard
                if keyboard_selection_frames == 9:
                    # print("UP" + str(gaze_ratio) + " " + str(blinking_ratio))
                    cv2.putText(frame, "UP", (50, 100), font, 2, (0, 0, 255), 3)
                    if letter_index_i > 0 and blinking_ratio < 5:
                        letter_index_i -= 1
                    keyboard_selection_frames = 0

            elif dir1 == 'DOWN':
                keyboard_selection_frames += 1
                # If Kept gaze on one side more than 9 frames, move to keyboard
                if keyboard_selection_frames == 9:
                    # print("DOWN" + str(gaze_ratio) + " " + str(blinking_ratio))
                    cv2.putText(frame, "DOWN" + str(gaze_ratio), (50, 100), font, 2, (0, 0, 255), 3)
                    if letter_index_i < 3 and blinking_ratio < 5:
                        letter_index_i += 1
                    keyboard_selection_frames = 0

        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31),
                    1)
        cv2.putText(frame, "direction: " + str(dir1), (90, 255), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    for i in range(4):
        for j in range(11):
            if i == letter_index_i and j == letter_index_j:
                light = True
            else:
                light = False
            letter(i, j, key_arr_1[i][j], light)

    cv2.imshow("Frame", frame)
    cv2.imshow("Virtual keyboard", keyboard)
    cv2.imshow("Board", board)

    key = cv2.waitKey(1)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
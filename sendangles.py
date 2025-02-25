import cv2
import mediapipe as mp
import serial


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

left_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
right_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER

cap = cv2.VideoCapture(0)

# Connect to Arduino
ser = serial.Serial('COM8', 9600)  

# Function to send pan and tilt angles to Arduino
def send_angles(pan_angle, tilt_angle):
    ser.write(f"{pan_angle} {tilt_angle}\n".encode())

def scale_to_range(value, old_min, old_max, new_min, new_max):
    return int(((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue


        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            left_shoulder_x = results.pose_landmarks.landmark[left_shoulder].x * image.shape[1]
            left_shoulder_y = results.pose_landmarks.landmark[left_shoulder].y * image.shape[0]
            right_shoulder_x = results.pose_landmarks.landmark[right_shoulder].x * image.shape[1]
            right_shoulder_y = results.pose_landmarks.landmark[right_shoulder].y * image.shape[0]

            center_x = (left_shoulder_x + right_shoulder_x) // 2
            center_y = (left_shoulder_y + right_shoulder_y) // 2

            pan_angle = scale_to_range(center_x, 60, 580, 62, 0)
            tilt_angle = scale_to_range(center_y, 50, 445, 5, 30) 

            send_angles(pan_angle, tilt_angle)

            cv2.putText(image, f"Center X: {center_x}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Center Y: {center_y}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Pan Angle: {pan_angle}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Tilt Angle: {tilt_angle}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

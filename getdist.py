import cv2
import mediapipe as mp

# Load drawing utils and pose estimation model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define shoulder landmark IDs
left_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
right_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load pose estimation model (choose a suitable model for your needs)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw pose landmarks
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

      # Get landmark coordinates
      left_shoulder_x = int(results.pose_landmarks.landmark[left_shoulder].x * image.shape[1])
      left_shoulder_y = int(results.pose_landmarks.landmark[left_shoulder].y * image.shape[0])
      right_shoulder_x = int(results.pose_landmarks.landmark[right_shoulder].x * image.shape[1])
      right_shoulder_y = int(results.pose_landmarks.landmark[right_shoulder].y * image.shape[0])

      # Calculate distance between shoulder points
      shoulder_distance = ((left_shoulder_x - right_shoulder_x) ** 2 + (left_shoulder_y - right_shoulder_y) ** 2) ** 0.5
      real_distance = 17732/shoulder_distance

      # Display distance on the frame
      cv2.putText(image, f"Shoulder Distance: {int(shoulder_distance)} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      cv2.putText(image, f"horizontal distance: {int(real_distance)} cms", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()
53
335
import cv2
import streamlit as st
import numpy as np
import pyrealsense2 as rs
import pandas  as pd
import mediapipe as mp
def main():
    st.title("BIAC AGC Project")
    video = st.empty()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    data = pd.read_csv('./sheet.csv')
    pipeline.start(config)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    upper_lip_center = tuple(
                        map(int, (face_landmarks.landmark[13].x * color_image.shape[1], face_landmarks.landmark[13].y * color_image.shape[0]))
                    )
                    lower_lip_center = tuple(
                        map(int, (face_landmarks.landmark[14].x * color_image.shape[1], face_landmarks.landmark[14].y * color_image.shape[0]))
                    )
                    cv2.circle(color_image, upper_lip_center, 5, (0, 0, 255), -1)
                    cv2.circle(color_image, lower_lip_center, 5, (0, 0, 255), -1)

                    distance = np.linalg.norm(np.array(upper_lip_center) - np.array(lower_lip_center))

                    upper_lip_depth = depth_image[upper_lip_center[1], upper_lip_center[0]]
                    lower_lip_depth = depth_image[lower_lip_center[1], lower_lip_center[0]]

                    index = (data["Point1_Depth"] - upper_lip_depth).abs().idxmin()
                    multiplication_factor = data.iloc[index]["Multiplication Factor"]
                    actual_length = distance * multiplication_factor
                    actual_length = actual_length * 10

                    text = f"Upper Lip Depth: {upper_lip_depth} mm, Lower Lip Depth: {lower_lip_depth} mm, Actual Length: {actual_length} mm"
                    cv2.putText(color_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    video.image(color_image, channels="BGR")
    except:
        pass

if __name__ == "__main__":
    main()
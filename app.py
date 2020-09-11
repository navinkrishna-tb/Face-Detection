import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

# Loading pre-trained parameters for the cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade-smile.xml')
except Exception:
	st.write("Error loading cascade classifiers")

def detect(image):

	'''
	Function to detect faces/eyes and smiles in the image passed to this function
	'''

	image = np.array(image.convert('RGB'))

	faces = face_cascade.detectMultiScale(image=image, scaleFactor = 1.3, minNeighbors = 5)


	# Draw rectangle around faces
	for (x, y, w, h) in faces:
		cv2.rectangle(img=image, pt1=(x,y), pt2=(x+w, y+h),color=(255, 0,0), thickness=2)

		roi = image[y:y+h, x:x+w]

		# Detecting eyes in the face detected
		eyes = eye_cascade.detectMultiScale(roi)

		# Detecting smile in the face(s) detected
		#smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)

		# Drawing rectangle around smile
		#for (sx,sy,sw,sh) in smile:
			#cv2.rectangle(roi, (sx,sy), (sx+sw, sy+sh), (0,0,255),2)

		return image, faces


def about():
	st.write(
		'''
		*** Har Cascade*** is an object detection algorithm.
		It can be used to detect objects in images or videos.

		The algorithm has four stages:

		1. Haar Feature Selection
		2.Creating Integral Images
		3. Adaboost Training
		4. Cascading Calssifiers
				''')

def main():
	st.title("Face Detection App : sunglasses: ")
	st.write("**Using the Haar cascade Classifiers**")
	st.set_option('deprecation.showfileUploaderEncoding', False)

	activities = ['Home', 'About']
	choice = st.sidebar.selectbox("Pick something fun", activities)

	if choice == "Home":
		st.write("Go to About Section from the sidebar to learn more about it.")

		# You can specify more file types below if you want
		image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

		if image_file is not None:

			image = Image.open(image_file)

			if st.button("Process"):

				#result_img is the image with rectangle drawn on it (in case there are faces detected)
				# result_faces is the array with co-ordinates o fbounding box(es)
				result_img, result_faces = detect(image)
				st.image(result_img, use_column_width=True)
				st.success("Found {} faces \n".format(len(result_faces)))

	elif choice == "About":
		about()


if __name__ == "__main__":
	main()
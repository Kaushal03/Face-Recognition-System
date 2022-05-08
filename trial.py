import numpy as np
import urllib
import cv2
# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

urls = [
	"https://www.google.com/url?sa=i&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FAshnoor_Kaur&psig=AOvVaw2YztyWgk_NbjjoOSmhzeCq&ust=1652082105630000&source=images&cd=vfe&ved=0CAwQjRxqFwoTCNjjusazz_cCFQAAAAAdAAAAABAD",
]
# loop over the image URLs
for url in urls:
	# download the image URL and display it
	#print "downloading %s" % (url)
	image = url_to_image(url)
	cv2.imshow("Image", image)
	cv2.waitKey(0)

import image_utils
from skimage.filters import median
from skimage.morphology import ball


def suppresds_noise(image_array):
  clean_image = median(image_array, ball(3))
  return clean_image
def detect_edges(clean_image):
  edges = edge_detection(clean_image)
  return edges
def convert_to_binary(edges, threshold):
  binary_image = (edges>threshold).astype(np.unit8)
  return binary_image
def save_binary_image(binary_image, file_name):
  edge_image = Image.fromarray(binary_image*255)
  edge_image.save(file_name)


image_array = load_image("color1.jpg")
clean_image = suppress_noise(image_array)
edges = detect_edges(clean_image)
binary_edges = convert_to_binary(edges, threshold=50)
save_binary_image(binary_edges, 'my_edges.png")

plt.imshow(binary_edges, cmap = 'gray')
plt.title('binary_edges')
plt.show()











  

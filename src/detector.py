from pathlib import Path

from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw

mtcnn = MTCNN(keep_all=True) 

image_path = Path(__file__).resolve().parent.parent / "data" / "test_images" / "images.jpg"
img = Image.open(image_path)
output = mtcnn.detect(img)

print(output)

img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)
if output[0] is not None:
    for box in output[0]:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=2)
img_draw.show()

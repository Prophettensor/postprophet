from PIL import Image, ImageFilter
import numpy as np

img = Image.open('/opt/data/cache/images/img_74163fbce1f6.jpg').convert('RGB')
arr = np.array(img)
h, w = arr.shape[0], arr.shape[1]

is_white = (arr[:, :, 0] > 100) & (arr[:, :, 1] > 100) & (arr[:, :, 2] > 100)
is_black = (arr[:, :, 0] < 50) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50)

white_mask = Image.fromarray(is_white.astype(np.uint8) * 255)
white_dilated = white_mask.filter(ImageFilter.MaxFilter(7))
white_dilated_arr = np.array(white_dilated) > 0

is_background_black = is_black & ~white_dilated_arr

rgba = np.zeros((h, w, 4), dtype=np.uint8)
rgba[:, :, :3] = arr
rgba[:, :, 3] = 255
rgba[is_background_black, 3] = 0

is_gray = ~is_white & ~is_black
gray_mask = Image.fromarray(is_gray.astype(np.uint8) * 255)
gray_near_white = np.array(gray_mask.filter(ImageFilter.MaxFilter(7))) > 0
gray_far = is_gray & ~gray_near_white & ~white_dilated_arr
rgba[gray_far, 3] = 0

result = Image.fromarray(rgba, 'RGBA')
bbox = result.getbbox()
if bbox:
    result = result.crop(bbox)
    print(f"Cropped to: {result.size}")

result.save('/opt/data/postprophet/docs/logo.png')
print("Logo saved with outline preserved")

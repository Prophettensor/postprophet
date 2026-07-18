from PIL import Image
import numpy as np
from collections import deque

img = Image.open('/opt/data/cache/images/img_74163fbce1f6.jpg').convert('RGB')
arr = np.array(img)
print(f"Image size: {img.size}")

rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
rgba[:, :, :3] = arr
rgba[:, :, 3] = 255

is_black = (arr[:, :, 0] < 30) & (arr[:, :, 1] < 30) & (arr[:, :, 2] < 30)

h, w = arr.shape[0], arr.shape[1]
visited = np.zeros((h, w), dtype=bool)
queue = deque()

for x in range(w):
    if is_black[0, x] and not visited[0, x]:
        queue.append((0, x)); visited[0, x] = True
    if is_black[h-1, x] and not visited[h-1, x]:
        queue.append((h-1, x)); visited[h-1, x] = True
for y in range(h):
    if is_black[y, 0] and not visited[y, 0]:
        queue.append((y, 0)); visited[y, 0] = True
    if is_black[y, w-1] and not visited[y, w-1]:
        queue.append((y, w-1)); visited[y, w-1] = True

while queue:
    y, x = queue.popleft()
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        ny, nx = y+dy, x+dx
        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and is_black[ny, nx]:
            visited[ny, nx] = True
            queue.append((ny, nx))

rgba[visited, 3] = 0
result = Image.fromarray(rgba, 'RGBA')

bbox = result.getbbox()
if bbox:
    result = result.crop(bbox)
    print(f"Cropped to: {result.size}")

result.save('/opt/data/postprophet/docs/logo.png')
print("Logo saved")

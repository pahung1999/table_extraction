import numpy as np
import io
from pdf2image import convert_from_bytes


def image_to_bytes(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")  # You can specify the desired image format
    image_byte_string = image_bytes.getvalue()
    image_bytes.close()
    return image_byte_string

def convert_pdf_to_images(uploaded_pdf):
    # Convert PDF to a list of PIL images
    pdf_images = convert_from_bytes(uploaded_pdf.read())
    
    # Convert PIL images to image bytes
    image_bytes_list = []
    for image in pdf_images:
        image_bytes = image_to_bytes(image)
        image_bytes_list.append(image_bytes)
    
    return image_bytes_list

def draw_polygon(image, polygon, line_width = 2, color = (0, 255, 0)):
    polygon = [(point[0], point[1]) for point in polygon]
    return image.polygon(polygon, outline=color, width=line_width)

def convert_XYXY_to_polygon(box):
    return [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]

def XY4_to_XYXY(bboxes):
    new_bboxes=[]
    for box in bboxes:
        x1=box[0][0]
        y1=box[0][1]
        x2=box[2][0]
        y2=box[2][1]
        new_bboxes.append([x1,y1,x2,y2])
    return new_bboxes

def polygon_to_XYXY(polygon):
    x_min = x_max = polygon[0][0]
    y_min = y_max = polygon[0][1]

    for point in polygon:
        x, y = point
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)

    return [x_min, y_min, x_max, y_max]

def arrange_bbox(bboxes):
    if type(bboxes[0][0]) is list:
        bboxes = [polygon_to_XYXY(polygon) for polygon in bboxes]
    n = len(bboxes)
    xcentres = [(b[0] + b[2]) // 2 for b in bboxes]
    ycentres = [(b[1] + b[3]) // 2 for b in bboxes]
    heights = [abs(b[1] - b[3]) for b in bboxes]
    width = [abs(b[2] - b[0]) for b in bboxes]

    def is_top_to(i, j):
        result = (ycentres[j] - ycentres[i]) > ((heights[i] + heights[j]) / 3)
        return result

    def is_left_to(i, j):
        return (xcentres[i] - xcentres[j]) > ((width[i] + width[j]) / 3)

    # <L-R><T-B>
    # +1: Left/Top
    # -1: Right/Bottom
    g = np.zeros((n, n), dtype='int')
    for i in range(n):
        for j in range(n):
            if is_left_to(i, j):
                g[i, j] += 10
            if is_left_to(j, i):
                g[i, j] -= 10
            if is_top_to(i, j):
                g[i, j] += 1
            if is_top_to(j, i):
                g[i, j] -= 1
    return g


def arrange_row(bboxes=None, g=None, i=None, visited=None):
    if visited is not None and i in visited:
        return []
    if g is None:
        g = arrange_bbox(bboxes)
    if i is None:
        visited = []
        rows = []
        for i in range(g.shape[0]):
            if i not in visited:
                indices = arrange_row(g=g, i=i, visited=visited)
                visited.extend(indices)
                rows.append(indices)
        return rows
    else:
        indices = [j for j in range(g.shape[0]) if j not in visited]
        indices = [j for j in indices if abs(g[i, j]) == 10 or i == j]
        indices = np.array(indices)
        g_ = g[np.ix_(indices, indices)]
        order = np.argsort(np.sum(g_, axis=1))
        indices = indices[order].tolist()
        indices = [int(i) for i in indices]
        return indices
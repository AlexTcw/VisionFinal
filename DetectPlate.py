from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import cv2
import numpy as np

# Imagen de entrada
image_path = './img/car9.jpg'
car_image = imread(image_path, as_gray=True)

# car_image = imutils.rotate(car_image, 270)  # si necesitas rotarla

gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
ax2.imshow(binary_car_image, cmap="gray")
plt.show()

label_image = measure.label(binary_car_image)
plate_dimensions = (0.03*label_image.shape[0], 0.08*label_image.shape[0], 0.15*label_image.shape[1], 0.3*label_image.shape[1])
plate_dimensions2 = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
plate_like_objects = []

fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray")
flag = 0

for region in regionprops(label_image):
    if region.area < 50:
        continue
    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        flag = 1
        plate_like_objects.append(binary_car_image[min_row:max_row, min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col, max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)

if(flag == 1):
    plt.show()
else:
    min_height, max_height, min_width, max_width = plate_dimensions2
    plate_objects_cordinates = []
    plate_like_objects = []
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_car_image, cmap="gray")
    for region in regionprops(label_image):
        if region.area < 50:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            plate_like_objects.append(binary_car_image[min_row:max_row, min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col, max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red", linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
    plt.show()


def detectar_con_opencv(image_path):
    print("Método alternativo: usando OpenCV")

    global plate_like_objects, plate_objects_cordinates

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        if len(approx) == 4:  # Forma rectangular
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 6:  # Rango típico de placas
                # Convertir la placa candidata a formato binario (como en el método original)
                candidate_gray = gray[y:y+h, x:x+w]
                _, candidate_binary = cv2.threshold(candidate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Normalizar a booleano como `binary_car_image`
                candidate_binary_bool = candidate_binary == 255

                plate_like_objects = [candidate_binary_bool]
                plate_objects_cordinates = [(y, x, y+h, x+w)]

                # Mostrar visualización (opcional)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.imshow("Candidato de placa", candidate_binary)
                cv2.imshow("Detección", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return  # Solo tomamos el primer buen candidato

    print("No se encontró ninguna placa usando OpenCV")


# Ejecutar solo si el método original falla
if len(plate_like_objects) == 0:
    detectar_con_opencv(image_path)

# Si aún no se detecta nada, usar imagen completa
if len(plate_like_objects) == 0:
    print("No se detectó ninguna placa. Usando imagen completa como fallback.")
    plate_like_objects = [binary_car_image]
    plate_objects_cordinates = [(0, 0, binary_car_image.shape[0], binary_car_image.shape[1])]

    # Visualización opcional
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_car_image, cmap="gray")
    rectBorder = patches.Rectangle((0, 0), binary_car_image.shape[1], binary_car_image.shape[0], edgecolor="blue", linewidth=2, fill=False)
    ax1.add_patch(rectBorder)
    plt.title("Usando imagen completa como fallback")
    plt.show()

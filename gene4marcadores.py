import numpy as np
import argparse
from utils import ARUCO_DICT
import cv2
import sys

# Configurar el analizador de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output folder to save ArUCo tags")
ap.add_argument("-i", "--ids", type=int, nargs="+", required=True, help="IDs of ArUCo tags to generate")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to generate")
ap.add_argument("-s", "--size", type=int, default=200, help="Size of the ArUCo tags")
args = vars(ap.parse_args())

# Verificar si el diccionario es compatible
if ARUCO_DICT.get(args["type"], None) is None:
    print(f"ArUCo tag type '{args['type']}' is not supported")
    sys.exit(0)

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])

# Iterar sobre los IDs y generar los marcadores
for marker_id in args["ids"]:
    print(f"Generating ArUCo tag of type '{args['type']}' with ID '{marker_id}'")
    tag_size = args["size"]
    tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
    cv2.aruco.drawMarker(arucoDict, marker_id, tag_size, tag, 1)

    # Guardar la etiqueta generada
    tag_name = f'{args["output"]}/{args["type"]}_id_{marker_id}.png'
    cv2.imwrite(tag_name, tag)

    # Mostrar el marcador generado
    cv2.imshow(f"ArUCo Tag {marker_id}", tag)

# Esperar hasta que se presione una tecla para cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()

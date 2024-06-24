import tkinter as tk
import cv2
from tkinter import filedialog
from utils import ARUCO_DICT, aruco_display

# Definir las rutas de las imágenes de superposición y sus respectivos IDs
superimpose_data = {
    87: "Images/dino.png",
    42: "Images/mickey.png",
    0: "Images/Hamburguesa.png"
}

# Variable global para almacenar el tipo de ArUco seleccionado
aruco_type_var = None

def ejecutar():
    global aruco_type_var
    path = filedialog.askopenfilename()
    if not path:
        return
    
    print(f"Cargando imagen: {path}")
    image = cv2.imread(path)
    if image is None:
        print(f"No se pudo cargar la imagen desde {path}")
        return

    h, w, _ = image.shape
    width = 600
    height = int(width * (h / w))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    aruco_type = aruco_type_var.get()
    if ARUCO_DICT.get(aruco_type, None) is None:
        print(f"Tipo de tag ArUco '{aruco_type}' no es soportado")
        return

    print(f"Detectando tags '{aruco_type}'....")
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    # Cargar las imágenes de superposición
    superimpose_images = {}
    for marker_id, img_path in superimpose_data.items():
        superimpose_img = cv2.imread(img_path)
        if superimpose_img is not None:
            superimpose_images[marker_id] = superimpose_img
        else:
            print(f"No se pudo cargar la imagen de superposición desde {img_path}")

    detected_markers = aruco_display(corners, ids, rejected, image, superimpose_images)
    cv2.imshow("Image", detected_markers)
    cv2.waitKey(0)

def ejecutar_con_camara():
    global aruco_type_var
    aruco_type = aruco_type_var.get()
    if ARUCO_DICT.get(aruco_type, None) is None:
        print(f"Tipo de tag ArUco '{aruco_type}' no es soportado")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    print(f"Detectando tags '{aruco_type}'....")
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters_create()

    # Cargar las imágenes de superposición
    superimpose_images = {}
    for marker_id, img_path in superimpose_data.items():
        superimpose_img = cv2.imread(img_path)
        if superimpose_img is not None:
            superimpose_images[marker_id] = superimpose_img
        else:
            print(f"No se pudo cargar la imagen de superposición desde {img_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se puede recibir frame (stream end?). Saliendo ...")
            break

        corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        frame = aruco_display(corners, ids, rejected, frame, superimpose_images)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Función para iniciar la aplicación gráfica
def iniciar_aplicacion():
    global aruco_type_var
    root = tk.Tk()
    root.title("Detector de Marcadores ArUco")

    # Crear el marco principal
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    # Etiqueta de bienvenida al sistema
    tk.Label(frame, text="Bienvenido al sistema de Realidad Aumentada aplicado a marcadores ArUco",
             font=("Arial", 16), pady=20).grid(row=0, column=0, columnspan=2)

    # Etiqueta y menú desplegable para seleccionar el tipo de ArUco
    tk.Label(frame, text="Seleccione tipo de ArUco:").grid(row=1, column=0, padx=5, pady=5)
    aruco_types = list(ARUCO_DICT.keys())
    aruco_type_var = tk.StringVar(root)
    aruco_type_var.set(aruco_types[0])
    dropdown = tk.OptionMenu(frame, aruco_type_var, *aruco_types)
    dropdown.grid(row=1, column=1, padx=5, pady=5)

    # Botones para ejecutar desde imagen y desde cámara
    btn_ejecutar = tk.Button(frame, text="Detectar marcador desde una imágen", command=ejecutar)
    btn_ejecutar.grid(row=2, column=0, padx=5, pady=10)

    btn_camara = tk.Button(frame, text="Detectar marcador en tiempo real", command=ejecutar_con_camara)
    btn_camara.grid(row=2, column=1, padx=5, pady=10)

    # Iniciar la interfaz gráfica
    root.mainloop()

# Llamar a la función para iniciar la aplicación
iniciar_aplicacion()

import pickle
import SegmentCharacters
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import cv2
import numpy as np

def recognize_license_plate(model_path='./finalized_model.sav'):
    """
    Procesa los caracteres segmentados y devuelve el texto reconocido de la placa.
    Si falla, intenta reconocimiento usando OCR (Tesseract).
    
    Returns:
        str: Texto reconocido de la placa.
    """
    print("Loading model...")
    model = pickle.load(open(model_path, 'rb'))
    print("Model loaded. Predicting characters...")

    classification_result = []

    if SegmentCharacters.characters:
        for character in SegmentCharacters.characters:
            character = character.reshape(1, -1)
            result = model.predict(character)
            classification_result.append(result)

        # Reconstrucción y orden
        plate_string = ''.join([pred[0] for pred in classification_result])
        column_list_copy = SegmentCharacters.column_list[:]
        sorted_indices = sorted(range(len(column_list_copy)), key=lambda k: column_list_copy[k])
        rightplate_string = ''.join([plate_string[i] for i in sorted_indices])

        if rightplate_string.strip():
            return rightplate_string

    # Si no se reconocieron caracteres, intentar OCR
    print("El modelo no logró reconocer caracteres. Probando con OCR...")

    # Convertir placa a formato compatible con OCR (por ejemplo la imagen completa)
    if hasattr(SegmentCharacters, 'license_plate') and SegmentCharacters.license_plate is not None:
        # Asegúrate de que sea imagen uint8 (requerido por Tesseract)
        license_plate_img = SegmentCharacters.license_plate.astype(np.uint8) * 255

        # Redimensionar para mejorar OCR
        license_plate_img = cv2.resize(license_plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        ocr_result = pytesseract.image_to_string(license_plate_img, config='--psm 7')
        ocr_result = ''.join(filter(str.isalnum, ocr_result))  # limpia ruido
        print("OCR result:", ocr_result)

        return ocr_result.strip()

    print("No se pudo aplicar OCR porque no hay imagen de placa disponible.")
    return ""
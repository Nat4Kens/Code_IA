import torch
import torchvision
import cv2
import numpy as np
import pygame
from pygame.locals import *

# Initialiser Pygame
pygame.init()

# Sélectionner la webcam avec OpenCV
cap = cv2.VideoCapture(0)
# Obtenir la résolution de la webcam
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Définir la résolution de l'écran pour qu'elle corresponde à la résolution de la webcam
RESOLUTION = (width, height)

# Créer une fenêtre Pygame en plein écran
screen = pygame.display.set_mode(RESOLUTION, pygame.FULLSCREEN)
pygame.display.set_caption("Flux vidéo en direct")

# Charger le modèle Faster R-CNN pré-entraîné
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Boucle principale
running = True
with torch.no_grad():
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        
        # Capturer une image depuis la webcam avec OpenCV
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir l'image en format RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transposer les dimensions de l'image pour PyTorch
        img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Ajouter une dimension de lot
        
        # Faire la détection d'objets
        predictions = model(img_tensor)
        
        # Filtrer les prédictions pour les humains uniquement
        human_class_index = 1  # Indice de la classe humaine dans le modèle Faster R-CNN
        human_indices = [i for i, label in enumerate(predictions[0]['labels']) if label == human_class_index]
        
        seuil_confiance = 0.75
        # Sélectionner la prédiction la plus confiante parmi les humains détectés
        if len(human_indices) > 0:
            human_confidences = predictions[0]['scores'][human_indices].detach().numpy()
            max_confidence_index = np.argmax(human_confidences)
            max_confidence = human_confidences[max_confidence_index]
            if max_confidence > seuil_confiance:
                # Dessiner la boîte englobante du seul humain détecté avec une grande confiance
                human_box = predictions[0]['boxes'][human_indices[max_confidence_index]].detach().numpy()
                x, y, w, h = map(int, human_box)
                viseur = [x+w/2, y+h/2]
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Convertir l'image pour Pygame
        img_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        
        # Afficher l'image dans la fenêtre Pygame
        screen.blit(img_surface, (0, 0))
        pygame.display.flip()

# Libérer la capture de la webcam et quitter Pygame
cap.release()
pygame.quit()

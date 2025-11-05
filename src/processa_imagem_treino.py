import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords_translated = coords - wrist
    max_dist = np.max(np.linalg.norm(coords_translated, axis=1))
    if max_dist == 0:
        return None
    coords_normalized = coords_translated / max_dist
    return coords_normalized.flatten().tolist()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

CAMINHO_DATASET = r'C:\Users\mathe\Documents\TopEsp\LIA_alfabeto\dataset_libras\train' 
ARQUIVO_SAIDA = 'alfabeto_landmarks_treino.csv'

all_data = []

print("Iniciando processamento do dataset de imagens (lendo nomes de arquivos)...")

arquivos_de_imagem = [f for f in os.listdir(CAMINHO_DATASET) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in tqdm(arquivos_de_imagem, desc="Processando Imagens"):
    
    classe = img_name[0].upper()
    img_path = os.path.join(CAMINHO_DATASET, img_name)
    
    try:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Aviso: Não foi possível ler a imagem {img_path}")
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            pontos = normalize_landmarks(hand_landmarks)
            
            if pontos:
                linha = [classe] + pontos
                all_data.append(linha)
                
    except Exception as e:
        print(f"Erro ao processar {img_path}: {e}")

hands.close()

if not all_data:
    print("Nenhum dado foi extraído! Verifique o CAMINHO_DATASET.")
else:
    colunas = ['class'] + [f'p{i}' for i in range(63)]
    df_final = pd.DataFrame(all_data, columns=colunas)
    df_final.to_csv(ARQUIVO_SAIDA, index=False)
    
    print(f"\nProcessamento concluído! {len(all_data)} amostras salvas em '{ARQUIVO_SAIDA}'.")
    print("\nAmostras coletadas por classe:")
    print(df_final['class'].value_counts().sort_index())
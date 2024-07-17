import cv2
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
import pandas as pd

# Transforma a imagem colorida em tons de cinza seja em 256 ou 16 tons
def get_grayscale_image(colored_img, is_sixteen_shades):
        gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)

        if(is_sixteen_shades):
            gray_img = (gray_img//16)*16
        
        return gray_img

# Cria a matriz de co-ocorrencia de acordo com o valor de i selecionado na interface
def get_coocurrence_matrix(i, colored_img):
        h, w = colored_img.shape[:2]
        
        gray_img = get_grayscale_image(colored_img, True)
        
        coocurrence_matrix = [[0 for _ in range(256)] for _ in range(256)]
        
        gray_img_matrix = np.reshape(gray_img.flatten(), (h, w))

        h = len(gray_img_matrix)
        w = len(gray_img_matrix[0])
                    
        for lin in range(h - i):
            for col in range(w - i):
                coocurrence_matrix[gray_img_matrix[lin][col]][gray_img_matrix[lin + i][col + i]] += 1
                        
        return coocurrence_matrix

# Calcula os descritores de Haralick
def get_haralick_features(i, current_image):
        coocurrence_matrix = get_coocurrence_matrix(i, current_image)
        
        # Normaliza a matriz de co-ocorrência para obter probabilidades
        normalized_matrix = coocurrence_matrix / np.sum(coocurrence_matrix)
        
        # Inicializa variáveis para os descritores
        entropy = 0
        homogeneity = 0
        contrast = 0

        # Dimensão da matriz de co-ocorrência
        size = normalized_matrix.shape[0]
        
        for i in range(size):
            for j in range(size):
                p_ij = normalized_matrix[i, j]
                
                # Cálculo da entropia: soma -p_ij * log2(p_ij) para cada elemento
                if p_ij > 0:
                    entropy -= p_ij * np.log2(p_ij)
                
                # Cálculo da homogeneidade: soma p_ij / (1 + |i - j|) para cada elemento
                homogeneity += p_ij / (1 + abs(i - j))
                
                # Cálculo do contraste: soma p_ij * (i - j)^2 para cada elemento
                contrast += p_ij * (i - j) ** 2
        
        return entropy, homogeneity, contrast

def get_hu_moments(gray_img, h_channel, s_channel, v_channel):
        # Calcular momentos invariantes de Hu para a imagem em tons de cinza
        moments = cv2.moments(gray_img)
        hu_moments_gray = cv2.HuMoments(moments).flatten()

        # Calcular momentos invariantes de Hu para cada canal HSV
        moments = cv2.moments(h_channel)
        hu_moments_h = cv2.HuMoments(moments).flatten()
        moments = cv2.moments(s_channel)
        hu_moments_s = cv2.HuMoments(moments).flatten()
        moments = cv2.moments(v_channel)
        hu_moments_v = cv2.HuMoments(moments).flatten()
        
        return hu_moments_gray, hu_moments_h, hu_moments_s, hu_moments_v

# Calcula os momentos de Hu para a imagem em tons de cinza e para os canais H, S e V
def get_hu_moments_menu(colored_img):
        # Converter para tons de cinza
        gray_img = get_grayscale_image(colored_img, False)

        # Converter para HSV
        hsv_image = cv2.cvtColor(colored_img, cv2.COLOR_BGR2HSV)

        # Separar canais HSV
        h_channel, s_channel, v_channel = cv2.split(hsv_image)
        
        # Definir o valor de corte para a binarização
        threshold_h = 130
        threshold_s = 62
        threshold_v = 62
        threshold_g = 75
        
        # Binarização de todos os canais a partir do limiar definido acima
        _,h_channel_bin = cv2.threshold(h_channel, threshold_h, 255, cv2.THRESH_BINARY)
        _,s_channel_bin = cv2.threshold(s_channel, threshold_s, 255, cv2.THRESH_BINARY)
        _,v_channel_bin = cv2.threshold(v_channel, threshold_v, 255, cv2.THRESH_BINARY)
        _,gray_img_bin = cv2.threshold(gray_img, threshold_g, 255, cv2.THRESH_BINARY)
        
        # Calcula os momentos de hu
        hu_moments_gray, hu_moments_h, hu_moments_s, hu_moments_v = get_hu_moments(gray_img_bin, h_channel_bin, s_channel_bin, v_channel_bin)

        return hu_moments_gray, hu_moments_h, hu_moments_s, hu_moments_v

# Calcula os momentos de Hu com a alicação de alguns filtros para o modelo raso SVM
def get_hu_moments_model(colored_img):
        # Converter para tons de cinza
        gray_img = get_grayscale_image(colored_img, False)

        # Converter para HSV
        hsv_image = cv2.cvtColor(colored_img, cv2.COLOR_BGR2HSV)

        # Separar canais HSV
        h_channel, s_channel, v_channel = cv2.split(hsv_image)
        
        # Aplica o filtro Gaussiano em todos os canais da imagem
        gray_img_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        h_channel_blur = cv2.GaussianBlur(h_channel, (5, 5), 0)
        s_channel_blur = cv2.GaussianBlur(s_channel, (5, 5), 0)
        v_channel_blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
        
        # Aplica o filtro média na imagem em tons de cinza e no canal Value para diminuir mais ainda o ruído preservando as bordas
        gray_img_blur = cv2.medianBlur(gray_img_blur, 5)
        v_channel_blur = cv2.medianBlur(v_channel_blur, 5)
        
        # Definir o valor de corte para a binarização
        threshold_g = 75
        threshold_v = 62
        
        # Binarização da imagem em tons de cinza e do canal Value a partir do limiar definido acima
        _,gray_img_bin = cv2.threshold(gray_img_blur, threshold_g, 255, cv2.THRESH_BINARY)
        _,v_channel_bin = cv2.threshold(v_channel_blur, threshold_v, 255, cv2.THRESH_BINARY)
        
        # Aplica o filtro filtro Canny Edges para identificar as bordas no canal Hue
        h_channel_edges = cv2.Canny(h_channel_blur, 50, 150, apertureSize=7)
        s_channel_edges = cv2.Canny(s_channel_blur, 50, 150, apertureSize=3)
        
        # Calcula os momentos de hu
        hu_moments_gray, hu_moments_h, hu_moments_s, hu_moments_v = get_hu_moments(gray_img_bin, h_channel_edges, s_channel_edges, v_channel_bin)
        
        return hu_moments_gray, hu_moments_h, hu_moments_s, hu_moments_v

# Calcula os momentos de Hu com a alicação de alguns filtros para o modelo raso binário SVM
def get_hu_moments_model_bin(colored_img):
        # Converter para tons de cinza
        gray_img = get_grayscale_image(colored_img, False)

        # Converter para HSV
        hsv_image = cv2.cvtColor(colored_img, cv2.COLOR_BGR2HSV)

        # Separar canais HSV
        h_channel, s_channel, v_channel = cv2.split(hsv_image)
        
        # Aplica o filtro média na imagem em tons de cinza (remover ruído preservando as bordas)
        gray_img_m_blur = cv2.medianBlur(gray_img, 5)
        
        # Aplica o filtro Gaussiano seguido do filtro Canny Edges para identificar as bordas no canal Hue
        h_channel_blur = cv2.GaussianBlur(h_channel, (5, 5), 0)
        h_channel_edges = cv2.Canny(h_channel_blur, 50, 150, apertureSize=7)
        
        # Definir o valor de corte para a binarização
        threshold_g = 75
        threshold_s = 62
        threshold_v = 62
        
        # Binarização da imagem em tons de cinza e dos canais Saturation e Value a partir do limiar definido acima
        _,gray_img_bin = cv2.threshold(gray_img_m_blur, threshold_g, 255, cv2.THRESH_BINARY)
        _,s_channel_bin = cv2.threshold(s_channel, threshold_s, 255, cv2.THRESH_BINARY)
        _,v_channel_bin = cv2.threshold(v_channel, threshold_v, 255, cv2.THRESH_BINARY)
        
        # Calcula os momentos de hu
        hu_moments_gray, hu_moments_h, hu_moments_s, hu_moments_v = get_hu_moments(gray_img_bin, h_channel_edges, s_channel_bin, v_channel_bin)
        
        return hu_moments_gray, hu_moments_h, hu_moments_s, hu_moments_v

# Classifica a imagem de acordo com o modelo raso SVM
def classify_image_SVM(current_img, svm):
        hu_values = {"Hu_gray":[],
                     "Hu_H":[],
                     "Hu_S":[],
                     "Hu_V":[]}
        hu_gray, hu_h, hu_s, hu_v = get_hu_moments_model_bin(current_img)
        hu_values["Hu_gray"].append(hu_gray)
        hu_values["Hu_H"].append(hu_h)
        hu_values["Hu_S"].append(hu_s)
        hu_values["Hu_V"].append(hu_v)

        df_model = pd.DataFrame(hu_values)
        for col in [i for i in df_model.columns if "Hu" in i]:
                n_cols = [col+str(j) for j in range(1, 8)]
                df_model[n_cols] = df_model[col].tolist()
                df_model = df_model.drop(columns=col)
        
        predicted_class = svm.predict(df_model)[0]
        
        return predicted_class

# Classifica a imagem de acordo com o modelo raso binário SVM
def classify_image_SVM_bin(current_img, svm_bin):
        hu_moments = get_hu_moments_table(current_img)
        
        predicted_class = 'Doente' if svm_bin.predict(hu_moments)[0] == "1" else 'Não Doente'
        
        return predicted_class

# Classifica a imagem de acordo com o modelo prfundo ResNet50
def classify_image_ResNet(image_path, ResNet50):
        labels = ['ASC-H', 'ASC-US', 'HSIL', 'LSIL', 'NFIL', 'SCC']
        # Carregar a imagem
        img = image.load_img(image_path, target_size=(224, 224))
        # Converter a imagem para um array numpy
        img_array = image.img_to_array(img)
        # Expandir as dimensões da imagem para (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocessar a imagem da mesma forma que durante o treino
        img_array = preprocess_input(img_array)
        
        # Fazer a previsão
        predictions = ResNet50.predict(img_array)
        # Interpretar a previsão
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = labels[predicted_class_index]
        
        return predicted_class

# Classifica a imagem de acordo com o modelo prfundo binário ResNet50
def classify_image_ResNet_bin(image_path, ResNet50_bin):  
        # Carregar a imagem
        img = image.load_img(image_path, target_size=(224, 224))
        # Converter a imagem para um array numpy
        img_array = image.img_to_array(img)
        # Expandir as dimensões da imagem para (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocessar a imagem da mesma forma que durante o treino
        img_array = preprocess_input(img_array)
        
        predictions = ResNet50_bin.predict(img_array)
        print(predictions)

        predicted_class = 'Doente' if predictions[0][0] > 0.5 else 'Não Doente'
        
        return predicted_class

# Formata os momentos de Hu da imagem para classificação pelo modelo raso SVM e modelo raso binário SVM
def get_hu_moments_table(current_img):
        hu_values = {"Hu_gray":[],
                     "Hu_H":[],
                     "Hu_S":[],
                     "Hu_V":[]}
        hu_gray, hu_h, hu_s, hu_v = get_hu_moments_model_bin(current_img)
        hu_values["Hu_gray"].append(hu_gray)
        hu_values["Hu_H"].append(hu_h)
        hu_values["Hu_S"].append(hu_s)
        hu_values["Hu_V"].append(hu_v)

        df_model = pd.DataFrame(hu_values)
        for col in [i for i in df_model.columns if "Hu" in i]:
                n_cols = [col+str(j) for j in range(1, 8)]
                df_model[n_cols] = df_model[col].tolist()
                df_model = df_model.drop(columns=col)
                
        return df_model
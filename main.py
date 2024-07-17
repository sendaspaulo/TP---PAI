import image_operations
import os
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model # type: ignore
import joblib

# Carregar o modelo raso SVM
svm = joblib.load("src/modelo_raso.pkl")
# Carregar o modelo raso SVM
svm_bin = joblib.load("src/modelo_raso_binario.pkl")
# Carregar o modelo profundo binário ResNet50
ResNet50_bin = load_model('src/modelo_profundo_binario.keras')
# Carregar o modelo profundo ResNet50
ResNet50 = load_model('src/modelo_profundo.h5')

class Menu:
    def __init__(self, root):
        self.root = root
        self.root.title("Menu")
        self.root.geometry("1150x710")

        self.subdirectories = ["ASC-US", "ASC-H", "LSIL", "HSIL", "SCC", "Negative for intraepithelial lesion"]
        self.current_subdirectory = tk.StringVar(value=self.subdirectories[0])
        self.image_files = []
        self.current_image_index = 0
        self.zoom_value = 5
        self.is_searched_image = False
        self.searched_image_path = ""
        self.is_gray = False
        self.i = 1

        self.create_widgets()
        self.load_images(self.current_subdirectory.get())

    def create_widgets(self):
        # Mostra a imagem selecionada
        self.panel = ttk.Label(self.root)
        self.panel.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Cria o frame que comporta os botões da parte inferior da imagem
        self.bot_button_frame = ttk.Frame(self.root)
        self.bot_button_frame.pack(side="bottom", fill="x")
        
        # Cria o frame que comporta os botões da parte superior da imagem
        self.top_button_frame = ttk.Frame(self.root)
        self.top_button_frame.pack(side="top", fill="x")
        
        # Botão para passar para a próxima imagem
        self.next_button = ttk.Button(self.bot_button_frame, text="Próxima", command=self.show_next_image)
        self.next_button.pack(side="right")

        # Botão para passar para a imagem anterior
        self.prev_button = ttk.Button(self.bot_button_frame, text="Anterior", command=self.show_prev_image)
        self.prev_button.pack(side="right")
        
        # Menu para selecionar a classe da imagem
        self.subdir_menu = ttk.OptionMenu(self.bot_button_frame, self.current_subdirectory, *self.subdirectories, command=self.update_images)
        self.subdir_menu.pack(side="right")
        
        # Barra para pesquisar imagem da classe selecionada pelo nome
        self.search_entry = ttk.Entry(self.bot_button_frame)
        self.search_entry.pack(side="right")
        
        # Botão para confirmar pesquisa pelo nome da imagem digitada na barra de pesquisa
        self.search_button = ttk.Button(self.bot_button_frame, text="🔍", command=self.show_searched_image)
        self.search_button.pack(side="right")
        
        # Botão para aumentar o zoom da imagem
        self.zoom_button = ttk.Button(self.bot_button_frame, text="Zoom-out", command=self.toggle_zoom_out)
        self.zoom_button.pack(side="left")
        
        # Botão para reduzir o zoom da imagem
        self.zoom_button = ttk.Button(self.bot_button_frame, text="Zoom-in", command=self.toggle_zoom_in)
        self.zoom_button.pack(side="left")
                
        # Botão para classificar a imagem de acordo com o classificador raso SVM
        self.svm = ttk.Button(self.bot_button_frame, text="SVM", command=self.show_predicted_class_SVM)
        self.svm.pack(side="left")
        
        # Botão para classificar a imagem de acordo com o classificador raso binário SVM
        self.svm = ttk.Button(self.bot_button_frame, text="SVM binário", command=self.show_predicted_class_SVM_bin)
        self.svm.pack(side="left")
        
        # Botão para classificar a imagem de acordo com o classificador profundo ResNet50
        self.resnet = ttk.Button(self.bot_button_frame, text="ResNet50", command=self.show_predicted_class_ResNet)
        self.resnet.pack(side="left")
        
        # Botão para classificar a imagem de acordo com o classificador profundo binário ResNet50
        self.resnet_bin = ttk.Button(self.bot_button_frame, text="ResNet50 binário", command=self.show_predicted_class_ResNet_bin)
        self.resnet_bin.pack(side="left")
        
        # Classificação gerada pelos modelos acionados pelos botões acima
        self.classification = tk.Label(self.bot_button_frame, text="Classe: ")
        self.classification.pack(side="left", anchor="w")
        
        # Botão para aplicar tons de cinza à imagem
        self.grayscale_button = ttk.Button(self.top_button_frame, text="Tons de Cinza(256)", command=self.load_grayscale_image_256)
        self.grayscale_button.pack(side="left")
        
        # Botão para aplicar 16 tons de cinza à imagem
        self.grayscale_button = ttk.Button(self.top_button_frame, text="Tons de Cinza(16)", command=self.load_grayscale_image_16)
        self.grayscale_button.pack(side="left")
        
        # Botão para mostrar histogramas
        self.histogram_button = ttk.Button(self.top_button_frame, text="Histogramas", command=self.show_histograms)
        self.histogram_button.pack(side="left")
        
        # Botão para mostrar as características de Haralick
        self.haralick_button_32 = ttk.Button(self.top_button_frame, text="Descritores Haralick", command=self.show_haralick_features)
        self.haralick_button_32.pack(side="left")
        
        # Muda o valor de i para 1
        self.haralick_button_1 = ttk.Button(self.top_button_frame, text="(i=1)", command=self.set_i_1)
        self.haralick_button_1.pack(side="left")
        
        # Muda o valor de i para 2
        self.haralick_button_2 = ttk.Button(self.top_button_frame, text="(i=2)", command=self.set_i_2)
        self.haralick_button_2.pack(side="left")
        
        # Muda o valor de i para 4
        self.haralick_button_4 = ttk.Button(self.top_button_frame, text="(i=4)", command=self.set_i_4)
        self.haralick_button_4.pack(side="left")
        
        # Muda o valor de i para 8
        self.haralick_button_8 = ttk.Button(self.top_button_frame, text="(i=8)", command=self.set_i_8)
        self.haralick_button_8.pack(side="left")
        
        # Muda o valor de i para 16
        self.haralick_button_16 = ttk.Button(self.top_button_frame, text="(i=16)", command=self.set_i_16)
        self.haralick_button_16.pack(side="left")
        
        # Muda o valor de i para 32
        self.haralick_button_32 = ttk.Button(self.top_button_frame, text="(i=32)", command=self.set_i_32)
        self.haralick_button_32.pack(side="left")
        
        # Valores de Haralick na tela
        self.entropyLabel = tk.Label(root, text="")
        self.entropyLabel.pack(side="top", anchor="w")
        self.homogeneityLabel = tk.Label(root, text="")
        self.homogeneityLabel.pack(side="top", anchor="w")
        self.contrastLabel = tk.Label(root, text="")
        self.contrastLabel.pack(side="top", anchor="w")
        
        # Botão para aplicar os Contornos gerados pelos Momentos de Hu à imagem e mostrar os valores na tela
        self.hu_button = ttk.Button(self.top_button_frame, text="Momentos de Hu", command=self.show_hu_moments)
        self.hu_button.pack(side="left")
        
        # Valores dos Momentos de Hu na tela
        self.grayImageHuLabel = tk.Label(root, text="")
        self.grayImageHuLabel.pack(side="top", anchor="w")
        self.hHuLabel = tk.Label(root, text="")
        self.hHuLabel.pack(side="top", anchor="w")
        self.sHuLabel = tk.Label(root, text="")
        self.sHuLabel.pack(side="top", anchor="w")
        self.vHuLabel = tk.Label(root, text="")
        self.vHuLabel.pack(side="top", anchor="w")
        
        
    # Carrega as imagens da pasta selecionada no menu
    def load_images(self, subdirectory):
        self.image_files = []
        subdir_path = os.path.join("src/", subdirectory)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('.png', '.jpg')):
                    self.image_files.append(os.path.join(subdir_path, file))
        self.image_files.sort()
        self.current_image_index = 0
        if self.image_files:
            self.load_image(cv2.imread(self.image_files[self.current_image_index]), False)
        else:
            self.panel.config(image='', text="Nenhuma imagem encontrada")
    
    # Mostra a imagem a partir do número e pasta pesquisados
    def show_searched_image(self):
        subdirectory = self.current_subdirectory.get()
        file_path = f"src/{subdirectory}/{self.search_entry.get()}"
        if os.path.isfile(file_path+".png"):
            self.is_searched_image = True
            self.searched_image_path = file_path+".png"
            self.load_image(cv2.imread(self.searched_image_path), False)
        elif os.path.isfile(file_path+".jpg"):
            self.is_searched_image = True
            self.searched_image_path = file_path+".jpg"
            self.load_image(cv2.imread(self.searched_image_path), False)
        else:
            self.panel.config(image='', text="Nenhuma imagem encontrada")
        
    # Mostra a imagem recebida no menu
    def load_image(self, img, is_gray):
        self.clear_values()
        height, width = img.shape[:2]
        img = cv2.resize(img, (int(width * self.zoom_value), int(height * self.zoom_value)), interpolation=cv2.INTER_LINEAR)
        if not is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.panel.config(image=img)
        self.panel.image = img

    # Mostra a imagem anterior da pasta selecionada
    def show_prev_image(self):
        if self.current_image_index > 0:
            self.is_searched_image = False
            self.current_image_index -= 1
            self.load_image(cv2.imread(self.image_files[self.current_image_index]), False)

    # Mostra a imagem seguinte da pasta selecionada
    def show_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.is_searched_image = False
            self.current_image_index += 1
            self.load_image(cv2.imread(self.image_files[self.current_image_index]), False)
    
    # Aplica zoom na imagem
    def toggle_zoom_in(self):
        if self.zoom_value < 7:
            self.zoom_value += 2
            if self.is_searched_image:
                self.show_searched_image()
            else:
                self.load_image(cv2.imread(self.image_files[self.current_image_index]), False)
    
    # Retira zoom da imagem
    def toggle_zoom_out(self):
        if self.zoom_value > 1:
            self.zoom_value -= 2
            if self.is_searched_image:
                self.show_searched_image()
            else:
                self.load_image(cv2.imread(self.image_files[self.current_image_index]), False)
    
    # Mostra a predição de classe da imagem mostrada na tela pelo modelo raso SVM
    def show_predicted_class_SVM(self):
        predicted_class = image_operations.classify_image_SVM(self.get_current_image(), svm)
        self.classification.config(text=f"Classe: {predicted_class}")
    
    # Mostra a predição de classe da imagem mostrada na tela pelo modelo raso binário SVM
    def show_predicted_class_SVM_bin(self):
        predicted_class = image_operations.classify_image_SVM_bin(self.get_current_image(), svm_bin)
        self.classification.config(text=f"Classe: {predicted_class}")
    
    # Mostra a predição de classe da imagem mostrada na tela pelo modelo profundo ResNet50
    def show_predicted_class_ResNet(self):
        predicted_class = image_operations.classify_image_ResNet(self.get_current_image_path(), ResNet50)
        self.classification.config(text=f"Classe: {predicted_class}")
    
    # Mostra a predição de classe da imagem mostrada na tela pelo modelo profundo binário ResNet50
    def show_predicted_class_ResNet_bin(self):
        predicted_class = image_operations.classify_image_ResNet_bin(self.get_current_image_path(), ResNet50_bin)
        self.classification.config(text=f"Classe: {predicted_class}")

    # Carrega as imagens da pasta de classe selecionada no menu
    def update_images(self, selected_subdir):
        self.is_searched_image = False
        self.load_images(selected_subdir)
    
    # Retorna a imagem que está sendo exibida na tela
    def get_current_image(self):
        if self.is_searched_image:
            return cv2.imread(self.searched_image_path)
        else:
            return cv2.imread(self.image_files[self.current_image_index])
        
    # Retorna o caminho da imagem que está sendo exibida na tela
    def get_current_image_path(self):
        if self.is_searched_image:
            return self.searched_image_path
        else:
            return self.image_files[self.current_image_index]
    
    # Retorna a imagem em 256 tons de cinza
    def load_grayscale_image_256(self):
        colored_img = self.get_current_image()
        
        gray_img = image_operations.get_grayscale_image(colored_img, False)
                
        self.load_image(gray_img, True)
    
    # Retorna a imagem em 16 tons de cinza
    def load_grayscale_image_16(self):
        colored_img = self.get_current_image()
        
        gray_img = image_operations.get_grayscale_image(colored_img, True)
                
        self.load_image(gray_img, True)
    
    # Muda o valor de i que será passado para Matriz de Co-ocorrencia para 1
    def set_i_1(self):
        self.i = 1
    
    # Muda o valor de i que será passado para Matriz de Co-ocorrencia para 2
    def set_i_2(self):
        self.i = 2
    
    # Muda o valor de i que será passado para Matriz de Co-ocorrencia para 4  
    def set_i_4(self):
        self.i = 4
    
    # Muda o valor de i que será passado para Matriz de Co-ocorrencia para 8   
    def set_i_8(self):
        self.i = 8
    
    # Muda o valor de i que será passado para Matriz de Co-ocorrencia para 16
    def set_i_16(self):
        self.i = 16
    
    # Muda o valor de i que será passado para Matriz de Co-ocorrencia para 32
    def set_i_32(self):
        self.i = 32
    
    # Mostra na tela os valores das descrições de Haralick para o valor de i selecionado no menu
    def show_haralick_features(self):
        entropy, homogeneity, contrast = image_operations.get_haralick_features(self.i, self.get_current_image())

        self.entropyLabel.config(text=f"Entropia: {round(entropy, 2)}")
        self.homogeneityLabel.config(text=f"Homogeneidade: {round(homogeneity, 2)}")
        self.contrastLabel.config(text=f"Contraste: {round(contrast, 2)}")
    
    # Mostra na tela os valores dos momentos de Hu para a imagem em tons de cinza e para os canais H, S e V
    def show_hu_moments(self):
        hu_moments_gray, hu_moments_h, hu_moments_s, hu_moments_v = image_operations.get_hu_moments_menu(self.get_current_image())
        
        # Formata os momentos de Hu para serem exibidos na tela
        hu_moments_gray = ',\n'.join(str(valor) for valor in hu_moments_gray)
        hu_moments_h = ',\n'.join(str(valor) for valor in hu_moments_h)
        hu_moments_s = ',\n'.join(str(valor) for valor in hu_moments_s)
        hu_moments_v = ',\n'.join(str(valor) for valor in hu_moments_v)

        # Atualiza os valores dos momentos de Hu na tela
        self.grayImageHuLabel.config(text=f"Cinza: {hu_moments_gray}")
        self.hHuLabel.config(text=f"Canal H: {hu_moments_h}")
        self.sHuLabel.config(text=f"Canal S: {hu_moments_s}")
        self.vHuLabel.config(text=f"Canal V: {hu_moments_v}")
    
    # Mostra na tela os histogramas gerados para imagem em tons de cinza 256 e 16 tons, e os histogramas de cores no espaço RGB e HSV
    def show_histograms(self):
        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        
        colored_img = self.get_current_image()

        # Gera o histograma para a imagem 256 tons de cinza
        gray_img = image_operations.get_grayscale_image(colored_img, False)
        
        sns.histplot(gray_img.flatten(), color="gray", ax=axes[0])
        axes[0].set_title("Histograma de Tons de Cinza")
        axes[0].set_xlabel("Intensidade")
        axes[0].set_ylabel("Frequência")
        axes[0].set_xlim([0, 256])
        
        # Gera o histograma para a imagem 16 tons de cinza
        gray_img = image_operations.get_grayscale_image(colored_img, True)
        
        sns.histplot(gray_img.flatten(), color="gray", ax=axes[1])
        axes[1].set_title("Histograma de Tons de Cinza - 16 Tons")
        axes[1].set_xlabel("Intensidade")
        axes[1].set_ylabel("Frequência")
        axes[1].set_xlim([0, 256])

        # Gera o histograma de cores no espaço RGB
        blue, green, red = cv2.split(colored_img)

        sns.histplot(red.flatten(), color="red", alpha=0.6, binrange=(0, 256), binwidth=8, kde=True, ax=axes[2])        
        sns.histplot(green.flatten(), color="green", alpha=0.5, binrange=(0, 256), binwidth=8, kde=True, ax=axes[2])        
        sns.histplot(blue.flatten(), color="blue", alpha=0.3, binrange=(0, 256), binwidth=8, kde=True, ax=axes[2])

        axes[2].set_title("Histograma de cores no espaço RGB")
        axes[2].set_xlabel("Intensidade")
        axes[2].set_ylabel("Frequência")
        axes[2].set_xlim([0, 256])
        
        # Gera o histograma de cores no espaço HSV utilizando os canais Value e Hue
        h_channel, _, v_channel = cv2.split(colored_img)

        value = v_channel.flatten()
        value = [(pixel//8)*8 for pixel in value]
        hue = h_channel.flatten()
        hue = [(pixel//16)*16 for pixel in hue]

        sns.histplot(value, color="black", ax=axes[3])
        sns.histplot(hue, color="orange", ax=axes[3])
        
        axes[3].set_title("Histograma de cores no espaço HSV")
        axes[3].set_xlabel("Intensidade")
        axes[3].set_ylabel("Frequência")
        axes[3].legend(('Value', "Hue"))

        plt.tight_layout()
        plt.show()
    
    # Limpa os valores dos descritores de Haralick e dos momentos de Hu ao trocar a imagem na tela
    def clear_values(self):
        self.entropyLabel.config(text="")
        self.homogeneityLabel.config(text="")
        self.contrastLabel.config(text="")
        self.grayImageHuLabel.config(text="")
        self.hHuLabel.config(text="")
        self.sHuLabel.config(text="")
        self.vHuLabel.config(text="")
    
root = ThemedTk(theme="Arc")
app = Menu(root)
root.mainloop()
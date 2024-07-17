import pandas as pd
import os
from PIL import Image

# Definir caminhos
csv_path = 'classifications.csv'
images_dir = 'dataset'  # Diretório contendo as imagens originais
output_dir = 'src/'  # Diretório para salvar as sub-imagens
not_found_filename = 'not_found_images.json'  # Arquivo JSON para armazenar os nomes dos arquivos não encontrados

# Criar diretórios de saída para cada classe
classes = ["Negative for intraepithelial lesion", "ASC-US", "ASC-H", "LSIL", "HSIL", "SCC"]
for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# Inicializar lista para armazenar nomes de arquivos não encontrados
not_found_images = []

# Ler a planilha
data = pd.read_csv(csv_path)
print(f"planilha {csv_path} lida")

# Processar cada linha da planilha
for _, row in data.iterrows():
    image_filename = row['image_filename']
    cell_id = row['cell_id']
    bethesda_system = row['bethesda_system']
    nucleus_x = row['nucleus_x']
    nucleus_y = row['nucleus_y']

    # Caminho completo da imagem
    image_path = os.path.join(images_dir, image_filename)
    
    # Verificar se o arquivo de imagem existe
    if not os.path.exists(image_path):
        print(f"Arquivo de imagem não encontrado: {image_path}")
        if image_filename not in not_found_images:
            not_found_images.append(image_filename)
        continue

    # Abrir a imagem
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # Calcular as coordenadas da sub-imagem
            left = max(nucleus_x - 50, 0)
            upper = max(nucleus_y - 50, 0)
            right = min(nucleus_x + 50, width)
            lower = min(nucleus_y + 50, height)

            # Garantir que a sub-imagem tenha o tamanho 100x100
            sub_img = img.crop((left, upper, right, lower))
            sub_img = sub_img.resize((100, 100))

            # Definir o caminho de saída
            output_path = os.path.join(output_dir, bethesda_system, f'{cell_id}.png')
            
            # Salvar a sub-imagem
            sub_img.save(output_path)
            print(f"Sub-imagem salva em: {output_path}")
    except Exception as e:
        print(f"Erro ao processar a imagem {image_filename}: {e}")
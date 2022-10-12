import numpy as np
import cv2

def remover_fundo(path):
    target = path
    input_image = cv2.imread(target) 

    scale_percentual = 20
    width = int(input_image.shape[1] * scale_percentual/100)
    height = int(input_image.shape[0] * scale_percentual/100)
    dimension = (width, height)
    
    two_dimensional_image = input_image.reshape((-1, 3))
    two_dimensional_image = np.float32(two_dimensional_image)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    clusters = 3
    flags = cv2.KMEANS_PP_CENTERS

    result_image_compactness, result_image_labels, result_image_centers = cv2.kmeans(two_dimensional_image, clusters, None, criteria, 10, flags)

    result_image_centers = np.uint8(result_image_centers)
    result_image = result_image_centers[result_image_labels.flatten()]
    result_image = result_image.reshape((input_image.shape))

    thresholded_blue_component, thresholded_green_component, thresholded_red_component = cv2.split(result_image)

    used_threshold, thresholded_bgr_image = cv2.threshold(result_image, 130, 255, cv2.THRESH_BINARY)
    thresholded_blue_component, thresholded_green_component, thresholded_red_component = cv2.split(thresholded_bgr_image)

    mask_kmeans = thresholded_red_component
    mask_kmeans_filtered = cv2.medianBlur(mask_kmeans, 5)

    result_image = cv2.bitwise_and(input_image, input_image, mask = mask_kmeans_filtered)
    return input_image

#APLICAR TREINO
def detectar_embriao(result_image, path_cascade):
    input_image = result_image
    scale_percentual = 20
    width = int(input_image.shape[1] * scale_percentual/50)
    height = int(input_image.shape[0] * scale_percentual/50)
    dimension = (width, height)

    #IMPORTANDO ML TREINADA
    Classificador = cv2.CascadeClassifier(path_cascade)
    
    #OBRIGATÓRIO transformar em cinza para ficar mais preciso.
    cinza = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    '''
    Aplica a ML na imagem CINZA;
    scaleFactor = o fator de escala define a porcentagem de down ou upscaling entre dois níveis e, 
    portanto, basicamente decide quantos níveis haverá. Por exemplo, 1.10 significa que cada vez que você reduz a imagem em 10%.;
    '''
    detecta = Classificador.detectMultiScale(cinza, scaleFactor=4, minNeighbors=3)
    #Detecta vai receber de resultado um vetor de 4 posições. Ex.: detecta = [x, y, l, a]
    
    for(x, y, l, a) in detecta:
        cv2.rectangle(input_image, (x, y), (x + l, y + a), (0, 0, 255), 3)  
          
    cv2.imshow('Original Image', cv2.resize(input_image, dimension, interpolation = cv2.INTER_AREA))
    cv2.waitKey(0)
    cv2.destroyAllWindows()




#PATH DA IMAGEM
result_image = remover_fundo(path='D:\Projetos_VSCODE\Python\Area_Embriao\Interna_Reduzida.jpeg')
#PATH DO CASCADE.XML
detectar_embriao(result_image, 'D:\Projetos_VSCODE\Python\Area_Embriao\classifier\cascade.xml')


# -*- coding: utf-8 -*-
#
# Author: Gökçe Nur Beken
#
# Haziran 2021
#
# !!Kodu çalıştırmadan önce 20 ve 21.satırda yer alan değişkenlere görüntülerin bulunduğu yol eklenmelidir aksi taktirde kod hata vercektir!!

from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os

#görüntü boyutları 112 x 92
width  = 92
height = 112

#görüntü dosyaları konumu

TRAIN_IMG_FOLDER = 'Train_path/' 
TEST_IMG_FOLDER = 'Test_path/'
train_set_files = os.listdir(TRAIN_IMG_FOLDER)
test_set_files = os.listdir(TEST_IMG_FOLDER)

#------------------------ Veri Seti ----------------------------------


train_image_names = os.listdir(TRAIN_IMG_FOLDER)
# tüm eğitim görüntülerini bir dizide saklamak için dizi oluşturulur
training_dataset   = np.ndarray(shape=(len(train_image_names), height*width), dtype=np.float64)

for i in range(len(train_image_names)):
    img = plt.imread(TRAIN_IMG_FOLDER + train_image_names[i]) #her görüntü isimlerine göre alınıp eğitim veri seti dizisine eklenir.
    training_dataset[i,:] = np.array(img, dtype='float64').flatten()
#görüntü çizdirilir.
    #plt.subplot(10,6,1+i)
    #plt.imshow(img, cmap='gray')
    #plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
    #plt.subplots_adjust(right=1.2, top=2.5)
#print('Eğitim Görüntüleri:')
#plt.show()


test_image_names = os.listdir(TEST_IMG_FOLDER)
testing_dataset   = np.ndarray(shape=(len(test_image_names), height*width), dtype=np.float64)

for i in range(len(test_image_names)):
    img = imread(TEST_IMG_FOLDER + test_image_names[i])
    testing_dataset[i,:] = np.array(img, dtype='float64').flatten()
    #plt.subplot(11,4,1+i)
    #plt.imshow(img, cmap='gray')
    #plt.subplots_adjust(right=0.8, top=2.0)
    #plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#print('Test Görüntüleri:')
#plt.show()
#print("Eğitim veri seti boyutu:",training_dataset.shape)
#print("Test veri seti boyutu:",testing_dataset.shape)

#------------------------------------------------------------------------

#------------------------ Ortalama Yüz ----------------------------------

#ortalama yüz vektörü için sıfır matrisi oluşturulur.
mean_face = np.zeros((1,height*width))

for i in training_dataset: #eğitim veri setindeki görüntüler toplanır.
    mean_face = np.add(mean_face,i)

mean_face = np.divide(mean_face,float(len(training_dataset))).flatten() #ortalama yüz hesaplanır.

#print('Ortalama Yüz:')
#ortalama yüz çizdirilir.
#plt.imshow(mean_face.reshape(height, width), cmap='gray')
#plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#plt.show()
#print("Ortalama yüzün boyutu: ",mean_face.shape)

#------------------------------------------------------------------------

#------------------------ Normalize Yüzler ----------------------------------

#her görüntü gerçek görüntüden ortalama kadar farklıdır. her görüntüden ortalamanın çıkarılıp A dizisine atanır.
A_dataset = np.ndarray(shape=(len(training_dataset), height*width))

for i in range(len(training_dataset)):
    A_dataset[i] = np.subtract(training_dataset[i],mean_face) #eğitim görüntüsünden ortalama yüz çıkarılıp A_dataset dizisine atanır.

#for i in range(len(training_dataset)):
    #img = A_dataset[i].reshape(height,width)
    #plt.subplot(10,6,1+i)
    #plt.imshow(img, cmap='gray')
    #plt.subplots_adjust(right=1.2, top=2.5)
    #plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#plt.show()

#print("Boyutu: ",A_dataset.shape)
#print("\nElde edilen matris\n",A_dataset)

#------------------------------------------------------------------------

#------------------------ Kovaryans Matris ----------------------------------

cov_matrix = np.cov(A_dataset)
cov_matrix = np.divide(cov_matrix,60.0)
#print("Kovaryans Matris Boyutu: ",cov_matrix.shape)
#print("\nA'nın Kovaryans Matrisi\n",cov_matrix)

#------------------------------------------------------------------------

#------------------------ Özvektörler ve Özdeğerler ----------------------------------

eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
#print("Boyut: ",eigenvectors.shape)
#print('\nÖzvektörler:\n',eigenvectors)
#print("\nBoyut: ",eigenvalues.shape)
#print('\nÖzdeğerler:\n',eigenvalues)

idx = eigenvalues.argsort()[::-1]   
eigen_val = eigenvalues[idx]
eigen_vector = eigenvectors[:,idx]
eigen_val = np.diag(eigen_val)
#print("...Özdeğerler...")
#print(eigen_val.shape)
#print()
#print("...Özvektörler...")
#print(eigen_vector.shape)

#print(eigenvectors.shape)
k=30 #en büyük özdeğere sahip özyüz sayısı
k_eigenvectors = eigen_vector[:,0:k]
#print(k_eigenvectors.shape)

eigen_faces = np.transpose(k_eigenvectors).dot(A_dataset)
#print(eigen_faces.shape)
#print(eigen_faces)

#print("En büyük özdeğere sahip 30 özyüz görüntüsü:")
#for i in range(eigen_faces.shape[0]):
    #img = np.transpose(eigen_faces[i]).reshape(height,width)
    #plt.subplot(10,6,1+i)
    #plt.imshow(img, cmap='gray')
    #plt.subplots_adjust(right=1.2, top=2.5)
    #plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#plt.show()

#eigen_faces.shape

#eğitim setinin ağırlık matrisinin elde edilmesi.
weights = np.transpose(eigen_faces.dot(np.transpose(A_dataset)))
#print("Eğitim seti eğırlık vektörü boyutu:",weights.shape)
#print("\nAğırlık vektörü:\n",weights)

#------------------------------------------------------------------------

#------------------------ Test Görüntüleri ----------------------------------

#test görüntülerinden eğitim setinin ortalaması çıkarılır ve bir dizide saklanır.

fi_T_dataset = np.ndarray(shape=(len(testing_dataset), height*width))

for i in range(len(testing_dataset)):
    fi_T_dataset[i] = np.subtract(testing_dataset[i],mean_face)

#for i in range(len(testing_dataset)):
    #img = fi_T_dataset[i].reshape(height,width)
    #plt.subplot(11,4,1+i)
    #plt.imshow(img, cmap='gray')
    #plt.subplots_adjust(right=1.2, top=2.5)
    #plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#plt.show()
#print("Boyutu:",fi_T_dataset.shape)

#test görüntülerinin ağırlık matrisi hesaplanır.

wt = (eigen_faces).dot(np.transpose(fi_T_dataset))
#print("Boyutu:",wt.shape)
#print("\nTest Ağırlık Matrisi:\n",wt)

#------------------------------------------------------------------------

#------------------------ Yüz tanıma ----------------------------------

count        = 0
num_images   = 0
def Visualization(img, train_image_names,proj_data,w, t0):
    global count,highest_min,num_images,correct_pred
    unknown_face        = plt.imread(TEST_IMG_FOLDER+img)
    num_images          += 1
    unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()
    normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)
    
    plt.subplot(11,8,1+count)
    plt.imshow(unknown_face, cmap='gray')
    plt.title('Input:'+'.'.join(img.split('.')[:2]))
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
    count+=1
    
    #ölkid mesafesi hesaplanır
    w_unknown = np.dot(proj_data, normalised_uface_vector)
    diff  = w - w_unknown
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)
    
   

    
    plt.subplot(11,8,1+count)
    if norms[index] < t0: #en küçük ölkid mesafesi olan t0'dan küçükse bir yüz olarak algılanır.
            
        match = img.split()[0] == train_image_names[index].split()[0]
        
        if match:
            
            plt.title('False matched:')
            plt.imshow(imread(TRAIN_IMG_FOLDER+train_image_names[index]), cmap='gray')   
        else:
            
            plt.title('Matched:')
            plt.imshow(imread(TRAIN_IMG_FOLDER+train_image_names[index]), cmap='gray')
               
    else:
        
        if img.split()[0] not in [i.split()[0] for i in train_image_names]:
            plt.title('Unknown face')
            
        else:
            plt.title('Unknown face')
                
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
    plt.subplots_adjust(right=1.2, top=2.5)
   
    count+=1

    
fig = plt.figure(figsize=(10, 10))

test_image_names2 = sorted(test_image_names)
for i in range(len(test_image_names2)):
    Visualization(test_image_names2[i], train_image_names,eigen_faces,weights, t0=2.7e7)

plt.show()

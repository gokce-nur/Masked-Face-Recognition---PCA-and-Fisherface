# -*- coding: utf-8 -*-
#
# Author: Gökçe Nur Beken
#
# Haziran 2021
#
# !!Kodu çalıştırmadan önce 20 ve 21.satırda yer alan değişkenlere görüntülerin bulunduğu yol eklenmelidir aksi taktirde kod hata vercektir!!

from matplotlib import pyplot as plt
from matplotlib.image import imread
from numpy.linalg import inv
import numpy as np
import os

#görüntü boyutları 112 x 92
width  = 92
height = 112

#------------------------ Veri Seti ----------------------------------

number_of_classes=10 #sınıf sayısı
img_in_class=6 #her sınıftaki görüntü sayısı

training_dataset   = np.ndarray(shape=(number_of_classes*img_in_class, height*width), dtype=np.float64)

for i in range(number_of_classes):
    for j in range(img_in_class):
        img = plt.imread('Train_path/s'+str(i+1)+'/'+str(j+1)+'.pgm')
        
        training_dataset[img_in_class*i+j,:] = np.array(img, dtype='float64').flatten()
       
        #plt.subplot(number_of_classes,img_in_class,1+img_in_class*i+j)
        #plt.imshow(img, cmap='gray')
        #plt.subplots_adjust(right=1.2, top=2.5)
        #plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#print('Eğitim Görüntüleri:')
#plt.show()

testing_dataset   = np.ndarray(shape=(44, height*width), dtype=np.float64)

for i in range(44):
    img = imread('Test_path/'+str(i+1)+'.pgm')
    testing_dataset[i,:] = np.array(img, dtype='float64').flatten()
    #plt.subplot(11,4,1+i)
    #plt.imshow(img, cmap='gray')
    #plt.subplots_adjust(right=0.8, top=2.0)
    #plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#print('Test Görüntüleri:')
#plt.show()

#------------------------------------------------------------------------

#------------------------ PCA ----------------------------------

def PCA(training_dataset):
    
    mean_face = np.zeros((1,height*width))
    for i in training_dataset:
        mean_face = np.add(mean_face,i)
    mean_face = np.divide(mean_face,float(training_dataset.shape[0])).flatten()
    
    A_dataset = np.ndarray(shape=(len(training_dataset), height*width))

    for i in range(len(training_dataset)):
        A_dataset[i] = np.subtract(training_dataset[i],mean_face)

    cov_matrix = np.cov(A_dataset)
    cov_matrix = np.divide(cov_matrix,60.0)
 
    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]   
    eigen_val = eigenvalues[idx]
    eigen_vector = eigenvectors[:,idx]
    eigen_val = np.diag(eigen_val)
    k=30
    k_eigenvectors = eigen_vector[:,0:k]
    eigen_faces = np.transpose(k_eigenvectors).dot(A_dataset)
    
    weights = np.transpose(eigen_faces.dot(np.transpose(A_dataset)))
    
    return eigen_faces, weights

#------------------------------------------------------------------------

projected_eigenfaces, projected_weights = PCA(training_dataset) #PCA'dan özyüz vektörü ve ağırlık matrisi alınır ve değişkenlere atanır.
#print("PCA'dan gelen ağırlık matrisi boyutu:",projected_weights.shape)
#print("\nPCA'dan gelen ağırlık matrisi:\n",projected_weights)

#------------------------ Ortalama hesabı ----------------------------------

k = 30
mi = np.zeros((number_of_classes, k))
mo = np.zeros((1,k))

for i in range(number_of_classes): #Sınıf içi ortalama görüntünün hesaplanması
    xa = projected_weights[img_in_class*i:img_in_class*i+img_in_class,:]
    for j in xa:
        mi[i,:] = np.add(mi[i,:],j)
    mi[i,:] = np.divide(mi[i,:],float(len(xa)))

for i in projected_weights: #Tüm görüntülerin ortalamasının hesaplanması
    mo = np.add(mo,i)
mo = np.divide(mo,float(len(projected_weights)))

#print("mo boyutu: ",mo.shape)
#print("mi boyutu: ",mi.shape)

#------------------------------------------------------------------------

#------------------------ Sınıf içi saçılma ve Sınıflar arası saçılma ----------------------------------

normalised_wc_proj_weights = np.ndarray(shape=(number_of_classes*img_in_class, k), dtype=np.float64)

for i in range(number_of_classes):
    for j in range(img_in_class): 
        normalised_wc_proj_weights[i*img_in_class+j,:] = np.subtract(projected_weights[i*img_in_class+j,:],mi[i,:]) #yeni veri setindeki (PCA'dan gelen) her görüntüden sınıf içi ortalamnın çıkarılması.
normalised_wc_proj_weights.shape

#sınıf içi saçılma matrisinin hesaplanması.
sw = np.zeros((k,k))

for i in range(number_of_classes):
    xa = normalised_wc_proj_weights[img_in_class*i:img_in_class*i+img_in_class,:]
    xa = xa.transpose()
    cov = np.dot(xa,xa.T)
    sw = sw + cov
#sw.shape

#sınıflar arası saçılma matrisinin hesaplanması
normalised_proj_weights = np.ndarray(shape=(number_of_classes*img_in_class, k), dtype=np.float64)
for i in range(number_of_classes*img_in_class):
    normalised_proj_weights[i,:] = np.subtract(projected_weights[i,:],mo)

sb = np.dot(normalised_proj_weights.T,normalised_proj_weights)
sb = np.multiply(sb,float(img_in_class))
#sb.shape

#------------------------------------------------------------------------

#------------------------ Özvektörler ve Özdeğerler ----------------------------------

#genelleştirilmiş özdeğer propleminin kullanılması ile j matrisi bulunur.
J = np.dot(inv(sw), sb)
#J.shape

#özvektörler ve özdeğerler j matrisindedir.
eigenvalues, eigenvectors, = np.linalg.eig(J)

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
k=15 #en yüksek özdeğer sayısı
k_eigenvectors = eigen_vector[:,0:k]
#print(k_eigenvectors.shape)

#------------------------------------------------------------------------

#------------------------ Fisher Yüzler ----------------------------------

projected_weights.shape
FP = np.dot(projected_weights, k_eigenvectors)
#FP.shape

#print("Fisher yüzler:")

fisher_face = np.dot(training_dataset.transpose(),FP)
fisher_face = fisher_face.transpose()
#fisher_face.shape


#for i in range(fisher_face.shape[0]):
    #img = fisher_face[i].reshape(height,width)
    #print(img)
    #plt.subplot(10,3,1+i)
    #plt.imshow((img.real*img.real + img.imag*img.imag)**0.5, cmap='gray')
    #plt.subplots_adjust(right=1.2, top=2.5)
    #plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#plt.show()

#print("Eğitim setinin ortalama yüz görüntüsü:")

mean_face = np.zeros((1,height*width))

for i in training_dataset:
    mean_face = np.add(mean_face,i)

mean_face = np.divide(mean_face,float(len(training_dataset))).flatten()

#plt.imshow(mean_face.reshape(height, width), cmap='gray')
#plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#plt.show()

#------------------------------------------------------------------------

#------------------------ Yüz Tanıma ----------------------------------

count=0
num_images=0
def recogniser(img_number):
    global count,highest_min,num_images,correct_pred
    
    num_images          += 1
    unknown_face_vector = testing_dataset[img_number,:]
    normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)
    
    plt.subplot(11,8,1+count)
    plt.imshow(unknown_face_vector.reshape(height,width), cmap='gray')
    plt.title('Input:'+str(img_number+1))
    plt.subplots_adjust(right=1.2, top=2.5)
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
    count+=1
    
    #ölkid mesafesi hesaplanır
    PEF = np.dot(projected_eigenfaces,normalised_uface_vector)
    proj_fisher_test_img = np.dot(k_eigenvectors.T,PEF)
    diff  = FP - proj_fisher_test_img
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)
    
    plt.subplot(11,8,1+count)
    
    set_number = int(img_number/4)


    t0 = 7000000 
    
    
    if norms[index] < t0: #en küçük ölkid mesafesi olan t0'dan küçükse bir yüz olarak algılanır.
        if(index>=(6*set_number) and index<(6*(set_number+1))):
            plt.title('Matched with:'+str(index+1))
            plt.imshow(training_dataset[index,:].reshape(height,width), cmap='gray')
            
        else:
            plt.title('False Matched with:'+str(index+1))
            plt.imshow(training_dataset[index,:].reshape(height,width), cmap='gray')
    else:
        if(img_number>=40):
            plt.title('Unknown face!')
            
        else:
            plt.title('Unknown face!')
    plt.subplots_adjust(right=1.2, top=2.5)
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
    count+=1

fig = plt.figure(figsize=(10, 10))
for i in range(len(testing_dataset)):
    recogniser(i)

plt.show()


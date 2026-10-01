import tensorflow
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
(tri,trl),(testi,testl)=tensorflow.keras.datasets.cifar10.load_data()
tri,testi=tri/255.0,testi/255.0
trl,testl=to_categorical(trl),to_categorical(testl)
def b(n):
    units=[512,128,64,32]
    model=models.Sequential([Flatten(input_shape=(32,32,3))])
    for i in units:
        model.add(Dense(i,activation='relu',kernel_initializer=n))

        model.add(Dropout(0.3))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
  hmap={}
for i in ['glorot_normal','he_normal']:
    model=b(i)
    print(f'Training with{i}')
    hist=model.fit(tri,trl,epochs=15,batch_size=1024,validation_data=(testi,testl))
    hmap[i]=hist
    _,acc=model.evaluate(testi,testl)
    print(f'ACC of {i}:{acc}')

for i,j in hmap.items():
    
    plt.plot(j.history['val_accuracy'],label=i)
plt.legend()
plt.show()
for i,j in hmap.items():
    plt.plot(j.history['accuracy'],label='x_train')
    plt.plot(j.history['val_accuracy'],label='x_test')
    plt.show()

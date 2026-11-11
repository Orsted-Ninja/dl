import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,LSTM,GRU,SimpleRNN
import numpy as np
import matplotlib.pyplot as plt
t=np.linspace(0,10,1000)

data=np.sin(t)+0.1*np.random.randn(1000)


ts=20
X=np.array([data[i:i+ts] for i in range(len(data)-ts)]).reshape(-1,ts,1)
Y=np.array([data[i+ts] for i in range(len(data)-ts)])
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,shuffle=False)
m={'SimpleRNN':SimpleRNN,'LSTM':LSTM,'GRu':GRU}
for n,l in m.items():
    model=Sequential([l(50,activation='relu',input_shape=(ts,1)),Dense(1)])
    model.compile(optimizer='adam',loss='mse')
    history=model.fit(x_train,y_train,epochs=10,batch_size=64)
    pred=model.predict(x_test)
    print(mean_squared_error(y_test,pred))
    
    
    plt.plot(history.history["loss"],label=f"n loss")
plt.xlabel("epochs")
plt.ylabel("LOSS")
plt.legend()
plt.show()

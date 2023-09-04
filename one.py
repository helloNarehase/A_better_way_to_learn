import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plot
import time

x = np.arange(0, 900)
y = np.tan(np.radians((x-450)*0.1))*1.5

inputs = Input((1))
a = Dense(64, activation='relu')(inputs)
a = Dense(64, activation='relu')(a)
a = Dense(64, activation='relu')(a)
a = Dense(64, activation='relu')(a)
output = Dense(1)(a)
model = Model(inputs, output)
print(model.summary())

model.compile(tf.optimizers.Adam(), loss=tf.losses.MeanSquaredError())

xs = [x[100*i: 100*(i+1)] for i in range(9)]
ys = [y[100*i: 100*(i+1)] for i in range(9)]

# plot.plot(x/1000, label="input *0.001")
# plot.plot(y , label="output")
# plot.legend(loc="best")
# plot.savefig("data.png")
# plot.show()

#  문제가 제시된 학습 방식 >> 

# et = time.time()
# fig, axes = plot.subplots(2, 5, figsize=(18, 8))
# for i in range(len(xs)):
#     model.fit(xs[i],ys[i], epochs=2000, verbose=0)
#     print(f"{i} : {time.time() - et}")

#     prediction = model.predict(x)
#     c = (i, 0) if i < 5 else (i-5, 1)
#     print(c)
#     axes[c[1]][c[0]].plot(prediction, label="predict")
#     axes[c[1]][c[0]].plot(y , label="data")
#     axes[c[1]][c[0]].legend(loc="best")
#     axes[c[1]][c[0]].set_title(f"batch : {len(xs)} / {i+1}")

# <<<


#  개선된 학습 방식 >> 

et = time.time()
fig, axes = plot.subplots(2, 5, figsize=(18, 8))
for l in range(1000):
    for ss in range(2):
        for i in range(len(xs)):
            model.fit(xs[i],ys[i], epochs=1, verbose=0)
    if (l+1)%10 == 0:
        print(f"{l+1} : {time.time() - et}")
    
    if (l+1)%100 == 0 and l > 10:        
        prediction = model.predict(x)
        c = ((l//100), 0) if (l//100) < 5 else ((l//100)-5, 1)
        print(c)
        print(f"full batch train: {1000} / {l+1}")
        axes[c[1]][c[0]].plot(prediction, label="predict")
        axes[c[1]][c[0]].plot(y , label="data")
        axes[c[1]][c[0]].legend(loc="best")
        axes[c[1]][c[0]].set_title(f"full batch train: {1000} / {l+1}")

# <<<
        
# plot.savefig("cooki3.png")

score = model.evaluate(x, y, verbose=0)
# 정확도 출력
print("loss:", score)

# prediction = model.predict(x)
# plot.plot(prediction, label="predict")
# plot.plot(y , label="data")
# plot.legend(loc="best")
plot.show()

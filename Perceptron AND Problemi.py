#Perceptron Ağ Yapısı Örneğinin AND Probleminin Pythonla Örnek Çözümü
X, Y = [(0,0),(0,1),(1,0),(1,1)], [0,0,0,1]
w1, w2, b = -0.7, -0.3, 0.1

for epoch in range(7):
    for(x1, x2), t in zip(X, Y):
        a=1 if (w1*x1 + w2*x2 + b) > 0 else 0
        w1, w2, b = w1 + (t-a)*x1, w2 + (t-a)*x2, b + (t-a)

print(f"Hatayı 0 yapan değerler ---> w1: {w1:.1f}, w2: {w2:.1f}, b: {b:.1f}")

def svm_sgd_plot(X, Y):
    
    w = np.zeros(len(X[0]))
   
    eta = 1
   
    epochs = 100000
   
    errors = []

    
    for epoch in range(1,epochs):
        error = 0
        for i, x in enumerate(X):
            
            if (Y[i]*np.dot(X[i], w)) < 1:
               
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
                error = 1
            else:
               
                w = w + eta * (-2  *(1/epoch)* w)
        errors.append(error)
        

   
    plt.plot(errors, '|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()
    
    return w
 
for d, sample in enumerate(X):
   
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
   
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)


plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')


x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')
w = svm_sgd_plot(X,y)

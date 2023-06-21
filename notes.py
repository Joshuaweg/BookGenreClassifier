##random tensor
#x = torch.rand([5])
#print("random: ",x)

##empty tensor
#x= torch.empty([2,5])
#print("empty: ",x)

##zero tensor
#x= torch.zeros([2,5])
#print("zero: ",x)

##one tensor
#x=torch.ones([2,5])
#print("one: ",x)

##int tensor
#x=torch.ones([2,5], dtype=torch.int)
#print("int : ",x)

##double tensor
#x=torch.rand([5,2], dtype=torch.double)
#print("double: ",x)
#print("size: ",x.size())

##tensor from data [[1, 3], [5, 7],[11,13], [17, 19], [23, 29]]
#x=torch.tensor([[1, 3], [5, 7],[11,13], [17, 19], [23, 29]])
#print("data tensor: ",x)
#print("size ",x.size())

##Operations

##oper
#a = torch.rand([2,2]);
#b = torch.rand([2,2]);

#print(a,"\n",b)
#print("addition:")
#print(a+b)
#print("substraction:")
#print(a-b)
#print("multiplication:")
#print(a*b)

#c = torch.rand([3,2]);
#d = torch.rand([2,3]);
#print("Matrix Multiplication: ")
#print(c,"\n",d)
#print(c.matmul(d))
#print("Transpose:")
#print(c.T.matmul(d.T))

##reshape

#e = torch.rand([6,10])

#print(e)

#print("reshape 2 X 30")

#print(e.view([2,30]))
##torch to numpt
#n = e.numpy()

#print(n)

#grad = torch.rand(3, requires_grad=True)
#y = grad+2
#print(y)
#z = y*y*2
#print(z)
#v=torch.tensor([0.1,1.0,0.001], dtype=torch.float32)
#z.backward(v)#dz/dgrad
#print(grad.grad)

#prevent grad tracking
#x.requires_grad_(False)
#x.detach()
#with torch.no_grad()

#weights = torch.ones(4, requires_grad=True)
#optimmizer = torch.optim.SGD(weights, lr=0.01)

#for epoch in range(3):
#    model_output =(weights*3).sum()
#    model_output.backward()
#    print(weights.grad)
#    weights.grad.zero_()

#backpropagation
#a = torch.tensor(1.0)
#b = torch.tensor(2.0)
#w = torch.tensor(1.0,requires_grad=True)

#for epoch in range(2):
## forward pass
#    y_hat = w*a
#    loss = (y_hat-b)**2

#    print(loss)

##backward
#    loss.backward()
#    print(w.grad)



##Manual   
# # f = wx

# # f = 2 *x

#X = np.array([1,2,3,4], dtype=np.float32)

#Y = np.array([2,4,6,8], dtype=np.float32)

#w = 0.0

## model prediction
#def forward(x):
#    return w*x


##loss
#def loss(y, y_hat):
#    return((y_hat-y)**2).mean()


##gradient
##MSE = 1/N (wx-y)**2
##MSE = 1/N (w^2 X^2 - 2wxy + y^2)
##dJ/dw = 1/N (2wX^2 - 2xy)
#def gradient(x,y, y_hat):
#    return np.dot(2*x,y_hat-y).mean()

#print(f'Prediction before training: f(5) = {forward(5):.3f}')

##Training

#learning_rate = 0.01
#n_iters = 20

#for epoch in range(n_iters):
#    y_pred = forward(X)
#    l = loss(Y,y_pred)
#    dw = gradient(X,Y,y_pred)

#    #update weight
#    w -= learning_rate*dw

#    if epoch % 10 == 0:
#        print(f'epoch {epoch+1}: w={w:.3f} loss={l:.8f}')

#print(f'Prediction after training: f(5) = {forward(5):.3f}')

#end of Manual

#Pytorch Pipepline
# 1) design model (input, output size, forward pass)
# 2) Construct Loss and optimizer
# 3) Training loop
#    - forward pass: compute prediction
#    -backward pass: gradients
#    -update weights
 # f = wx

# # f = 2 *x
#import torch.nn as nn
#X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)

#Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
#n_samples, n_features = X.shape
#X_test = torch.tensor([5],dtype=torch.float32)
#input_size = n_features
#output_size =n_features
##model = nn.Linear(input_size,output_size)
#class LinearRegression(nn.Module):
#    def __init__(self,input_dim,output_dim):
#        super(LinearRegression, self).__init__()

#        self.lin=nn.Linear(input_dim,output_dim)
#    def forward(self, x):
#       return self.lin(x)

#model = LinearRegression(input_size, output_size)
#print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

##loss


##gradient
##MSE = 1/N (wx-y)**2
##MSE = 1/N (w^2 X^2 - 2wxy + y^2)
##dJ/dw = 1/N (2wX^2 - 2xy)

##Training

#learning_rate = 0.04
#n_iters = 1000
#loss = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#for epoch in range(n_iters):
#    y_pred = model(X)
#    l = loss(Y,y_pred)
#    l.backward() #dl/dw

#    #update weight
#    optimizer.step()
#    optimizer.zero_grad()
#    if epoch % 200 == 0:
#        [w, b] = model.parameters()
#        print(f'epoch {epoch+1}: w={w[0][0].item():.3f} loss={l:.8f}')

#print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

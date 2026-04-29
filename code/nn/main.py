



def predict(X, parameters):
    W = parameters['W']
    b = parameters['b']

    Z = W @ X + b
    return Z

df = pd.read_csv("/Users/truthixify/logs/learning-journal/code/nn/data/toy_dataset.csv")

X = np.array(df[['x1','x2']]).T
Y = np.array(df['y']).reshape(1,-1)

parameters = nn_model(X,Y, num_iterations = 5000, print_cost= True)
y_hat = predict(X,parameters)
df['y_hat'] = y_hat[0]

for i in range(10):
    print(f"(x1,x2) = ({df.loc[i,'x1']:.2f}, {df.loc[i,'x2']:.2f}): Actual value: {df.loc[i,'y']:.2f}. Predicted value: {df.loc[i,'y_hat']:.2f}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load your data
df = pd.read_csv('Your Data set')

# --- UNIVARIATE CASE: Only one feature ('area') ---
X_uni = df['against_value'].values.reshape(-1, 1)
y = df['prediction_value'].values.reshape(-1, 1)

# Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_uni = scaler_X.fit_transform(X_uni).flatten()
y = scaler_y.fit_transform(y).flatten()

# --- Functions as per your friend’s code ---

def param_init(Y):
    m = 0.1
    c = Y.mean()
    return m, c

def generate_predictions(m, c, X):
    prediction = []
    for x in X:
        pred = (m * x) + c
        prediction.append(pred)
    return prediction

def compute_cost(prediction, Y):
    cost = mean_squared_error(Y, prediction)
    return cost

def gradients(prediction, Y, X):
    n = len(Y)
    Gm = (2/n) * np.sum((prediction - Y) * X)
    Gc = (2/n) * np.sum(prediction - Y)
    return Gm, Gc

def param_update(m_old, c_old, Gm_old, Gc_old, alpha):
    m_new = m_old - alpha * Gm_old
    c_new = c_old - alpha * Gc_old
    return m_new, c_new

def result(m, c, X, Y, cost, predictions, i):
    if i < max_iter - 1:
        print(f'* Gradient Descent converged at iteration {i} *')
    else:
        print(f'* Result after {max_iter} iterations *')
    
    plt.figure(figsize=(14,7), dpi=120)
    plt.scatter(X, Y, color='red', label='Data points')
    label = f'Final Regression Line: m = {m:.4f}, c = {c:.4f}, cost = {cost:.6f}'
    plt.plot(X, predictions, color='green', label=label)
    plt.legend()
    plt.title('Univariate Linear Regression Result')
    plt.show()

# --- Gradient Descent Loop ---

max_iter = 5000
alpha = 0.05  # You can tune it manually or via loop
cost_old = 0
Y = y
X = X_uni
m, c = param_init(Y)

for i in range(max_iter):
    predictions = generate_predictions(m, c, X)
    cost_new = compute_cost(predictions, Y)

    if abs(cost_new - cost_old) < 1e-10:
        break

    Gm, Gc = gradients(predictions, Y, X)
    m, c = param_update(m, c, Gm, Gc, alpha)

    if i % 100 == 0:
        print(f'Iteration {i}: m={m:.4f}, c={c:.4f}, cost={cost_new:.6f}')

    cost_old = cost_new

# Final Results
result(m, c, X, Y, cost_new, predictions, i)

# --- Hypothesis function ---
print(f"Hypothesis Function: h(x) = {m:.4f} * x + {c:.4f}")

# --- Accuracy ---
r2 = r2_score(Y, predictions)
print(f'R² Score (Accuracy): {r2:.4f}')

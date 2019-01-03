import tensorflow as tf
import pandas as pd


W = tf.Variable([.5], dtype=tf.float32)
b = tf.Variable([-.5], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Define our loss function with standard logic.
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(linear_model - y))

# Use an optimizer with our loss function.
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Start a session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())


data = pd.read_csv('Salary_Data.csv')
Exp = data['YearsExperience'].tolist()
Salary = data['Salary'].tolist()
print(Exp[0])
print(Salary[0])
# print("This is Lenght")
print(type(Exp[0]))

for _ in range(1000):
    sess.run(train, {x: Exp[0], y:Salary[0] })

# Get optimized values for "W" and "b" with trained model.
result = sess.run([W, b])
print(result)

# Extract our learned values and print them.
optimal_W = result[0][0]
optimal_b = result[1][0]
print("optimal W:", optimal_W)
print("optimal b:", optimal_b)

# Run the linear model with a time of 5.
result_final = sess.run(linear_model, {x:[1.1]})
print("result:", result_final)

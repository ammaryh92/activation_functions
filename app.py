import streamlit as st
import numpy as np
import plotly.graph_objects as go

########################################
# Utility Code
z = np.linspace(-8,8,200)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)

def logistic(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def plot_function(func, title, alpha=None):
    fig = go.Figure()
    if alpha:
        fig.add_trace(go.Scatter(x=z, y=func(z, alpha=alpha), mode='lines', line=dict(color='red', width=3)))
    else:
        fig.add_trace(go.Scatter(x=z, y=func(z), mode='lines', line=dict(color='red', width=3)))
    fig.update_layout(title = title, xaxis_title='Z',width=700, height=400,
                            font=dict(family="Courier New, monospace",size=16,color="White"), margin=dict(t=30, b=0, l=0, r=0))
    fig.update_xaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')
    fig.update_yaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')

    return fig

def plot_function_derivative(func, title, alpha=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=derivative(func, z), mode='lines', line=dict(color='red', width=3)))
    fig.update_layout(title = title, xaxis_title='Z', width=700, height=400,
                            font=dict(family="Courier New, monospace",size=16,color="White"), margin=dict(t=30, b=0, l=0, r=0))
    fig.update_xaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')
    fig.update_yaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')
    return fig
#########################################


st.title('Activation Functions')

activation_function = st.selectbox('Choose an activation function', ['None', 'Logistic (Sigmoid) Function', 'Hyperbolic Tangent (Tanh) Function', 'ReLU Function', 'LeakyReLU Function', 'Variants of LeakyReLU'])

## Logistic Function
if activation_function == 'Logistic (Sigmoid) Function':

    st.header('Logistic Function')

    st.subheader('Description')
    st.write('It is a sigmoid function with a characteristic "S"-shaped curve.')
    st.markdown(r'**$sigmoid(z)=\frac{1}{1+exp(-z)}$**')
    st.write('The output of the logistic (sigmoid) function is always between 0 and 1.')   

    st.subheader('Plot')
    logistic_fig  = plot_function(logistic, title='Logistic (Sigmoid) Activation Functoin')
    logistic_fig.add_annotation(x=7, y=1, text='<b>Saturation</b>', showarrow=True,
     font=dict(family="Montserrat", size=16, color="#1F8123"),
        align="center",arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A835E1", ax=-20, ay=30,)
    logistic_fig.add_annotation(x=-7, y=0, text='<b>Saturation</b>', showarrow=True,
     font=dict(family="Montserrat", size=16, color="#1F8123"),
        align="center",arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A835E1", ax=0, ay=-30,)
    st.plotly_chart(logistic_fig)
    with st.expander('Plot Explanation'):
        st.write('- The logistic function saturates as the inputs become larger (either positive or negative).')
        st.write('- For large positive and negative values, the function gets asymptotically close to 1 and 0, respectively.')
        st.write('- When the function saturates, its gradient becomes very close to zero, which slows down learning.')

    st.subheader('Derivative')
    st.markdown(r'$sigmoid^{\prime}(z)=sigmoid(z)(1−sigmoid(z))$')
    st.text("")
    logistic_der_fig = plot_function_derivative(logistic, title='Derivative of the Logistic Function')
    st.plotly_chart(logistic_der_fig)
    with st.expander('Plot Explanation'):
        st.write('Notice that the derivative of the logistic function gets very close to zero for large positive and negative inputs.')

    st.subheader('Pros')
    st.write('1. The logistic function introduces non-linearity into the network which allows it to solve more complex problems than linear activation functions.\n2. It is continuous and differentiable everywhere.\n3. Because its output is between 0 and 1, it is very common to use in the output layer in binary classification problems.')

    st.subheader('Cons')
    st.write("1. Limited Sensitivity\nThe logistic function saturates across most of its domain. It is only sensitive to inputs around its midpoint 0.5.")
    st.write("2. Vanishing Gradients in Deep Neural Networks\nBecause the logistic function can get easily saturated with large inputs, its gradient gets very close to zero. This causes the gradients to get smaller and smaller as backpropagation progresses down to the lower layers of the network. Eventually, the lower layers' weights receive very small updates and never converge to their optimal values.")

## Tanh Function
if activation_function == 'Hyperbolic Tangent (Tanh) Function':
    st.header('Hyperbolic Tangent Function (Tanh)')

    st.subheader('Description')
    st.write('The tanh function is also a sigmoid "S"-shaped function.')
    st.markdown(r'$tanh(z)=\frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$')
    st.write('The range of the tanh function is between -1 and 1.')

    st.subheader('Plot')
    tanh_fig = plot_function(np.tanh, title='Hyperbolic Tangent Function')
    tanh_fig.add_annotation(x=7, y=1, text='<b>Saturation</b>', showarrow=True,
     font=dict(family="Montserrat", size=16, color="#1F8123"),
        align="center",arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A835E1", ax=-20, ay=30,)
    tanh_fig.add_annotation(x=-7, y=-1, text='<b>Saturation</b>', showarrow=True,
     font=dict(family="Montserrat", size=16, color="#1F8123"),
        align="center",arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A835E1", ax=0, ay=-30,)
    st.plotly_chart(tanh_fig)
    with st.expander('Plot Explanation'):
        st.write('- The tanh function saturates as the inputs become larger (either positive or negative).')
        st.write('- For large positive and negative values, the function gets asymptotically close to 1 and -1, respectively.')
        st.write('- When the function saturates, its gradient becomes very close to zero, which slows down learning.')
    
    st.subheader('Derivative')
    st.markdown(r'$tanh^{\prime}(z)= 1 - (tanh(z))^{2}$')
    st.text("")
    tanh_der_fig = plot_function_derivative(np.tanh, title='Derivative of the Tanh Function')
    st.plotly_chart(tanh_der_fig)
    with st.expander('Plot Explanation'):
        st.write('Notice that the derivative of the tanh function gets very close to zero for large positive and negative inputs.')
    
    st.subheader('Pros')
    st.write("1. The tanh function introduces non-linearity into the network which allows it to solve more complex problems than linear activation functions.\n2. It is continuous, differentiable, and have non-zero derivatives everywhere.\n3. Because its output value ranges from -1 to 1, that makes each layer's output more or less centered around 0 at the beginning of training, whcih speed up convergence.")

    st.subheader('Cons')
    st.write("1. Limited Sensitivity\nThe tanh function saturates across most of its domain. It is only sensitive to inputs around its midpoint 0.")
    st.write("2. Vanishing Gradients in Deep Neural Networks\nBecause the tanh function can get easily saturated with large inputs, its gradient gets very close to zero. This causes the gradients to get smaller and smaller as backpropagation progresses down to the lower layers of the network. Eventually, the lower layers' weights receive very small updates and never converge to their optimal values.")

    st.markdown("**Note**: the vanishing gradient problem is less severe with the tanh function because it has a mean of 0 (instead of 0.5 like the logistic function).")

## RelU
if activation_function == 'ReLU Function':
    st.header('Rectified Linear Unit (ReLU) Function')

    st.subheader('Description')
    st.write('It is a piecewise linear function with two linear pieces that will output the input directly is it is positive, otherwise, it will output zero.')
    st.markdown('$$ReLU(z) = max(0, z)$$')
    st.write('It has become the default activation function for many neural netowrks because it is easier to train and achieves better performance.')

    st.subheader('Plot')
    relu_fig = plot_function(relu, title = 'ReLU Function')
    st.plotly_chart(relu_fig)

    st.subheader('Derivative')
    st.markdown(r'$$ Relu^{\prime}(z)= \left\{\begin{array}{ll}1 & z>0 \\0 & z<=0 \\\end{array}\right.$$')
    st.text("")
    relu_der_fig = plot_function_derivative(relu, title='Derivative of the ReLU Function')
    st.plotly_chart(relu_der_fig)
    with st.expander('Plot Explanation'):
        st.write('- The derivative of the ReLU function is 1 for z > 0, and 0 for z < 0.')
        st.write('- The ReLU function is not differentiable at z = 0.')

    st.subheader('Pros')
    st.write("1. Computationally Efficient\n- The ReLU function does not require a lot of computation (Unlike the logistic and the tanh function which include an exponential function).\n- Because the ReLU function mimics a linear function when the input is positive, it is very easy to optimize.")
    st.write("2. Sparse Representaion\n- The ReLU function can output true zero values when the input is negative. This results in sparse weight matricies which help simplify the model architecure and speed up the learning process.\n- In contrast, the logistic and the tanh function always output non-zero values (sometimes the output is very close to zero, but not a true zero), which results in a dense representation.")
    st.write("3. Avoid Vanishing Gradients\n- The ReLU function does not saturate for positive values which helps avoid the vanishing gradient problem.\n- Switching from the logistic (sigmoid) activation function to ReLU has helped revolutionize the field of deep learning.")

    st.subheader('Cons')
    st.write("1. Dying ReLUs\n- A problem where ReLU neurons become inactive and only output 0 for any input.\n- This usually happens when the weighted sum of the inputs for all training examples is negative, coupled with a large learning rate.\n- This causes the ReLU function to only output zeros and gradient descent algorithm can not affect it anymore.\n- One of the explanation of this phenomenon is using symmetirc weight distributions to initialize weights and biases.")


## LeakyReLU Function
if activation_function == "LeakyReLU Function":
    st.title('Leaky ReLU Function')

    st.subheader('Description')
    st.write('A variant of the ReLU function that solves the dying ReLUs problem.')
    st.markdown(r'$LeakyReLU_{\alpha}(z) = max({\alpha}z, z)$')
    st.write('It will output the input directly if it is positive, but it will output (α * input) if it is negative.')
    st.write('This will ensure that the LeakyReLU function never dies.')

    st.subheader('Plot')
    with st.form('leakage'):
        alpha = st.slider('α Value', 0.0, 1.0, 0.2)
        st.form_submit_button('Apply Changes')

    def leaky_relu(z, alpha=alpha):
        return np.maximum(alpha*z, z)

    leaky_fig = plot_function(leaky_relu, title="LeakyReLU Function", alpha=alpha)
    st.plotly_chart(leaky_fig)

    with st.expander('Plot Explanation'):
        st.write("- This plot will automatically change when you change the value of α from the above slider.")
        st.write('- Notice that the output of the LeakyReLU function is never a true zero for negative inputs, which helps avoid the dying ReLUs problem.')
        st.write('- The value α is a hyperparameter that defines how much the function leaks.')
        st.write('- α represents the slope of the function when the input is negative.')
    
    st.subheader('Derivative')
    st.markdown(r'$$ LeakyRelu^{\prime}(z)= \left\{\begin{array}{ll}1 & z>0 \\{\alpha} & z<=0 \\\end{array}\right.$$')
    leaky_der_fig = plot_function_derivative(leaky_relu, title="Derivative of the LeakyReLU Function")
    st.plotly_chart(leaky_der_fig)
    with st.expander('Plot Explanation'):
        st.write("- This plot will automatically change when you change the value of α from the above slider.")
        st.write("- Notice that the derivative of the function when the input is negative is equal to the value of α.")
        st.write("- The derivative of the function is never a true zero for negative inputs.")
    
    st.subheader("Pros")
    st.write("1. Avoids the Dead ReLUs Problem\nBy allowing the function to have a small gradient when the input is negative, we ensure that the neuron never dies.")
    st.write("2. Better Performance\n The LeakyReLU function along with its variants almost always outperforms the standard ReLU.")

## Variants of LeakyReLU
if activation_function == 'Variants of LeakyReLU':
    st.title("Randomized LeakyReLU (RReLU)")
    st.write('In this variant, the value of α is picked randomly in a given range during training, and is fixed to an average during testing.')
    st.write('In addition to having the same advantages of the LeakyReLU, it also has a slight regularization effect (reduce overfitting).')

    st.title('Parametric LeakyReLU (PReLU)')
    st.write('In this variant, the value of α is a trainable (learnable) parameter rather than a hyperparameter.')
    st.write('In other words, the backpropagation algorithm can tweak its value like any other model parameter.')
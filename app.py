from click import style
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

def plot_function(func, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=func(z), mode='lines', line=dict(color='firebrick', width=2)))
    fig.update_layout(title = title, xaxis_title='Z',width=700, height=400,
                            font=dict(family="Courier New, monospace",size=16,color="White"), margin=dict(t=30, b=0, l=0, r=0))
    return fig

def plot_function_derivative(func, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=derivative(func, z), mode='lines', line=dict(color='firebrick', width=2)))
    fig.update_layout(title = title, xaxis_title='Z', width=700, height=400,
                            font=dict(family="Courier New, monospace",size=16,color="White"), margin=dict(t=30, b=0, l=0, r=0))
    return fig
#########################################


st.title('Activation Functions')

activation_function = st.selectbox('Choose an activation function', ['None', 'Logistic'])

## Logistic Function
if activation_function == 'Logistic':

    st.header('Logistic Function')

    st.subheader('Description')
    st.write('It is a sigmoid function with a characteristic "S"-shaped curve.')
    st.markdown(r'**$$sigmoid(z)=\frac{1}{1+exp(-z)}$$**')
    st.write('The output of the logistic (sigmoid) function is always between 0 and 1.')   

    st.subheader('Plot')
    logistic_fig  = plot_function(logistic, title='The Logistic Activation Functoin')
    logistic_fig.add_annotation(x=7, y=1, text='<b>Saturation</b>', showarrow=True,
     font=dict(family="Montserrat", size=16, color="#1F8123"),
        align="center",arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A835E1", ax=-10, ay=30,)
    logistic_fig.add_annotation(x=-7, y=0, text='<b>Saturation</b>', showarrow=True,
     font=dict(family="Montserrat", size=16, color="#1F8123"),
        align="center",arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A835E1", ax=-10, ay=-30,)
    st.plotly_chart(logistic_fig)
    with st.expander('Plot Explanation'):
        st.write('- The logistic function saturates as the inputs become larger (either positive or negative).')
        st.write('- For large positive and negative values, the function gets asymptotically close to 1 and 0, respectively.')
        st.write('- When the function saturates, its gradient becomes very close to zero, which slows down learning.')

    st.subheader('Derivative')
    st.markdown(r'$sigmoid^{\prime}(z)=sigmoid(z)(1−sigmoid(z))$')
    st.text("")
    logistic_der_fig = plot_function_derivative(logistic, title='The Derivative of the Logistic Function')
    st.plotly_chart(logistic_der_fig)
    with st.expander('Plot Explanation'):
        st.write('Notice that the derivative of the logistic function gets very close to zero for large positive and negative inputs.')

    st.subheader('Pros')
    st.write('1. The logistic function introduces non-linearity into the network which allows it to solve more complex problems than linear activation functions.\n2. It is continuous, differentiable, and have non-zero derivatives everywhere.')

    st.subheader('Cons')
    st.write("1. Vanishing Gradients\nBecause the logistic function can get easily saturated with large inputs, its gradient gets very close to zero. This causes the gradients to get smaller and smaller as backpropagation progresses down to the lower layers of the network. Eventually, the lower layers' weights receive very small updates never converge to their optimal values.")
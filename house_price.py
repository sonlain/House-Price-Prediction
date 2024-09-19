from lab_utils_multi import  load_house_data, run_gradient_descent
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
import streamlit as st
import copy,math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
np.set_printoptions(precision=2)
st.markdown("<h1 style='text-align: center;'>House Price Prediction</h1>", unsafe_allow_html=True)

bedroom = st.selectbox(
    'Select number of rooms:',
    ['1 BHK', '2 BHK', '3 BHK']
)
size = st.number_input('Enter size of house (In 1000 sqft):', min_value=0, step=1, value=100)
floor = st.number_input('Enter number of floors:', min_value=0, step=1)
age = st.number_input('Enter age of house (In Years):', min_value=0, step=1)

bed_dict={'1 BHK':1, '2 BHK':2, '3 BHK':3}
bed=int(bed_dict[bedroom])
# Display the input values
# st.write(f"Number of Rooms: {bedroom}")
# st.write(f"House Size: {size} sqft")
# st.write(f"Number of Floors: {floor}")
# st.write(f"Age of House: {age} years")

def create_rooms_with_partitions_top_view(num_rooms,size):
    # Define the room dimensions
    j=np.sqrt(size)
    x=j/10
    print(x)
    room_length = j/x
    room_width = j/x
    
    # Create a figure for the top view
    fig, ax = plt.subplots(figsize=(3,3))

    # Draw the main room with a brown bold boundary

    ax.add_patch(Rectangle((0, 0), room_length+0.01, room_width+0.01, color='blue', edgecolor='black', linewidth=4))
    above_rectangle = Rectangle((0.01, 0.01), room_length, room_width, 
                            color='lightgrey', edgecolor='Black', linewidth=1)  # 1 inch in points

# Add the rectangle to the axes
    ax.add_patch(above_rectangle)

    # Calculate the number of partitions needed
    num_partitions = num_rooms - 1

    # Draw vertical partitions based on the number of rooms
    if num_partitions > 0:
        partition_positions = np.linspace(0, room_length, num_partitions + 1)[1:-1]
        for pos in partition_positions:
            # Draw vertical partition
            ax.add_patch(Rectangle((pos, 0), 0.1, room_width, color='blue'))

    # Set the limits and labels
    ax.set_xlim(-1,11)
    ax.set_ylim(-1,11)
    ax.axis('off')
    # ax.set_aspect('equal')
    # ax.set_xlabel('Unit')
    # ax.set_ylabel('Unit')
    # plt.title(f'Scale 1 Unit is {1000*x:0.02f} ft')
    
    # Show the plot
    plt.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    # plt.grid()
    # plt.show()
    st.pyplot(fig)

# Example usage
# Change this number for different numbers of rooms


#Price of Home
def z_score_normalize_features(X):
  mu=np.mean(X,axis=0)
  sigma=np.std(X,axis=0)
  X_norm=(X-mu)/sigma
  return (X_norm,mu,sigma)

X_train,y_train=load_house_data()
X_norm,X_mu,X_sigma=z_score_normalize_features(X_train)
X_features=['Size(sqft)','Bedrooms','Floors','Age']
def result(size,bedroom,floor,age):
  x_house = np.array([size,bedroom,floor,age])
  x_house_norm = (x_house - X_mu) / X_sigma
  x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
  x = 42  # Example value of x
  st.markdown(f"<h3 style='text-align: center;'>Predicted Price : ${1000*x_house_predict:0.02f}</h3>", unsafe_allow_html=True)
  #st.markdown("<h3 style='text-align: center;'>**Predicted Price : {x_house_predict}**</h3>", unsafe_allow_html=True)
  #st.write(f" Predicted Price of a House with {size*1000} sqft, {bedroom} Bedrooms, {floor} Floor, {age} Years old cost ${x_house_predict*1000:0.0f}",)
def feature():
    fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True,facecolor='red')
    fig.set_facecolor('Orange')
    for i in range(len(ax)):
        ax[i].set_facecolor('white')
        ax[i].scatter(X_train[:,i],y_train,color='b')
        ax[i].set_xlabel(X_features[i],color='k')
        ax[i].tick_params(axis='x', colors='k')  # X-axis tick labels and marks
        ax[i].tick_params(axis='y', colors='k') 
    ax[0].set_ylabel('Price 1000$',color='k')
    st.pyplot(fig)

def normalized():
    #Difference between Unnormalized,noramlized and Z-Score normalized
    mu=np.mean(X_train,axis=0)
    sigma=np.std(X_train,axis=0)
    X_mean=X_train-mu
    X_norm=(X_mean)/sigma

    fig, ax =plt.subplots(1,3,figsize=(12,3))
    ax[0].scatter(X_train[:,0], X_train[:,3])
    ax[0].set_xlabel(X_features[0])
    ax[0].set_ylabel(X_features[3])
    ax[0].set_title('Unnormalized')
    ax[0].axis('equal')

    ax[1].scatter(X_mean[:,0], X_mean[:,3])
    ax[1].set_xlabel(X_features[0])
    ax[1].set_ylabel(X_features[3])
    ax[1].set_title('Normalized')
    ax[1].axis('equal')

    ax[2].scatter(X_norm[:,0], X_norm[:,3])
    ax[2].set_xlabel(X_features[0])
    ax[2].set_ylabel(X_features[3])
    ax[2].set_title(r'Standardized')
    ax[2].axis('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Algorithms for Feature Normalization')
    st.pyplot(fig)
def distribution_feature():
    fig,ax=plt.subplots(1,4,figsize=(12,3))
    for i in range(len(ax)):
        norm_plot(ax[i],X_train[:,i])
        ax[i].set_xlabel(X_features[i])
        ax[i].set_ylabel('Count')
    fig.suptitle('Before Normalization')
    st.pyplot(fig)
    fig,ax=plt.subplots(1,4,figsize=(12,3))
    for i in range(len(ax)):
        norm_plot(ax[i],X_norm[:,i])
        ax[i].set_xlabel(X_features[i])
        ax[i].set_ylabel('Count')
    fig.suptitle('After Normalization')
    st.pyplot(fig)

def predict_target():
    m=X_norm.shape[0]
    yp=np.zeros(m)
    for i in range(m):
        yp[i]=np.dot(X_norm[i],w_norm)+b_norm
    fig,ax=plt.subplots(1,4,figsize=(12,3))
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train,label='target')
        ax[i].set_xlabel(X_features[i])
        ax[i].set_ylabel('Price')
        ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"],label='predict')
        ax[i].legend()
    fig.suptitle("Target vs Predicted using Z score")
    st.pyplot(fig)
X_norm,X_mu,X_sigma=z_score_normalize_features(X_train)
w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )
result(size,bed,floor,age)
create_rooms_with_partitions_top_view(bed+1,size)


st.header('Linear Regression with multiple Variable')
st.markdown('Features Vs Price')
feature()
st.markdown('Features')
normalized()
st.markdown('Distribution of Features')
distribution_feature()
st.markdown('Predict Target Using Normalized Features')
predict_target()






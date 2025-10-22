import streamlit as st
from torchvision import models, transforms
from torchvision.models import vit_b_16
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from nutrition_data import nutrition
from huggingface_hub import hf_hub_download
HF_REPO_ID = "Trid3nz/indonesian-food-models"

# Page configuration
st.set_page_config(
    page_title="Food Recognition App",
    page_icon="🍕",
    layout="wide"
)

# Food class labels (must match order used during model training)
class_labels = [
    'Ayam Goreng', 'Burger', 'French Fries', 'Gado-Gado',
    'Ikan Goreng', 'Mie Goreng', 'Nasi Goreng', 'Nasi Padang', 'Pizza',
    'Rawon', 'Rendang', 'Sate', 'Soto'
]

# Preprocessing pipeline for classification models (ResNet, ViT)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@st.cache_resource
def download_from_huggingface(filename):
    """Download model from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            cache_dir="./models"
        )
        return model_path
    except Exception as e:
        st.error(f"Error downloading {filename}: {e}")
        return None

# Load models (cache them to avoid reloading)
@st.cache_resource
def load_resnet_model():
    """Load ResNet model"""
    try:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(class_labels))
        model.load_state_dict(torch.load('models/indonesia_food_resnet18.pth', map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading ResNet: {e}")
        return None

@st.cache_resource
def load_vit_model():
    """Load ViT model"""
    try:
        model = vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(class_labels))
        model.load_state_dict(torch.load('models/indonesia_food_vit.pth', map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading ViT: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    """Load YOLO model"""
    try:
        model = YOLO('models/epoch10batch32YOLO.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO: {e}")
        return None

# Classification utility function
def predict_class(model, image):
    """Predict class using ResNet or ViT"""
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
    return class_labels[predicted.item()]

def predict_yolo(model, image):
    """Predict using YOLO and return labels and annotated image"""
    # Convert PIL Image to numpy array for OpenCV
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run YOLO prediction
    results = model(img_cv)[0]
    annotated = results.plot()
    
    # Convert back to RGB for display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Extract unique labels
    labels = set()
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        labels.add(model.names[cls_id])
    
    return labels, annotated_rgb

def get_nutrition_info(food_items):
    """Get nutrition information for detected food items"""
    nutrition_results = []
    
    if isinstance(food_items, str):
        food_items = [food_items]
    
    for item in food_items:
        if item in nutrition:
            nutrition_results.append({
                'name': item,
                **nutrition[item]
            })
        else:
            nutrition_results.append({
                'name': item,
                'note': 'Nutrition data not available.'
            })
    
    return nutrition_results

# Title and description
st.title("🍕 Indonesian Food Recognition App")
st.write("Upload an image and select a model to identify Indonesian food items")

# Sidebar for model selection
with st.sidebar:
    st.header("Model Selection")
    model_choice = st.selectbox(
        "Choose a model:",
        options=["ResNet", "ViT", "YOLO"],
        index=0,
        help="Select the deep learning model for food recognition"
    )
    
    st.divider()
    
    st.header("About Models")
    if model_choice == "ResNet":
        st.write("**ResNet-18** - Residual Neural Network for image classification")
    elif model_choice == "ViT":
        st.write("**Vision Transformer** - Transformer-based architecture for image classification")
    else:
        st.write("**YOLOv8** - Object detection model that can detect multiple food items")
    
    st.divider()
    
    st.header("Supported Foods")
    with st.expander("View all food categories"):
        for food in class_labels:
            st.write(f"• {food}")

# Main app layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a food image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an Indonesian food image for recognition"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.header("🔍 Prediction Results")
    
    if uploaded_file is not None:
        # Add a predict button
        if st.button("Analyze Image", type="primary", use_container_width=True):
            with st.spinner(f"Running {model_choice} model..."):
                try:
                    if model_choice == "ResNet":
                        # Load model and predict
                        model = load_resnet_model()
                        if model is not None:
                            prediction = predict_class(model, image)
                            
                            # Display prediction
                            st.success("Analysis Complete!")
                            st.subheader(f"🍽️ Detected: **{prediction}**")
                            
                            # Get and display nutrition info
                            nutrition_results = get_nutrition_info(prediction)
                            
                    elif model_choice == "ViT":
                        # Load model and predict
                        model = load_vit_model()
                        if model is not None:
                            prediction = predict_class(model, image)
                            
                            # Display prediction
                            st.success("Analysis Complete!")
                            st.subheader(f"🍽️ Detected: **{prediction}**")
                            
                            # Get and display nutrition info
                            nutrition_results = get_nutrition_info(prediction)
                            
                    else:  # YOLO
                        # Load model and predict
                        model = load_yolo_model()
                        if model is not None:
                            labels, annotated_image = predict_yolo(model, image)
                            
                            # Display annotated image
                            st.success("Analysis Complete!")
                            st.image(annotated_image, caption="Detection Results", use_container_width=True)
                            
                            # Display detected items
                            if labels:
                                st.subheader(f"🍽️ Detected: **{', '.join(labels)}**")
                                nutrition_results = get_nutrition_info(list(labels))
                            else:
                                st.warning("No food items detected")
                                nutrition_results = []
                    
                    # Display nutrition information
                    if 'nutrition_results' in locals() and nutrition_results:
                        st.divider()
                        st.subheader("📊 Nutrition Information")
                        
                        for item in nutrition_results:
                            with st.expander(f"🍴 {item['name']}", expanded=True):
                                if 'note' in item:
                                    st.info(item['note'])
                                else:
                                    cols = st.columns(2)
                                    with cols[0]:
                                        st.metric("Calories", f"{item.get('calories', 'N/A')} kcal")
                                        st.metric("Protein", f"{item.get('protein', 'N/A')} g")
                                    with cols[1]:
                                        st.metric("Carbs", f"{item.get('carbs', 'N/A')} g")
                                        st.metric("Fat", f"{item.get('fat', 'N/A')} g")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.info("👆 Upload an image to get started")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Powered by ResNet-18, Vision Transformer (ViT), and YOLOv8</p>
    <p>Indonesian Food Recognition System</p>
</div>
""", unsafe_allow_html=True)
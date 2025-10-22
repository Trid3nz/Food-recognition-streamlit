# 🍜 Indonesian Food Recognition App

> **Because life's too short to wonder "What am I eating?" 🤔**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD21E?style=for-the-badge)](https://huggingface.co)

An AI-powered food recognition system that can identify Indonesian dishes faster than you can say "Nasi Goreng"! 🚀

---

## 🎯 What Does This Do?

Ever taken a photo of food and wondered what it is? Well, wonder no more! This app uses **THREE** powerful AI models to:

- 📸 Identify Indonesian food from images
- 🍽️ Detect multiple dishes in one photo (YOLO is watching!)
- 📊 Show you nutrition information (so you can pretend to care about calories)
- 🎨 Give you beautiful bounding boxes (because who doesn't love boxes?)

---

## 🤖 The AI Dream Team

We've assembled the Avengers of deep learning models:

### 1️⃣ **ResNet-18**

_The Reliable Veteran_ 🎖️

- Classic, battle-tested architecture
- Fast and accurate classification
- Your go-to for quick identification

### 2️⃣ **Vision Transformer (ViT)**

_The Modern Genius_ 🧠

- Attention is all you need (literally!)
- State-of-the-art architecture
- Sees food like you've never seen it before

### 3️⃣ **YOLOv8**

_The Show-off_ 🎯

- "You Only Look Once" but it's worth it!
- Detects multiple foods simultaneously
- Draws fancy boxes around your rendang

---

## 🍱 Supported Indonesian Dishes

Our AI has mastered the art of recognizing these 13 delicious dishes:

|   🍗 Ayam Goreng   |     🍔 Burger     |  🍟 French Fries   |    🥗 Gado-Gado    |
| :----------------: | :---------------: | :----------------: | :----------------: |
| 🐟 **Ikan Goreng** | 🍜 **Mie Goreng** | 🍚 **Nasi Goreng** | 🍛 **Nasi Padang** |
|    🍕 **Pizza**    |   🍲 **Rawon**    |   🍖 **Rendang**   |   � 串 **Sate**    |
|    🥣 **Soto**     |                   |                    |                    |

_Yes, we know Pizza and Burger aren't Indonesian... but they're honorary members! 🎉_

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A stomach for adventure (optional but recommended)

### Installation

```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/indonesian-food-recognition.git
cd indonesian-food-recognition

# Create virtual environment
python -m venv food-env

# Activate it
# Windows:
food-env\Scripts\activate
# Mac/Linux:
source food-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

🎉 **Boom!** Your app should open in your browser at `http://localhost:8501`

---

## 📖 How to Use

1. **Select Your Champion** 🏆

   - Choose between ResNet, ViT, or YOLO from the sidebar

2. **Upload a Food Photo** 📸

   - Click "Browse files" or drag & drop
   - Supports JPG, JPEG, PNG

3. **Hit "Analyze Image"** 🔍

   - Watch the AI work its magic
   - Get predictions faster than you can open a food delivery app

4. **Check Nutrition Info** 📊
   - See calories, protein, carbs, and fat
   - Feel guilty about that second serving of rendang

---

## 🎨 Features That'll Make You Smile

- ✨ **Clean & Modern UI** - Because ugly apps are so 2010
- 🎯 **Real-time Detection** - Faster than your friend saying "Wait, let me take a photo first!"
- 📊 **Nutrition Dashboard** - For when you pretend to diet
- 🎭 **Three Model Options** - Choose your fighter!
- 🖼️ **Visual Annotations** - YOLO draws boxes prettier than your art teacher
- 💾 **Smart Model Caching** - Downloads once, runs forever (well, until you close it)

---

## 🏗️ Architecture

```
📁 Project Structure
├── 📄 app.py                    # Main Streamlit application
├── 📄 nutrition_data.py         # Nutrition information database
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # You are here! 👋
└── 📁 models/                   # Model weights (downloaded from HuggingFace)
    ├── indonesia_food_resnet18.pth
    ├── indonesia_food_vit.pth
    └── epoch10batch32YOLO.pt
```

---

## 🎓 Technical Details (For the Nerds)

### Models

- **ResNet-18**: 18-layer residual network with custom classification head
- **ViT-B/16**: Base Vision Transformer with 16x16 patch size
- **YOLOv8**: Latest YOLO architecture for object detection

### Training

- Dataset: Custom Indonesian food dataset
- Input Size: 224x224 pixels
- Preprocessing: Standard ImageNet normalization
- Framework: PyTorch 2.0+

### Deployment

- **Frontend**: Streamlit
- **Model Hosting**: Hugging Face Hub
- **Inference**: CPU-optimized (GPU optional)

---

## 📊 Model Performance

| Model     | Accuracy   | Speed  | Best For                     |
| --------- | ---------- | ------ | ---------------------------- |
| ResNet-18 | ⭐⭐⭐⭐   | 🚀🚀🚀 | Single dish classification   |
| ViT       | ⭐⭐⭐⭐⭐ | 🚀🚀   | High accuracy needs          |
| YOLO      | ⭐⭐⭐⭐   | 🚀🚀🚀 | Multiple dishes in one image |

---

## 🤝 Contributing

Found a bug? Want to add more Indonesian dishes? We'd love your help!

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🐛 Known Issues

- Model might confuse Mie Goreng with Nasi Goreng (they're both delicious anyway!)
- YOLO sometimes gets too excited and draws boxes around the plate
- App might make you hungry (not a bug, it's a feature!)

---

## 📝 TODO

- [ ] Add more Indonesian dishes (Gudeg, Pempek, Martabak... the list goes on!)
- [ ] Recipe suggestions based on detected food
- [ ] Calorie counter that judges your life choices
- [ ] Integration with food delivery apps (because looking at food makes you hungry)
- [ ] Mobile app version
- [ ] AR mode (point your camera, see nutrition info in real-time!)

---

## 🙏 Acknowledgments

- **PyTorch Team** - For making deep learning accessible
- **Ultralytics** - For YOLO (You're awesome!)
- **Streamlit** - For making deployment stupid-easy
- **Hugging Face** - For hosting our chunky models
- **Indonesian Cuisine** - For being absolutely delicious
- **Coffee** - For keeping the developer awake during training

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Translation: Do whatever you want with it, just don't blame us if your model thinks your cat is rendang. 🐱

---

## 🎬 Demo

🌐 **Live Demo**: [Try it here!](https://huggingface.co/spaces/YOUR_USERNAME/indonesian-food-recognition)

---

## 📞 Contact

Got questions? Found a dish our AI can't recognize? Hit us up!

- 📧 Email: your.email@example.com
- 🐦 Twitter: @yourhandle
- 💼 LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- 🌐 Portfolio: [yourwebsite.com](https://yourwebsite.com)

---

## 🌟 Star This Repo!

If this project helped you identify your lunch or made you smile, give it a ⭐!

It costs nothing but means everything to developers who run on coffee and validation. ☕❤️

---

<div align="center">

### Made with ❤️, 🍜, and way too much ☕

**Now stop reading and go recognize some food!** 🚀

</div>

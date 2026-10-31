
<div align="center">

  <h1>🤟 Sign Language Recognition</h1>
  <h3>Real-Time Sign Language Recognition Using Deep Learning</h3>

  <p>
    <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow&logoColor=white">
    <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white">
    <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">
  </p>

  <p>
    A real-time hand gesture recognition system built with Python,
    OpenCV, and TensorFlow.
  </p>

</div>

---

<h2>📌 About The Project</h2>

<p>
  <strong>Sign Language Recognition</strong> is a computer vision and deep learning project
  designed to recognize hand gestures and classify them into sign language alphabet letters.
</p>

<p>
  The system captures hand gestures through a webcam, detects the hand using
  <strong>cvzone</strong>, processes the image with <strong>OpenCV</strong>,
  and predicts the corresponding letter using a
  <strong>Convolutional Neural Network (CNN)</strong> developed with TensorFlow/Keras.
</p>

<p><strong>Currently Supported Classes:</strong></p>

<div align="center">
  <img src="https://img.shields.io/badge/A-Class-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/B-Class-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/C-Class-blue?style=for-the-badge">
</div>

---

<h2>✨ Features</h2>

<ul>
  <li>Real-time hand detection using webcam</li>
  <li>Custom dataset collection</li>
  <li>Hand cropping and image preprocessing</li>
  <li>Image resizing and background normalization</li>
  <li>CNN-based image classification</li>
  <li>Real-time prediction</li>
  <li>Prediction confidence display</li>
  <li>Modular and extendable architecture</li>
</ul>

---

<h2>🛠️ Technologies & Libraries</h2>

<table>
  <tr>
    <th>Technology</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td>Python 3.11.9</td>
    <td>Main Programming Language</td>
  </tr>
  <tr>
    <td>OpenCV</td>
    <td>Image Processing and Webcam Access</td>
  </tr>
  <tr>
    <td>cvzone</td>
    <td>Hand Detection and Tracking</td>
  </tr>
  <tr>
    <td>NumPy</td>
    <td>Numerical Operations</td>
  </tr>
  <tr>
    <td>TensorFlow 2.15</td>
    <td>Deep Learning Framework</td>
  </tr>
  <tr>
    <td>Keras</td>
    <td>Neural Network Development</td>
  </tr>
</table>

---

<h2>📂 Project Structure</h2>

<pre>
Sign Language Recognition/
│
├── Data/
│   ├── A/
│   ├── B/
│   └── C/
│
├── output/
│   ├── keras_model.h5
│   └── labels.txt
│
├── src/
│   ├── dataset_gn.py
│   ├── train.py
│   └── sign_detection.py
│
├── .gitignore
└── README.md
</pre>

---

<h2>⚙️ Installation</h2>

<h3>1. Clone the Repository</h3>

<pre><code>git clone https://github.com/Soheilsamandar/Sign-Language-Recognition.git</code></pre>

<h3>2. Navigate to the Project Directory</h3>

<pre><code>cd Sign-Language-Recognition</code></pre>

<h3>3. Create a Virtual Environment</h3>

<pre><code>python -m venv .venv</code></pre>

<h3>4. Activate the Environment</h3>

<p><strong>Windows PowerShell:</strong></p>

<pre><code>.\.venv\Scripts\Activate.ps1</code></pre>

<p><strong>Windows CMD:</strong></p>

<pre><code>.venv\Scripts\activate</code></pre>

<h3>5. Install Dependencies</h3>

<pre><code>pip install opencv-python cvzone numpy tensorflow==2.15.0</code></pre>

---

<h2>🚀 How To Run</h2>

<h3>1. Dataset Collection</h3>

<p>Run the dataset collection script:</p>

<pre><code>python src/dataset_gn.py</code></pre>

<ul>
  <li>Press <strong>S</strong> to capture and save an image.</li>
  <li>Press <strong>ESC</strong> to exit.</li>
</ul>

<p>
  Before collecting images, select the desired class folder
  inside <code>dataset_gn.py</code>.
</p>

<h3>2. Model Training</h3>

<p>After collecting images for all classes, run:</p>

<pre><code>python src/train.py</code></pre>

<p>
  The training script loads the dataset, preprocesses the images,
  trains the CNN model, and saves the best model.
</p>

<p>Generated files:</p>

<pre>
output/
├── keras_model.h5
└── labels.txt
</pre>

<h3>3. Real-Time Recognition</h3>

<p>Run the recognition script:</p>

<pre><code>python src/sign_detection.py</code></pre>

<p>
  Show your hand to the webcam. The application displays
  the predicted letter and its confidence score.
</p>

<p>Press <strong>ESC</strong> to close the application.</p>

---

<h2>🧠 Model Architecture</h2>

<p>
  The classification model is based on a
  <strong>Convolutional Neural Network (CNN)</strong>
  implemented using TensorFlow/Keras.
</p>

<table>
  <tr>
    <th>Parameter</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Model Type</td>
    <td>CNN</td>
  </tr>
  <tr>
    <td>Input Image Size</td>
    <td>224 × 224</td>
  </tr>
  <tr>
    <td>Number of Classes</td>
    <td>3</td>
  </tr>
  <tr>
    <td>Classes</td>
    <td>A, B, C</td>
  </tr>
  <tr>
    <td>Optimizer</td>
    <td>Adam</td>
  </tr>
  <tr>
    <td>Loss Function</td>
    <td>Sparse Categorical Crossentropy</td>
  </tr>
  <tr>
    <td>Output Activation</td>
    <td>Softmax</td>
  </tr>
  <tr>
    <td>Framework</td>
    <td>TensorFlow / Keras</td>
  </tr>
</table>

<h3>Image Processing Pipeline</h3>

<ol>
  <li>Capture an image from the webcam.</li>
  <li>Detect the hand using cvzone.</li>
  <li>Crop the detected hand region.</li>
  <li>Resize the image and place it on a white background.</li>
  <li>Convert the image from BGR to RGB.</li>
  <li>Resize the image to the model input dimensions.</li>
  <li>Pass the processed image to the trained model.</li>
  <li>Display the predicted class and confidence score.</li>
</ol>

---

<h2>📊 Training Results</h2>

<p>The current training run achieved the following results:</p>

<table>
  <tr>
    <th>Metric</th>
    <th>Result</th>
  </tr>
  <tr>
    <td>Training Accuracy</td>
    <td>92.5%</td>
  </tr>
  <tr>
    <td>Validation Accuracy</td>
    <td>100%</td>
  </tr>
</table>

<p>
  <em>
    Note: These results are based on a relatively small dataset.
    Real-world performance may vary depending on lighting,
    backgrounds, hand positions, and different users.
  </em>
</p>

---

<h2>🔮 Future Improvements</h2>

<ul>
  <li>Support more letters of the sign language alphabet.</li>
  <li>Expand the dataset with more diverse hand gestures.</li>
  <li>Improve model generalization and accuracy.</li>
  <li>Support multiple hand gestures.</li>
  <li>Develop sentence-level sign language recognition.</li>
  <li>Create a graphical user interface.</li>
  <li>Optimize the model for mobile applications.</li>
</ul>

---

<h2>👨‍💻 Author</h2>

<div align="center">

  <h3>Soheil Samandar</h3>

  <p>
    Computer Engineering Student<br>
    Python Developer | AI & Machine Learning Enthusiast
  </p>

  <a href="https://github.com/Soheilsamandar">
    <img src="https://img.shields.io/badge/GitHub-Personal%20Profile-181717?style=for-the-badge&logo=github&logoColor=white" alt="Personal GitHub">
  </a>

  <a href="https://github.com/AlphaRoboticsTeam">
    <img src="https://img.shields.io/badge/GitHub-Alpha%20Robotics%20Team-blue?style=for-the-badge&logo=github&logoColor=white" alt="Alpha Robotics Team">
  </a>

  <a href="seyedsoheilsamandar12@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact%20Me-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>

</div>

---

<div align="center">

  <p>⭐ If you find this project interesting, consider giving it a star!</p>

  <p>Made with ❤️ by <strong>Soheil Samandar</strong></p>

</div>

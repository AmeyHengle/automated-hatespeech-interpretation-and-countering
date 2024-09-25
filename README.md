# Production Prototype:  Automated Hate Speech Interpretation and Counterspeech Generation.


## Project Structure

```
automated-hatespeech-interpretation-and-countering/
│
├── app.py                # Flask application (Backend logic)
├── templates/
│   └── index.html        # Main HTML file (Frontend)
├── static/
│   ├── styles.css        # CSS file (Styling)
│   └── script.js         # JavaScript file (Frontend logic)
├── requirements.txt      # Python dependencies
└── Procfile              # For deployment (e.g., Heroku)
```

### Major Files

1. **app.py**  
   Contains the backend logic using Flask. It handles routes for analyzing the statement and generating counterspeech with related facts.

2. **templates/index.html**  
   Frontend HTML file where users interact with the app. It includes input fields, a loading spinner, and sections for displaying results like counterspeech and facts.

3. **static/styles.css**  
   Custom CSS to style the webpage, including layout, dialog boxes, and loader styling.

4. **static/script.js**  
   Contains the JavaScript logic for interacting with the backend, displaying results, and handling user actions like submitting the hate speech statement or generating counterspeech.

5. **requirements.txt**  
   Specifies the Python dependencies for the project, such as Flask.

6. **Procfile**  
   Defines how the app is run, useful for deploying the app on platforms like Heroku.

---

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/hate-speech-analysis.git
   cd hate_speech_demo
   ```

2. **Install dependencies**:
   Use the following command to install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application locally**:
   To run the Flask app, use:
   ```bash
   python app.py
   ```
   The app will be available at `http://127.0.0.1:5000/`.

---

## Deployment

To deploy the app on Heroku:

1. **Log in to Heroku** (if you haven’t):
   ```bash
   heroku login
   ```

2. **Create a Heroku app**:
   ```bash
   heroku create
   ```

3. **Push the code to Heroku**:
   ```bash
   git push heroku master
   ```

4. **Visit the live app**:
   Once deployed, visit `https://your-app-name.herokuapp.com/`.

---
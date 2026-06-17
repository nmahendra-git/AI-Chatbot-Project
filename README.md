AI-Powered Chatbot using NLP and Flask

This project is an intent-based AI chatbot developed using core Natural Language Processing and Machine Learning principles. The chatbot understands what a user is trying to say by classifying their message into predefined categories called intents, and responds with an appropriate reply.
The system is built entirely in Python and uses scikit-learn for training a text classification model. User messages are converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) and then classified using Logistic Regression. The trained model is saved and loaded by a Flask web server that exposes a REST API. A clean, responsive chat UI built with HTML, CSS, and JavaScript connects to this API in real time.


🎯 Objectives

Understand how businesses use chatbots to handle large volumes of queries
Apply NLP techniques like tokenization and TF-IDF vectorization
Train a machine learning classifier on a custom JSON intent dataset
Build and serve a REST API using Flask
Design a functional chat interface with real-time responses

🔧 How It Works
User types message
       ↓
Flask API receives it
       ↓
TF-IDF converts text → numbers
       ↓
Logistic Regression predicts intent
       ↓
Random response picked for that intent
       ↓
Response sent back to chat UI

💡 Key Highlights

13 built-in intents — greetings, jokes, help, contact, weather, and more
Confidence scoring — every response shows how confident the model is
Fallback handling — gracefully handles unknown inputs
Extensible design — add new intents by editing one JSON file, no code changes needed
REST API — can be connected to any frontend or messaging platform


🛠 Technologies Used
	CategoryToolsLanguagePython 3.9+ML / NLPscikit-learn, TF-IDF, Logistic RegressionBackendFlaskFrontendHTML, CSS, JavaScriptDataJSON

📚 Learning Outcomes
	After completing this project you will understand:

How NLP pipelines work from raw text to prediction
How to train and save a machine learning model
How to build and consume a REST API
How to connect a frontend UI to a Python backend
How chatbots are structured and deployed in real applications


🔮 Scope for Improvement

	Upgrade to BERT or DistilBERT for better accuracy
	Add conversation memory to handle multi-turn dialogue
	Deploy to cloud platforms like Heroku, Render, or AWS
	Integrate with WhatsApp or Telegram using their APIs
	Add a database for logging conversations



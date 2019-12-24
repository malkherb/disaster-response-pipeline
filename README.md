# Disaster Response Pipeline Project

This project's objective is to create a pipeline that categorizes emergency messages according to the needs communicated by the sender.

The following files are in repo:
    - app folder:
        1- run.py: a python file is used to initiate the web app.
        2- templates: a folder containing supporting interface files.
    - data:
        1- disaster_messages.csv: the messages dataset
        2- disaster_categories.csv: the categories dataset
        3- disaster.db: the exported database
        4- process_data.py: a python file is used to cleanse and preprocess the datasets.
    - models:
        1- train_classifier.py: a python file that is used to build the classification model.
        2- classifier.pkl: the exported model that will be used to predict and classify the messages

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

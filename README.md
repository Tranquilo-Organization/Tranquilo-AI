# Tranquilo-Ml-Model
# Anxiety Level and GAD score classification Using AI
## Project Description
We often suffer from anxiety, especially during difficult and important times, such as exams and many other things 
Many people are not aware of the potential factors that increase their anxiety buildup apart from their work. The AI ​​model is trained on a dataset. Key factors include but are not limited to sleep quality, blood pressure, headaches, and self-esteem. People will be given a rating from the AI ​​model based on their suggested anxiety score and anxiety levels. This model is part of our project Tranquilo
# **Explanation of each feature within the dataset**
| Column	 | Description|
|----------|----------|
| age	 | Age of participants|
| gender	 |Gender of each participant |
| bmi	 | body mass index|
| who_bmi		 |bmi category |
| phq_score	 | measure the severity of symptoms related to depression, anxiety, andother related disorders in patients|
| depression_severity	 |degree or intensity of symptoms experienced by an individual with depression |
| depressiveness		 | Is he suffering from depression|
| suicidal		 | the candidate have suicide thought|
| depression_diagnosis		 |the candidate already have depression diagnosis|
| depression_treatment		 | the candidate already have depression treatment|
| gad_score			 |measure that assesses the severity of Generalized Anxiety Disorder|
| anxiety_severity		 | intensity of symptoms experienced by an individual wit anxiety|
| anxiousness		 |Feeling anxiety or not |
| anxiety_diagnosis			 |the candidate already have anxiety diagnosis |
| anxiety_treatment		 |the candidate already have anxiety treatment |
| epworth_score		 |score to assess daytime sleepiness daytime sleepiness |
| sleepiness	 |Feel sleepy or not |
# **Analysis of Each Prominent File/Directory**
`app.py`: The main Flask file that operates all backend services when routing webpages to the client. Aside from loading the AI model to be used alongside the web app, this file is responsible for providing the webpages with sufficient data personalized to each client. It takes any important information stored from a db file in the static directory and routes it to a Flask template for further manipulation. This file also starts Flask and notifies Flask where to find the webpages in the filesystem.

`model.ipynb`: The Python file responsible for building an AI model and handling classifaction when given. The file is made from scratch, This file is mainly responsible for manipulating instances of the AI model, with methods like viewing its dataset and converting GAD-7 numbers A test scenario is briefly documented.

`Understand_data.ipynb`:This file is for understanding the data set and the features in it, identifying the defects in it, and some illustrative visualization of some of the things in it.

`requirements.txt`:A file showing all the libraries used and their versions.

`model.pkl`:The file contains the pre-trained machine learning model used to predict anxiety levels, which we will use in the development process.

`Anxiety.csv`:This dataset is used in the model.

`templates`:It contains the HTML file used to build the api page.

`static`:It contains a css file to make the necessary formats on the html file.





from flask import Flask, render_template, request

# Initialize your Flask app
app = Flask(__name__)

app.static_folder = 'static'

# Define a route to display the input form
@app.route('/')
def input_form():
    return render_template('input_form.html')

@app.route('/empathy.html')
def empathy():
    return render_template('empathy.html')

@app.route('/define.html')
def define():
    return render_template('define.html')

@app.route('/ideate.html')
def ideate():
    return render_template('ideate.html')

@app.route('/prototype.html')
def prototype():
    return render_template('prototype.html')

@app.route('/test.html')
def test():
    return render_template('test.html')

@app.route('/author.html')
def author():
    return render_template('author.html')


# Define a route to handle form submission and display results
@app.route('/predict', methods=['POST'])
def predict():
    Ia = float(request.form.get('Ia'))
    Ib = float(request.form.get('Ib'))
    Ic = float(request.form.get('Ic'))
    Va = float(request.form.get('Va'))
    Vb = float(request.form.get('Vb'))
    Vc = float(request.form.get('Vc'))

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd  # read the dataset
    import numpy as np  # numerical python
    import matplotlib.pyplot as plt  # plot the graph
    import seaborn as sns  # plot gaphical

    df = pd.read_csv('dataset/fault_full_data.csv')
    x = df.iloc[:, 0:6]
    y = df.iloc[:, 6:10]

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    multi_output_classifier = MultiOutputClassifier(rf_classifier, n_jobs=-1)
    multi_output_classifier.fit(X_train, y_train)
    y_pred = multi_output_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)

    # Replace this with your actual input values
    single_sample = [Ia, Ib, Ic, Va, Vb, Vc]
    # Make a prediction for the single sample
    predicted_output = multi_output_classifier.predict([single_sample])

    # Print the predicted output for the single sample
    print("Predicted Output:", predicted_output)

    predicted_output_str = str(predicted_output)

    # Assuming you have a single predicted output as a string
      # Replace this with your actual predicted output

    # Convert the string to a list of lists
    predicted_output = [list(map(int, predicted_output_str.strip('[] ').split()))]

    # Define your conditions (as lists)
    condition1 = [0, 0, 0, 0]
    condition2 = [1, 0, 0, 0]
    condition3 = [0, 0, 0, 1]
    condition4 = [0, 0, 1, 0]
    condition5 = [0, 1, 0, 0]
    condition6 = [1, 0, 0, 1]
    condition7 = [1, 0, 1, 0]
    condition8 = [1, 1, 0, 0]
    condition9 = [0, 0, 1, 1]
    condition10 = [0, 1, 1, 0]
    condition11 = [0, 1, 0, 1]
    condition12 = [1, 0, 1, 1]
    condition13 = [1, 1, 0, 1]
    condition14 = [1, 1, 1, 0]
    condition15 = [0, 1, 1, 1]
    condition16 = [1, 1, 1, 1]

    # Check each condition using if statements
    if predicted_output == [condition1]:
        output_text = "No Fault"
        print(output_text)
    elif predicted_output == [condition2]:
        output_text = "Groud Fault"
        print(output_text)
    elif predicted_output == [condition3]:
        output_text = "Fault in Line A"
        print(output_text)
    elif predicted_output == [condition4]:
        output_text = "Fault in Line B"
        print(output_text)
    elif predicted_output == [condition5]:
        output_text = "Fault in Line C"
        print(output_text)
    elif predicted_output == [condition6]:
        output_text = "LG fault (Between Phase A and Ground)"
        print(output_text)
    elif predicted_output == [condition7]:
        output_text = "LG fault (Between Phase B and Ground)"
        print(output_text)
    elif predicted_output == [condition8]:
        output_text = "LG fault (Between Phase C and Ground)"
        print(output_text)
    elif predicted_output == [condition9]:
        output_text = "LL fault (Between Phase B and Phase A"
        print(output_text)
    elif predicted_output == [condition10]:
        output_text = "LL fault (Between Phase C and Phase B)"
        print(output_text)
    elif predicted_output == [condition11]:
        output_text = "LL fault (Between Phase C and Phase A)"
        print(output_text)
    elif predicted_output == [condition12]:
        output_text = "LLG Fault (Between Phases A,B and Ground)"
        print(output_text)
    elif predicted_output == [condition13]:
        output_text = "LLG Fault (Between Phases A,C and Ground)"
        print(output_text)
    elif predicted_output == [condition14]:
        output_text = "LLG Fault (Between Phases C,B and Ground)"
        print(output_text)
    elif predicted_output == [condition15]:
        output_text = "LLL Fault(Between all three phases)"
        print(output_text)
    elif predicted_output == [condition16]:
        output_text = "LLLG fault( Three phase symmetrical fault)"
        print(output_text)
    else:
        print("Output does not match any Fault.")

    return render_template('result.html', output_text=output_text)

if __name__ == '__main__':
    app.run(debug=True)

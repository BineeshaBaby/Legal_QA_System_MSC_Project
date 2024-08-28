import gradio as gr 
import requests  

# FastAPI server URL (assuming it's running locally on port 8000)
API_URL = "http://127.0.0.1:8000/ask"  # The URL where the FastAPI backend is running.

def answer_question(Question):
    """
    Sends the user's question to the FastAPI backend and returns the answer along with the updated chat history.

    Args:
        Question (str): The user's legal question.

    Returns:
        tuple: The answer from the backend and the updated chat history.
    """
    # Check for greetings
    if Question.lower() in ["hello", "hi"]:
        greeting = "Hello, it’s great to see you! How can I assist you today?"
        history.append((Question, greeting))
        return greeting, get_history()

    # Send the question to the FastAPI backend
    response = requests.post(API_URL, json={"question": Question})

    if response.status_code == 200:
        answer = response.json().get("answer", "")
        history.append((Question, answer))
        return answer, get_history()
    else:
        return "Error: Could not retrieve answer", get_history()

history = []  # Initializes an empty list to store the chat history.

def get_history():
    """
    Formats the chat history as HTML.

    Returns:
        str: The formatted chat history in HTML.
    """
    return "\n\n".join([f"<p><strong>Q: {q}</strong><br>A: {a}</p>" for q, a in history])
    # Iterates over the history and formats each question-answer pair as an HTML paragraph.

# Custom CSS for styling
custom_css = """
body {
    background-color: grey;
    font-family: Arial, sans-serif;
}
.gradio-container {
    display: flex;
    flex-direction: row;
}
.left-container {
    flex: 3;
    padding: 20px;
}
.right-container {
    flex: 1;
    padding: 20px;
    background-color: #e0e0e0;
    border-radius: 10px;
}
.history-box {
    height: 400px;
    overflow-y: scroll;
    padding: 10px;
    background-color: grey;
}
.footer {
    margin-top: 20px;
    text-align: center;
}
"""
# Defines custom CSS styles to customize the appearance of the Gradio interface.

# Define Gradio interface
iface = gr.Interface(
    fn=answer_question,  # The function to be called when the user submits a question.
    inputs=gr.Textbox(lines=2, placeholder="Enter your Legal Question..."),  # Defines a textbox for inputting the question.
    outputs=[
        gr.Textbox(lines=5, label="Answer"),  # Defines a textbox for displaying the answer.
        gr.HTML(label="History")  # Defines an HTML component for displaying the chat history.
    ],
    title="LEGA: A Perfect Legal Assistant",  # Sets the title of the interface.
    description="Hello! How can I assist you with your legal question today?",  # Sets the description of the interface.
    css=custom_css  # Applies the custom CSS defined above.
)

if __name__ == "__main__":
    iface.launch()  # Launches the Gradio interface when the script is run.

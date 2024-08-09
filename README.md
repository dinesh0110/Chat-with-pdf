# Chat with PDF

## Overview

**Chat with PDF** is a Streamlit application that allows users to upload PDF files, process them to extract and index the text, and interactively query the content using natural language. The application uses advanced AI techniques to provide detailed answers based on the content of the uploaded PDFs. Users can also download their chat history as a `.txt` file.

## Features

- **Upload and Process PDFs**: Upload multiple PDF files and process them to extract and index the text.
- **Interactive Question-Answering**: Ask questions related to the content of the uploaded PDFs and receive detailed answers.
- **Chat History**: View and download the history of your queries and responses as a `.txt` file.

## Installation

To run this application locally, follow these steps:

### 1. Clone the Repository

Clone the repository from Hugging Face Spaces to your local machine:


git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

###2. Install Dependencies

Make sure you have Python installed on your machine. Then, install the required Python libraries:

bash

pip install -r requirements.txt

###3. Set Up Environment Variables

Create a .env file in the root directory of your project with the following content:

makefile

GOOGLE_API_KEY=your_google_api_key

Replace your_google_api_key with your actual Google API key.
Usage
Running the Application

Start the Streamlit application by running the following command:

bash

streamlit run app.py

Using the Application

    Upload PDF Files: Use the file uploader in the sidebar to upload your PDF files.
    Process PDFs: Click the "Submit & Process" button to process the uploaded PDFs.
    Ask Questions: Enter your questions related to the PDF content in the text input field and press Enter.
    Download Chat History: Click the "Download Chat History (TXT)" button to download a .txt file of your chat history.

Contributing

Contributions are welcome! If you have suggestions or improvements, please submit a pull request or open an issue in the repository.
License


For any questions or inquiries, please contact dineshsaliyar@gmail.com
```bash

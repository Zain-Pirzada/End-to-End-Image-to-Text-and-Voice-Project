from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
import requests
import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN= os.getenv("HUGGINGFACEHUB_API_TOKEN")

# image to text

def img2text(url):

    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)

    return text

img2text("photo.jpg")



# Story generation using a Hugging Face model
def generate_story(scenario):
    template = """
    You are a storyteller;
    You can generate a short story based on a sample narrative, the story should be no more than 200 words:
    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    
    # Load the Hugging Face model and tokenizer
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Encode the prompt text and generate the story
    inputs = tokenizer(prompt, return_tensors="pt", max_length=200, truncation=True)
    story = model.generate(**inputs)
    
    return tokenizer.decode(story[0], skip_special_tokens=True)

scenario = img2text("photo.jpg")
story = generate_story(scenario)
print(story)





#text to speech
def text2speech(message):

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"

    headers = {"Authorization": "Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    payloads = {

        "inputs" : message
    }
    
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
	    file.write(response.content)

scenario = img2text("photo.jpg")
story = generate_story(scenario)
text2speech(story)

def main():

    st.set_page_config(page_title="img 2 audio story", page_icon=":p")

    st.header("Turn img into audio story")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
 
    if uploaded_file is not None:

        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image.",
                use_column_width=True)
        scenario=img2text(uploaded_file.name)
        story=generate_story(scenario)
        text2speech(story)
        
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
        st.audio("audio. flac")

if __name__ == '__main__':
   main()
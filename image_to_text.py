import openai
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import urllib.request
import cv2



st.set_page_config(page_title='Image-to-text',page_icon="🤖",layout='wide')
openai.api_key = st.sidebar.text_input("Please Enter your API key",placeholder='Please visit openai website for apikey',type='password')
def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Image Generating'], 
        icons=['house', 'images'], menu_icon="cast", default_index=1)
    selected

if selected == "Home":
    st.header("DALL·E 2")
    st.write("Dall-E is a generative AI technology that enables users to create new images with text to graphics prompts. Functionally, Dall-E is a neural network and is able to generate entirely new images in any number of different styles as specified by the user's prompts The name Dall-E is an homage to the two different core themes of the technology, hinting at the goal of merging art and AI technology. The first part (DALL) is intended to be evocative of famous Spanish surreal artist Salvador Dali, while the second part (E) is related to the fictional Disney robot Wall-E. The combination of the two names reflects the abstract and somewhat surreal illustrative power of the technology, that is automated by a machine. Dall-E was developed by AI vendor OpenAI and first launched in January 2021. The technology uses deep learning models alongside the GPT-3 large language model as a base to understand natural language user prompts and generate new images. Dall-E is an evolution of a concept that OpenAI first began to talk about in June 2020, originally called Image GPT, that was an initial attempt at demonstrating how a neural network can be used to create new high-quality images. With Dall-E, OpenAI was able to extend the initial concept of Image GPT, to enable users to generate new images with a text prompt, much like how GPT-3 can generate new text in response to natural language text prompts.")
    st.header("How does Dall-E Work")
    st.write("Dall-E works by using a number of technologies including natural language processing (NLP), large language models (LLMs) and diffusion processing. Dall-E was built using a subset of the GPT-3 LLM. Instead of the full 175 billion parameters that GPT-3 provides, Dall-E uses only 12 billion parameters in an approach that was designed to be optimized for image generation. Just like the GPT-3 LLM, Dall-E also makes use of a transformer neural network -- also simply referred to as a transformer -- to enable the model to create and understand connections between different concepts.Technically, the approach that enables Dall-E was originally detailed by Open AI researchers as Zero-Shot Text-to-Image Generation and explained in a 20-page research paper released in February 2021. Zero Shot is an AI approach where a model can execute a task, such as generating an entirely new image, by using prior knowledge and related concepts.To help prove that the Dall-E model was able to correctly generate images, Open AI also built the CLIP (Contrastive Language-Image Pre-training) model, which was trained on 400 million labeled images. OpenAI used CLIP to help evaluate Dall-E's output by analyzing which caption is most suitable for a generated image.The first iteration of Dall-E (Dall-E 1) generated images from text using a technology known as a Discreet Variational Auto-Encoder (dVAE) that was somewhat based on research conducted by Alphabet's DeepMind division with the Vector Quantized Variational AutoEncoder.Dall-E 2 improved on the methods used for its first generation to create more high-end and photorealistic images. Among the ways Dall-E 2 works is with the use of a diffusion model that integrates data from the CLIP model to help generate a higher quality image.")
    st.header("Dall-E use cases")
    st.markdown("✅**Creative inspiration**: The technology can be used to help inspire a creative person to create something new. It can also be used as a supplement to an existing creative process.")
    st.markdown("✅**Entertainment**:  Images created by Dall-E could potentially be used in books or games. Dall-E can go beyond the capabilities of traditionally computer-generated imagery (CGI) in that the prompt system is easier to use to create graphics.")
    st.markdown("✅**Education**: Teachers and educators use Dall-E to generate images to explain different concepts.")
    st.markdown("✅**Advertising and marketing**: The ability to create entirely unique and novel images can be useful for advertising and marketing.")
    st.markdown("✅**Product design**: A product designer can use Dall-E to visualize something new, just with the use of text, in an approach that can be significantly faster than using traditional computer-aided design (CAD) technologies.")
    st.markdown("✅**Art**: Dall-E can be used by anyone to create new art to be enjoyed and even displayed.")
    st.markdown("✅**Fashion Design**: As a supplement to existing tools, Dall-E can potentially be useful to help fashion designers come up with new items.")

if selected == "Image Generating":
    if openai.api_key:
        try:
            image_prompt = st.text_input("Enter your prompt to generate image")
            image_limit = st.slider("Enter the number of images you want to generate upto 10",0,10,1)
            image_size = st.text_input("Enter the size of image",placeholder='Eg : 1024x1024')
            response = openai.Image.create(
            prompt=image_prompt,
            n=image_limit,
            size=image_size
            )
            image_url = response['data'][0]['url']
            image_ = url_to_image(image_url)
            st.image(image_,caption=image_prompt)
        except:
            pass
    else:
        st.error("Please enter your api key")


    

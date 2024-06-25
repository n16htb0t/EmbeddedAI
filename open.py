import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
import fitz  # PyMuPDF
import os
import base64
import uuid
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize language models
chain_gpt_35 = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024, openai_api_key=openai_api_key)
chain_gpt_4_vision = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024, openai_api_key=openai_api_key)

# Initialize the vector store and storage layer
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
store = InMemoryStore()
id_key = "doc_id"

# Initialize the retriever
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

# Function to add documents to the retriever
def add_documents_to_retriever(summaries, original_contents):
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, original_contents)))

# Function for text summaries
def summarize_text(text_element):
    prompt = f"Summarize the following text:\n\n{text_element}\n\nSummary:"
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

# Function for table summaries
def summarize_table(table_element):
    prompt = f"Summarize the following table:\n\n{table_element}\n\nSummary:"
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

# Function for image summaries
def summarize_image(encoded_image):
    prompt = [
        AIMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {"type": "text", "text": "Describe the contents of this image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = chain_gpt_4_vision.invoke(prompt)
    return response.content

# Processing extracted elements
def process_elements(text_elements, table_elements, image_elements):
    table_summaries = []
    for i, te in enumerate(table_elements):
        summary = summarize_table(te)
        table_summaries.append(summary)
        print(f"{i + 1}th element of tables processed.")
    
    text_summaries = []
    for i, te in enumerate(text_elements):
        summary = summarize_text(te)
        text_summaries.append(summary)
        print(f"{i + 1}th element of texts processed.")
    
    image_summaries = []
    for i, ie in enumerate(image_elements):
        summary = summarize_image(ie)
        image_summaries.append(summary)
        print(f"{i + 1}th element of images processed.")
    
    # Add summaries to the retriever
    add_documents_to_retriever(text_summaries, text_elements)
    add_documents_to_retriever(table_summaries, table_elements)
    add_documents_to_retriever(image_summaries, image_summaries)

# Function to retrieve information
def retrieve_information(question):
    template = """Answer the question based only on the following context, which can include text, images and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain.invoke(question)

# Function to encode images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_elements_from_pdf(pdf):
    output_path = os.path.join(os.getcwd(), "output_path")
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    text_elements = []
    table_elements = []
    image_elements = []

    # Open the PDF file
    document = fitz.open(stream=pdf.read(), filetype="pdf")
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text("text")
        if text:
            text_elements.append(text)
        
        # Extract images
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            image_path = os.path.join(output_path, f"page_{page_num}_img_{img_index}.png")
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            encoded_image = encode_image(image_path)
            image_elements.append(encoded_image)
    
    return text_elements, table_elements, image_elements

def getting_userinfo():
    fccid = None
    if st.checkbox("Do you have FCCID?"):
        fccid = st.text_input("FCCID")
    return fccid

def get_datasheet():
    datasheet = None
    if st.checkbox("Do you have a datasheet?"):
        datasheet = st.file_uploader("Upload your datasheet here", accept_multiple_files=True)
    return datasheet

def extract_text_from_element(element, separator=" "):
    return separator.join(element.stripped_strings) if element else "N/A"

def scrape_fccid_info(fccid):
    url = f"https://fccid.io/{fccid}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Error fetching FCCID information: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    header_info = soup.find('div', class_='jumbotron')
    if not header_info:
        st.error("No header info found for the given FCCID.")
        return None

    try:
        fcc_id = extract_text_from_element(header_info.find('h1'))
        model_info = extract_text_from_element(header_info.find_all('h4')[1] if len(header_info.find_all('h4')) > 1 else None)

        details = soup.find('div', class_='well')
        if not details:
            st.error("No details section found for the given FCCID.")
            return None
        
        details = details.find_all('li', class_='list-group-item')
        application = extract_text_from_element(details[0]) if len(details) > 0 else "N/A"
        equipment_class = extract_text_from_element(details[1]) if len(details) > 1 else "N/A"
        short_link = details[2].find('a')['href'] if len(details) > 2 and details[2].find('a') else "N/A"
        sources = [a['href'] for a in details[3].find_all('a')] if len(details) > 3 else []
        registered_by = extract_text_from_element(details[4].find('a')) if len(details) > 4 and details[4].find('a') else "N/A"

        operating_freqs_table = soup.find('table', class_='table')
        freq_rows = operating_freqs_table.find_all('tr')[1:] if operating_freqs_table else []

        frequencies = []
        for row in freq_rows:
            cols = row.find_all('td')
            frequency_range = extract_text_from_element(cols[0]) if len(cols) > 0 else "N/A"
            power_output = extract_text_from_element(cols[1]) if len(cols) > 1 else "N/A"
            rule_parts = extract_text_from_element(cols[2]) if len(cols) > 2 else "N/A"
            line_entry = extract_text_from_element(cols[3]) if len(cols) > 3 else "N/A"
            frequencies.append({
                'Frequency Range': frequency_range,
                'Power Output': power_output,
                'Rule Parts': rule_parts,
                'Line Entry': line_entry
            })

        exhibits_div = soup.find('div', id='Exhibits')
        exhibits_table = exhibits_div.find('table') if exhibits_div else None
        exhibit_rows = exhibits_table.find_all('tr')[1:] if exhibits_table else []

        exhibits = []
        for row in exhibit_rows:
            cols = row.find_all('td')
            document = extract_text_from_element(cols[0].find('a')) if len(cols) > 0 and cols[0].find('a') else "N/A"
            doc_type = extract_text_from_element(cols[1]) if len(cols) > 1 else "N/A"
            date_submitted = extract_text_from_element(cols[2]).split('\n')[0] if len(cols) > 2 else "N/A"
            exhibits.append({
                'Document': document,
                'Type': doc_type,
                'Date Submitted': date_submitted
            })

        return {
            'FCC ID': fcc_id,
            'Model Information': model_info,
            'Application': application,
            'Equipment Class': equipment_class,
            'Short Link': short_link,
            'Sources': sources,
            'Registered By': registered_by,
            'Frequencies': frequencies,
            'Exhibits': exhibits
        }
    except Exception as e:
        st.error(f"Error parsing FCCID information: {e}")
        return None

def save_to_txt(fccid_info, datasheet, pcb_image):
    with open('fccid_info.txt', 'w') as f:
        f.write(f"FCC ID: {fccid_info['FCC ID']}\n")
        f.write(f"Model Information: {fccid_info['Model Information']}\n")
        f.write(f"Application: {fccid_info['Application']}\n")
        f.write(f"Equipment Class: {fccid_info['Equipment Class']}\n")
        f.write(f"Short Link: {fccid_info['Short Link']}\n")
        f.write("Sources:\n")
        for source in fccid_info['Sources']:
            f.write(f"{source}\n")
        f.write(f"Registered By: {fccid_info['Registered By']}\n")
        f.write("Operating Frequencies:\n")
        for freq in fccid_info['Frequencies']:
            f.write(f"{freq}\n")
        f.write("Exhibits:\n")
        for exhibit in fccid_info['Exhibits']:
            f.write(f"{exhibit}\n")
        if datasheet:
            f.write("Datasheet Files:\n")
            for d in datasheet:
                f.write(f"{d.name}\n")
        f.write(f"PCB Image: {pcb_image.name if pcb_image else 'Not uploaded'}\n")
    
    return 'fccid_info.txt'

def _sidebar():
    with st.sidebar:
        st.image("https://imgs.search.brave.com/8AnXVYNvZ6gHDj4wxCencYWvzGprUb3ejpP8xToI8h0/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9wbHVz/cG5nLmNvbS9pbWct/cG5nL2NhcnRvb24t/a2lkLXBuZy1iaWct/aW1hZ2UtcG5nLTE2/OTcucG5n", width=250)
        st.write("Please provide the following information")

        fccid = getting_userinfo()
        datasheet = get_datasheet()
        pcb_image = st.file_uploader("Upload your Circuit image here")

        if st.button("Proceed"):
            with st.spinner("Processing your data..."):
                if fccid:
                    fccid_info = scrape_fccid_info(fccid)
                    if fccid_info:
                        st.success("Data processed successfully")
                        st.write(f"FCCID: {fccid_info['FCC ID']}")
                        st.write(f"Model Information: {fccid_info['Model Information']}")
                        st.write(f"Application: {fccid_info['Application']}")
                        st.write(f"Equipment Class: {fccid_info['Equipment Class']}")
                        st.write(f"Short Link: {fccid_info['Short Link']}")
                        st.write(f"Sources: {', '.join(fccid_info['Sources'])}")
                        st.write(f"Registered By: {fccid_info['Registered By']}")
                        st.write("Operating Frequencies:")
                        for freq in fccid_info['Frequencies']:
                            st.write(freq)
                        st.write("Exhibits:")
                        for exhibit in fccid_info['Exhibits']:
                            st.write(exhibit)
                        
                        file_path = save_to_txt(fccid_info, datasheet, pcb_image)
                        st.write(f"All information saved to {file_path}")
                        with open(file_path, "rb") as file:
                            st.download_button(
                                label="Download information as .txt",
                                data=file,
                                file_name="fccid_info.txt",
                                mime="text/plain"
                            )
                    else:
                        st.error("Failed to fetch FCCID information.")
                
                if datasheet and isinstance(datasheet, list):
                    for pdf in datasheet:
                        text_elements, table_elements, image_elements = extract_elements_from_pdf(pdf)
                        process_elements(text_elements, table_elements, image_elements)
                        st.write(f"Extracted {len(text_elements)} text elements, {len(table_elements)} table elements, and {len(image_elements)} images from the PDF.")
                
                st.success("Data processed successfully")
                st.write(f"Datasheet: {datasheet}")
                st.write("PCB Image uploaded:", pcb_image is not None)

def main():
    load_dotenv()
    st.set_page_config(page_title="Embedded AI", page_icon="🧊", layout="wide")
    _sidebar()
    st.write(css, unsafe_allow_html=True)
    st.header("Embedded AI 🧊")
    st.write("Welcome to the Embedded AI app.")
    st.write("This app is designed to help you with your embedded AI projects.")
    
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 2])
    
        with col1:
            st.image("https://imgs.search.brave.com/_nRmlwNCRua5PpHqak_gdnkWJZagwfEw6BVnfst75SM/rs:fit:500:0:0/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy90/aHVtYi9hL2E0L1NF/R19EVkRfNDMwXy1f/UHJpbnRlZF9jaXJj/dWl0X2JvYXJkLTQy/NzYuanBnLzY0MHB4/LVNFR19EVkRfNDMw/Xy1fUHJpbnRlZF9j/aXJjdWl0X2JvYXJk/LTQyNzYuanBn", width=300)
        with col2:
            st.image("https://imgs.search.brave.com/_nRmlwNCRua5PpHqak_gdnkWJZagwfEw6BVnfst75SM/rs:fit:500:0:0/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy90/aHVtYi9hL2E0L1NF/R19EVkRfNDMwXy1f/UHJpbnRlZF9jaXJj/dWl0X2JvYXJkLTQy/NzYuanBnLzY0MHB4/LVNFR19EVkRfNDMw/Xy1fUHJpbnRlZF9j/aXJjdWl0X2JvYXJk/LTQyNzYuanBn", width=300)
        with col3:
            st.image("https://imgs.search.brave.com/_nRmlwNCRua5PpHqak_gdnkWJZagwfEw6BVnfst75SM/rs:fit:500:0:0/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy90/aHVtYi9hL2E0L1NF/R19EVkRfNDMwXy1f/UHJpbnRlZF9jaXJj/dWl0X2JvYXJkLTQy/NzYuanBnLzY0MHB4/LVNFR19EVkRfNDMw/Xy1fUHJpbnRlZF9j/aXJjdWl0X2JvYXJk/LTQyNzYuanBn", width=300)
        st.text_input("\n"+"Ask your questions", key="question")
    
    st.write(bot_template.replace("{{MSG}}", "Hello, how can I help you?"), unsafe_allow_html=True)
    st.write(user_template.replace("{{MSG}}", "I am looking for information on the ESP32 module."), unsafe_allow_html=True)

if __name__ == '__main__':
    main()

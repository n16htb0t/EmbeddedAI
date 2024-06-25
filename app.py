import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template, image_template
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS  # Updated import
from ultralytics import YOLO
from fpdf import FPDF
import base64
import os
import torch

# Setting the environment variable to handle OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

load_dotenv()

# Load YOLOv8 model
model_path = os.path.join('pcbv8', 'best.pt')
model = YOLO(model_path)

# Extracting text and generating placeholder references for images
def get_pdf_text_and_image_references(datasheets, additional_pdfs):
    text = ""
    image_references = []

    if additional_pdfs:
        datasheets += additional_pdfs
    
    for pdf in datasheets:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    
        pdf_document = fitz.open(stream=pdf.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images(full=True)
            for image_index, img in enumerate(image_list):
                image_ref = f"Image {len(image_references) + 1}"
                image_references.append(image_ref)
    return text, image_references

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({'question': user_question})  # Updated method
    st.session_state.chat_history = response['chat_history']

    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            if "Image" in message.content:
                st.write(image_template.replace("{{IMAGE_SRC}}", "https://via.placeholder.com/150"), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def getting_userinfo():
    fccid = None
    if st.checkbox("Do you have FCCID?"):
        fccid = st.text_input("FCCID")
    return fccid

def get_datasheets():
    datasheets = None
    if st.checkbox("Do you have a datasheet?"):
        datasheets = st.file_uploader("Upload your datasheet here", accept_multiple_files=True)
    return datasheets

def get_additional_pdfs():
    additional_pdfs = None
    if st.checkbox("Do you have additional PDFs?"):
        additional_pdfs = st.file_uploader("Upload additional PDFs here", accept_multiple_files=True)
    return additional_pdfs

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

def combine_data(text, fccid_info):
    fcc_text = f"""
    FCC ID: {fccid_info['FCC ID']}
    Model Information: {fccid_info['Model Information']}
    Application: {fccid_info['Application']}
    Equipment Class: {fccid_info['Equipment Class']}
    Short Link: {fccid_info['Short Link']}
    Sources: {', '.join(fccid_info['Sources'])}
    Registered By: {fccid_info['Registered By']}
    Operating Frequencies:
    {"".join([str(freq) for freq in fccid_info['Frequencies']])}
    Exhibits:
    {"".join([str(exhibit) for exhibit in fccid_info['Exhibits']])}
    """
    combined_text = text + "\n" + fcc_text
    return combined_text

def save_to_txt(fccid_info, datasheets, additional_pdfs, pcb_image):
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
        if datasheets:
            f.write("Datasheet Files:\n")
            for d in datasheets:
                f.write(f"{d.name}\n")
        if additional_pdfs:
            f.write("Additional PDF Files:\n")
            for d in additional_pdfs:
                f.write(f"{d.name}\n")
        f.write(f"PCB Image: {pcb_image.name if pcb_image else 'Not uploaded'}\n")
    
    return 'fccid_info.txt'

def create_pdf(fccid_info, datasheets, additional_pdfs, pcb_image_path, shodan_results, component_images, object_detected_image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="FCC ID Information", ln=True, align='C')
    pdf.cell(200, 10, txt=f"FCC ID: {fccid_info['FCC ID']}", ln=True)
    pdf.cell(200, 10, txt=f"Model Information: {fccid_info['Model Information']}", ln=True)
    pdf.cell(200, 10, txt=f"Application: {fccid_info['Application']}", ln=True)
    pdf.cell(200, 10, txt=f"Equipment Class: {fccid_info['Equipment Class']}", ln=True)
    pdf.cell(200, 10, txt=f"Short Link: {fccid_info['Short Link']}", ln=True)
    pdf.cell(200, 10, txt=f"Sources: {', '.join(fccid_info['Sources'])}", ln=True)
    pdf.cell(200, 10, txt=f"Registered By: {fccid_info['Registered By']}", ln=True)

    pdf.cell(200, 10, txt="Operating Frequencies:", ln=True)
    for freq in fccid_info['Frequencies']:
        pdf.cell(200, 10, txt=f"  - {freq}", ln=True)
    
    pdf.cell(200, 10, txt="Exhibits:", ln=True)
    for exhibit in fccid_info['Exhibits']:
        pdf.cell(200, 10, txt=f"  - {exhibit}", ln=True)

    if shodan_results:
        pdf.add_page()
        pdf.cell(200, 10, txt="Shodan Results", ln=True, align='C')
        for result in shodan_results.get('matches', []):
            pdf.cell(200, 10, txt=f"IP: {result.get('ip_str', 'N/A')}", ln=True)
            pdf.cell(200, 10, txt=f"Port: {result.get('port', 'N/A')}", ln=True)
            pdf.cell(200, 10, txt=f"Hostnames: {', '.join(result.get('hostnames', []))}", ln=True)
            pdf.cell(200, 10, txt=f"Location: {result.get('location', {}).get('city', 'N/A')}, {result.get('location', {}).get('country_name', 'N/A')}", ln=True)
            pdf.cell(200, 10, txt=f"ISP: {result.get('isp', 'N/A')}", ln=True)
            pdf.cell(200, 10, txt="---", ln=True)

    if pcb_image_path:
        pdf.add_page()
        pdf.cell(200, 10, txt="PCB Image", ln=True, align='C')
        pdf.image(pcb_image_path, x=10, y=20, w=190)

    if object_detected_image_path:
        pdf.add_page()
        pdf.cell(200, 10, txt="Object Detection Output", ln=True, align='C')
        pdf.image(object_detected_image_path, x=10, y=20, w=190)
    
    for i, (component_image, label) in enumerate(component_images):
        pdf.add_page()
        pdf.cell(200, 10, txt=f"Component {i+1}: {label}", ln=True, align='C')
        pdf.image(component_image, x=10, y=20, w=100, h=100)

    pdf.output("fccid_info.pdf")
    return "fccid_info.pdf"

def shodan_search(device_name, limit=5):
    shodan_api_key = os.getenv("SHODAN_API_KEY")
    url = f"https://api.shodan.io/shodan/host/search?key={shodan_api_key}&query={device_name}&limit={limit}"
    response = requests.get(url)
    return response.json()

def run_yolo_detection(image):
    results = model(image)
    return results

def _sidebar():
    with st.sidebar:
        st.image("https://imgs.search.brave.com/8AnXVYNvZ6gHDj4wxCencYWvzGprUb3ejpP8xToI8h0/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9wbHVz/cG5nLmNvbS9pbWct/cG5nL2NhcnRvb24t/a2lkLXBuZy1iaWct/aW1hZ2UtcG5nLTE2/OTcucG5n", width=250)
        st.write("Please provide the following information")

        fccid = getting_userinfo()
        datasheets = get_datasheets()
        additional_pdfs = get_additional_pdfs()
        pcb_image = st.file_uploader("Upload your Circuit image here")

        shodan_censys_search = False
        device_name = None
        if st.checkbox("Do you want to perform a Shodan search?"):
            shodan_censys_search = True
            device_name = st.text_input("Device Name")

        if st.button("Proceed"):
            with st.spinner("Processing your data..."):

                # Extracting text and image references from PDFs
                raw_text, image_references = get_pdf_text_and_image_references(datasheets, additional_pdfs)

                # Displaying extracted image references
                st.write("Extracted Image References from PDFs:")
                for img_ref in image_references:
                    st.write(img_ref)

                # FCCID Work
                if fccid:
                    fccid_info = scrape_fccid_info(fccid)
                    if fccid_info:
                        raw_text = combine_data(raw_text, fccid_info)
                        st.success("Data processed successfully")
                        st.markdown(f"**FCCID:** {fccid_info['FCC ID']}")
                        st.markdown(f"**Model Information:** {fccid_info['Model Information']}")
                        st.markdown(f"**Application:** {fccid_info['Application']}")
                        st.markdown(f"**Equipment Class:** {fccid_info['Equipment Class']}")
                        st.markdown(f"**Short Link:** {fccid_info['Short Link']}")
                        st.markdown(f"**Sources:** {', '.join(fccid_info['Sources'])}")
                        st.markdown(f"**Registered By:** {fccid_info['Registered By']}")
                        st.markdown("**Operating Frequencies:**")
                        for freq in fccid_info['Frequencies']:
                            st.markdown(f"  - {freq}")
                        st.markdown("**Exhibits:**")
                        for exhibit in fccid_info['Exhibits']:
                            st.markdown(f"  - {exhibit}")
                        
                        file_path = save_to_txt(fccid_info, datasheets, additional_pdfs, pcb_image)
                        st.write(f"All information saved to {file_path}")
                        with open(file_path, "rb") as file:
                            st.download_button(
                                label="Download information as .txt",
                                data=file,
                                file_name="fccid_info.txt",
                                mime="text/plain"
                            )
                        
                        # Perform Shodan search if requested
                        shodan_results = None
                        if shodan_censys_search and device_name:
                            shodan_results = shodan_search(device_name)
                            st.write("Shodan Results:")
                            for result in shodan_results.get('matches', []):
                                st.markdown(f"**IP:** {result.get('ip_str', 'N/A')}")
                                st.markdown(f"**Port:** {result.get('port', 'N/A')}")
                                st.markdown(f"**Hostnames:** {', '.join(result.get('hostnames', []))}")
                                st.markdown(f"**Location:** {result.get('location', {}).get('city', 'N/A')}, {result.get('location', {}).get('country_name', 'N/A')}")
                                st.markdown(f"**ISP:** {result.get('isp', 'N/A')}")
                                st.markdown("---")

                        # Create PDF and display download button
                        pcb_image_path = None
                        object_detected_image_path = None
                        component_images = []

                        if pcb_image:
                            pcb_image_path = "pcb_image.jpg"
                            with open(pcb_image_path, "wb") as img_file:
                                img_file.write(pcb_image.getbuffer())

                            img = Image.open(pcb_image)
                            results = run_yolo_detection(img)

                            img_draw = img.copy()
                            draw = ImageDraw.Draw(img_draw)
                            font = ImageFont.load_default()

                            for result in results[0].boxes:
                                x1, y1, x2, y2 = map(int, result.xyxy[0])
                                label = model.names[int(result.cls)]
                                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                                draw.text((x1, y1), label, fill="red", font=font)
                                cropped_img = img.crop((x1, y1, x2, y2))
                                cropped_img_path = f"cropped_component_{len(component_images) + 1}.jpg"
                                cropped_img.save(cropped_img_path)
                                component_images.append((cropped_img_path, label))

                            object_detected_image_path = "object_detected_image.jpg"
                            img_draw.save(object_detected_image_path)

                            st.session_state.pcb_image_path = pcb_image_path
                            st.session_state.object_detected_image_path = object_detected_image_path
                            st.session_state.component_images = component_images

                        pdf_path = create_pdf(fccid_info, datasheets, additional_pdfs, pcb_image_path, shodan_results, component_images, object_detected_image_path)
                        with open(pdf_path, "rb") as file:
                            st.download_button(
                                label="Download information as PDF",
                                data=file,
                                file_name="fccid_info.pdf",
                                mime="application/pdf"
                            )
                        
                    else:
                        st.error("Failed to fetch FCCID information.")

                # Creating vector store with OpenAI embeddings
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                
                # Creating conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

def main():
    load_dotenv()
    st.set_page_config(page_title="Embedded AI", page_icon="🧊", layout="wide")
    _sidebar()
    st.write(css, unsafe_allow_html=True)
    st.header("Embedded AI 🧊")
    st.write("Welcome to the Embedded AI app.")
    st.write("This app is designed to help you with your embedded AI projects.")

    user_question = st.text_input("\n"+"Ask your questions", key="question")
    if user_question:
        handle_userinput(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "pcb_image_path" in st.session_state and st.session_state.pcb_image_path is not None:
        pcb_image_path = st.session_state.pcb_image_path
        object_detected_image_path = st.session_state.object_detected_image_path
        component_images = st.session_state.component_images

        img = Image.open(pcb_image_path)
        img_draw = Image.open(object_detected_image_path)

        col1, col2 = st.columns([2, 2])

        with col1:
                st.image(img, caption='Uploaded Circuit Image', use_column_width=True)

        with col2:
                st.image(img_draw, caption='Object Detection Output', use_column_width=True)

        st.write(" Components:")
        cols = st.columns(8)
        for i, (cropped_img_path, label) in enumerate(component_images):
            with cols[i % 8]:
                st.image(cropped_img_path, caption=f'Component {i+1}: {label}', width=100)

if __name__ == '__main__':
    main()

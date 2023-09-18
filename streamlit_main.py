import streamlit as st
import os
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from table_transformer import table_structure_export, table_extraction_v2, table_extraction_v1, check_header

from utils import arrange_bbox, arrange_row, image_to_bytes, convert_pdf_to_images, draw_polygon, convert_XYXY_to_polygon
import pandas as pd

from vietocr.tool.config import get_config, list_configs
from vietocr.tool.predictor import Predictor
import base64

st.set_page_config(
    layout="wide"
    )

@st.cache_resource
def load_model():
    model_detection = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    img_preprocess_detection = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")

    model_structure = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    img_preprocess_structure = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")

    config = get_config('fvtr_v2_t@vn')
    model_recognition = Predictor(config, reset_cache=True)  

    return model_detection, img_preprocess_detection, model_structure, img_preprocess_structure, model_recognition

with st.spinner("Loading model..."):
    model_detection, img_preprocess_detection, model_structure, img_preprocess_structure, model_recognition = load_model()


ocr_endpoint = "http://10.10.1.37:10000/ocr"

post_param = dict()
post_param["refine_link"] = True
post_param["return_crops"] = False
post_param["return_fulltext"] = True
post_param["lang"] = 'vi' 
post_param["text_threshold"] = 0.7
post_param["link_threshold"] = 0.4



# Function to list images in the demo_image folder
def list_demo_images():
    demo_image_folder = "demo_images"
    image_path_list = [os.path.join(demo_image_folder, img) for img in os.listdir(demo_image_folder)]
    demo_image_dict = {}
    for image_path in image_path_list:
        demo_image_dict[os.path.basename(image_path)] = Image.open(image_path)
    return demo_image_dict

# Function to create a download link for a DataFrame as a CSV file
def download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
    return href

# Function to display an image and its size
def extract_result(image, table_version = 'Version 2'):
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", use_column_width=True)

    with col2:
        # pil_image = Image.open(image)
        pil_image = image.copy()
        #Extract Table structure
        with st.spinner(f"Performing Table Structure..."):
            table_structure = table_structure_export(pil_image, 
                                                    model_detection = model_detection, 
                                                    model_structure = model_structure, 
                                                    img_preprocess_detection = img_preprocess_detection, 
                                                    img_preprocess_structure = img_preprocess_structure, 
                                                    pad_rate = 0.05,
                                                    tabledetect_threshold = 0.95,
                                                    tablestructure_threshold = 0.8,
                                                    label_names = ['table', 
                                                                    'column', 
                                                                    'row',
                                                                    'column header',
                                                                    'projected row header',
                                                                    'merged cell'])['result']
    
        table_list = []
        if len(table_structure) == 0:
            st.write('Không tìm thấy bảng')
        else:
            for table_num, table_info in enumerate(table_structure):
                table_data = dict()
                table_data['image'] = pil_image

                table_data['table'] = convert_XYXY_to_polygon(table_info['tablebox'])
                table_data['column'] = []
                table_data['row'] = []
                image_copy = pil_image.copy()
                draw = ImageDraw.Draw(image_copy)

                for element in table_info['structure']:
                    if element['label'] == 'column':
                        draw_polygon(draw, convert_XYXY_to_polygon(element['bbox']), line_width = 2, color = (0, 255, 0))
                        table_data['column'].append(convert_XYXY_to_polygon(element['bbox']))
                    if element['label'] == 'row':
                        draw_polygon(draw, convert_XYXY_to_polygon(element['bbox']), line_width = 2, color = (0, 0, 255))
                        table_data['row'].append(convert_XYXY_to_polygon(element['bbox']))

                    if element['label'] == 'column header':
                        draw_polygon(draw, convert_XYXY_to_polygon(element['bbox']), line_width = 2, color = (0, 255, 255))
                        table_data['row'].append(convert_XYXY_to_polygon(element['bbox']))

                #Sort and create range of columns, rows
                table_data['column'] = sorted(table_data['column'], key=lambda polygon: polygon[0][0])
                table_data['row'] = sorted(table_data['row'], key=lambda polygon: polygon[0][1])

                table_data['column_range'] = [[polygon[0][0], polygon[2][0]] for polygon in table_data['column']]
                table_data['row_range'] = [[polygon[0][1], polygon[2][1]] for polygon in table_data['row']]
                table_data = check_header(table_data)  #Remove header if first row is header
                draw_polygon(draw, table_data['table'], line_width = 2, color = (255, 0, 0))

                with col1:
                    st.image(image_copy, caption="Table crop image", use_column_width=True)
                    st.image(pil_image.crop((int(table_data['table'][0][0]), 
                                             int(table_data['table'][0][1]), 
                                             int(table_data['table'][2][0]), 
                                             int(table_data['table'][2][1]))), 
                            caption="Table crop image", use_column_width=True)
                    

                if table_version == 'Version 1':
                    table_matrix_text = table_extraction_v1(table_data, post_param, ocr_endpoint)
                else:
                    table_matrix_text = table_extraction_v2(table_data, pil_image, model_recognition, cell_pad_rate = [0, 0])
                

                table_df = pd.DataFrame(table_matrix_text, columns=None)
                st.write(f"Table {table_num+1}")
                st.table(table_df)
                # Add a download button
                if st.button("Export CSV"):
                    st.markdown(download_link(table_df, "my_data"), unsafe_allow_html=True)
                # st.dataframe(table_df)
                # table_list.append(table_df)
            
# Streamlit app
def main():
    st.title("Table Extraction")

    # Checkbox for using demo images
    use_demo = st.checkbox("Use Sample Images")
    

    if use_demo:
        # List of demo images
        demo_image_dict = list_demo_images()
        
        # Dropdown to select a demo image
        selected_image = st.selectbox("Select a Sample Image", [key for key in demo_image_dict.keys()])
        
        table_version = st.selectbox("Select Table Extraction Version", ['Version 1', 'Version 2'])

        # Display the selected demo image
        extract_result(demo_image_dict[selected_image], table_version = table_version)
    else:
        # File uploader for custom images
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        
        table_version = st.selectbox("Select Table Extraction Version", ['Version 1', 'Version 2'])

        if uploaded_image:
            # Display the uploaded image
            
            extract_result(Image.open(uploaded_image), table_version = table_version)

if __name__ == "__main__":
    main()

from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image
import streamlit as st
from utils import arrange_bbox, arrange_row, image_to_bytes
import requests


def table_structure_export(image_or, 
                           model_detection, 
                           model_structure, 
                           img_preprocess_detection, 
                           img_preprocess_structure, 
                           pad_rate = 0.05,
                           tabledetect_threshold = 0.95,
                           tablestructure_threshold = 0.95,
                           label_names = ['table', 
                                        'table column', 
                                        'table row',
                                        'table column header',
                                        'table projected row header',
                                        'table spanning cell']
                            ):
    # image_or = Image.open(uploaded_image).convert("RGB")
    w_or, h_or = image_or.size

    ## TableDetection
    inputs = img_preprocess_detection(images=image_or, return_tensors="pt")
    outputs = model_detection(**inputs)
    target_sizes = torch.tensor([image_or.size[::-1]])
    result_detection = img_preprocess_detection.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    ## Get tablebox
    result_list = []
    table_count = 0
    for box, label, score in zip(result_detection['boxes'], result_detection['labels'], result_detection['scores']):
        if score < tabledetect_threshold:
            continue
        x1, y1, x2, y2 = box
        result_list.append({'id': table_count,
                            'tablebox':[int(x1), int(y1), int(x2), int(y2)],
                            'structure':[]})
        table_count+=1

    st.write(f"Number of tables found: {table_count}")
    ## Crop Image
    tableimg_list = []
    for i in range(table_count):
        box = result_list[i]['tablebox']
        padding_range_x = min(int(w_or*pad_rate), box[0], w_or - box[2])
        padding_range_y = min(int(h_or*pad_rate), box[1], h_or - box[3])
        x1 = box[0]-padding_range_x
        y1 = box[1]-padding_range_y
        x2 = box[2]+padding_range_x
        y2 = box[3]+padding_range_y
        cropped_image = image_or.crop((x1, y1, x2, y2))
        result_list[i]['crop_table'] = cropped_image
        result_list[i]['pad_x'] = padding_range_x
        result_list[i]['pad_y'] = padding_range_y

    ## Table Structure
    for i in range(table_count):
        
        tableimg = result_list[i]['crop_table']
        inputs = img_preprocess_structure(images=tableimg, return_tensors="pt")
        outputs = model_structure(**inputs)
        target_sizes = torch.tensor([tableimg.size[::-1]])
        result_structure = img_preprocess_structure.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        x_tab = result_list[i]['tablebox'][0] - result_list[i]['pad_x']
        y_tab = result_list[i]['tablebox'][1] - result_list[i]['pad_y']

        for box, label, score in zip(result_structure['boxes'], result_structure['labels'], result_structure['scores']):
            if score < tablestructure_threshold:
                continue
            x1, y1, x2, y2 = box
            
            result_list[i]['structure'].append({'label' : label_names[int(label)],
                                                'bbox'  : [float(x1+x_tab), float(y1+y_tab), float(x2+x_tab), float(y2+y_tab)]})

    return {"labels": label_names,
            "result": result_list}


def ocr_step(image, post_param, ocr_endpoint):
    # Prepare the image data as a dictionary
    files = {"image": image}
    # files = {"image": uploaded_image}
    
    with st.spinner(f"Performing OCR..."):
        # Send the OCR request
        res = requests.post(ocr_endpoint, data=post_param, files=files)
    
    return {
        'texts': res.json()['results'][0]['texts'],
        'boxes': res.json()['results'][0]['boxes']
    }

def check_header(table_data):
    first_row = table_data['row_range'][0]
    second_row = table_data['row_range'][1]

    first_range = first_row[1] - first_row[0]
    second_range = second_row[1] - second_row[0]

    intersect_range = max(0, min(first_row[1], second_row[1]) - max(first_row[0], second_row[0]))
    
    row_iou = intersect_range / (first_range + second_range - intersect_range)
    if row_iou > 0.7:
        table_data['row_range']=table_data['row_range'][1:] 
    return table_data

def table_extraction_v1(table_data, post_param, ocr_endpoint):
    #OCR
    ocr_result = ocr_step(image_to_bytes(table_data['image']), post_param, ocr_endpoint)

    #Arrange boxes
    g = arrange_bbox(ocr_result['boxes'])
    rows = arrange_row(g=g)
    new_text = []
    new_box = []
    for i in range(len(rows)):
        for j in rows[i]:
            new_text.append(ocr_result['texts'][j])
            new_box.append(ocr_result['boxes'][j])
    ocr_result['texts'] = new_text
    ocr_result['boxes'] = new_box

    #Create Table Matrix
    table_matrix_box = [[[] for column in table_data['column']] for row in table_data['row']]
    
    for box_id, box in enumerate(ocr_result['boxes']):
        center_x = sum([point[0] for point in box])/len(box)
        center_y = sum([point[1] for point in box])/len(box)

        col_pos = None
        row_pos = None
        for col_id in range(len(table_data['column_range'])):
            if table_data['column_range'][col_id][0] < center_x and center_x < table_data['column_range'][col_id][1]:
                col_pos = col_id
        for row_id in range(len(table_data['row_range'])):
            if table_data['row_range'][row_id][0] < center_y and center_y < table_data['row_range'][row_id][1]:
                row_pos = row_id

        if  col_pos is None or row_pos is None:
            continue
        else:
            table_matrix_box[row_pos][col_pos].append(box_id)

    table_matrix_text = [["" for column in table_data['column']] for row in table_data['row']]
    for i in range(len(table_matrix_box)):
        for j in range(len(table_matrix_box[i])): 
            table_matrix_text[i][j] = ' '.join([ocr_result['texts'][box_id] for box_id in table_matrix_box[i][j]])

    return table_matrix_text


def table_extraction_v2(table_data, image, model_recognition, cell_pad_rate = [-0.05, 0.1]):
    pil_image = image.copy()
    table_matrix_text = [[[] for column in table_data['column_range']] for row in table_data['row_range']]
    # table_matrix_box = [[[] for column in table_data['column']] for row in table_data['row']]
    with st.spinner("OCR with NewVietOCR..."):

        for i in range(len(table_matrix_text)):
            for j in range(len(table_matrix_text[i])): 
                #x1, y1, x2, y2
                x1 = table_data['column_range'][j][0]
                y1 = table_data['row_range'][i][0]
                x2 = table_data['column_range'][j][1]
                y2 = table_data['row_range'][i][1]

                cell_w = x2-x1 if x2-x1>=0 else x1-x2
                cell_h = y2-y1 if y2-y1>-0 else y1-y2

                #Padding
                x1 = x1 - cell_w*cell_pad_rate[0]
                x2 = x2 + cell_w*cell_pad_rate[0]
                y1 = y1 - cell_h*cell_pad_rate[1]
                y2 = y2 + cell_h*cell_pad_rate[1]

                # st.write(f"{x1}, {y1}, {x2}, {y2}")
                # st.write(f"img size: {image.size}")
                if int(y1) >= int(y2) or int(x1) >= int(x2):
                    table_matrix_text[i][j] = ""
                    continue

                cropped_image = pil_image.crop((int(x1), int(y1), int(x2), int(y2)))
                # st.write(f"{int(x1), int(y1), int(x2), int(y2)}")
                # st.write(type(pil_image))
                # st.image(cropped_image)
                # fulltext = ocr_step(image_to_bytes(cropped_image), fulltext_only = True)['fulltext']
                # st.write(fulltext)
                table_matrix_text[i][j] = model_recognition(cropped_image)[0][0]
                # st.write(table_matrix_text[i][j])

    # st.write(table_data['column_range'])

    # st.write([polygon[0][0] for polygon in table_data['column']])
        
    # for i in range(len(table_data['column'])):
    #         # for j in range(len(table_matrix_text[i])): 
    #         polygon = table_data['column'][i]
    #         cropped_image = pil_image.crop((int(polygon[0][0]), int(polygon[0][1]), int(polygon[2][0]), int(polygon[2][1])))
    #         st.write(int(polygon[0][0]), int(polygon[2][0]))
    #         st.image(cropped_image)

    return table_matrix_text

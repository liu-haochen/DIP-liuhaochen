import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = max(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    # image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    # image = np.array(image_new)
    #放缩
    pre_s = (image.shape[0],image.shape[1],3,)
    new_s = (int(image.shape[0]*scale),int(image.shape[1]*scale),3,)
    transformed_image = np.zeros(new_s)
    inverse_fx = np.linspace(0,pre_s[0]-1,new_s[0]).astype(int)
    inverse_fy = np.linspace(0,pre_s[1]-1,new_s[1]).astype(int)
    for row in range(new_s[0]):
        transformed_image[row,:,:] = image[inverse_fx[row],inverse_fy,:]
    transformed_image =transformed_image.astype(np.uint8)    

    #翻转
    if(flip_horizontal):
        transformed_image = transformed_image[:,::-1,:]
    #旋转+偏移

    theta = -np.pi * rotation/180
    central =  np.array(image_new.shape[0:2])//2
    f_row = np.zeros(image_new.shape[0:2])
    f_col = np.zeros(image_new.shape[0:2])
    f_row = np.repeat(np.linspace(0,f_row.shape[0]-1,f_row.shape[0]),f_row.shape[1]).reshape(f_row.shape)
    f_col = np.tile(np.linspace(0,f_col.shape[1]-1,f_col.shape[1]),f_col.shape[0]).reshape(f_col.shape)
    f_row_n = ( (f_col-central[1])*np.sin(theta) + (f_row-central[0])*np.cos(-theta) +central[0] ).astype(int)
    f_col_n = ( (f_col-central[1])*np.cos(-theta) + (f_row-central[0])*np.sin(-theta) +central[1] ).astype(int)
    # f_row = np.clip(f_row_n,0,image_new.shape[0]-1)
    # f_col = np.clip(f_col_n,0,image_new.shape[1]-1)
    f_row = np.mod(f_row_n-translation_y,image_new.shape[0])
    f_col = np.mod(f_col_n-translation_x,image_new.shape[1])

    # 放到image_new上
    start_x = max((image_new.shape[0] - new_s[0])//2,0)
    end_x = start_x+new_s[0]

    start_y = max((image_new.shape[1] - new_s[1])//2,0)
    end_y = start_y+new_s[1]
    
    image_new[start_x:end_x,start_y:end_y,:] = transformed_image

    #旋转+偏移
    image_new = image_new[f_row,f_col]
    


    return image_new

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()

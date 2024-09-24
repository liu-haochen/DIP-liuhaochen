import cv2
import numpy as np
import gradio as gr
import scipy
import scipy.stats

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    # temp = source_pts
    # source_pts = target_pts
    # target_pts = temp 

    warped_image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping

    #vector1drow = np.repeat(np.linspace(0,warped_image.shape[0]-1,warped_image.shape[0]),warped_image.shape[1] )
    #vector1dcol = np.tile(np.linspace(0,warped_image.shape[1]-1,warped_image.shape[1]),warped_image.shape[0])
    vector2d = np.array([np.repeat(np.linspace(0,warped_image.shape[0]-1,warped_image.shape[0]),warped_image.shape[1] ),
                         np.tile(np.linspace(0,warped_image.shape[1]-1,warped_image.shape[1]),warped_image.shape[0])]).transpose()
    #scipy.stats.norm.pdf()
    #(target_pts - source_pts)
    alpha = np.sqrt(np.sum((target_pts - source_pts)**2)/target_pts.shape[0])/100
    #alpha = np.sqrt(warped_image.shape[0]*warped_image.shape[1])/5
    gm = np.reshape(np.tile(target_pts,target_pts.shape[0]),(target_pts.shape[0],target_pts.shape[0],2))- (np.reshape(np.tile(target_pts,target_pts.shape[0]),(target_pts.shape[0],target_pts.shape[0],2)).transpose(1,0,2))
    gm = np.sqrt( np.sum( gm**2,2))
    gm = scipy.stats.norm.pdf(gm,0,alpha)
    #beta = 1/scipy.stats.norm.pdf(0,0,alpha)
    #gm = gm*beta
    #A=  g   X
    #    X^T 0
    X = np.column_stack((np.ones(target_pts.shape[0]),target_pts))
    Zeros = np.zeros((3,3))
    A = np.row_stack((np.column_stack((gm,X)),np.column_stack((X.transpose(),Zeros)) ))
    b= np.row_stack((source_pts,np.zeros((3,2))))
    s = np.linalg.lstsq(A,b)[0]
    s0 = s[0:target_pts.shape[0],:]
    s1 = s[target_pts.shape[0]+1:,:]
    c = s[target_pts.shape[0],:]
    x_i_minus_x_j = np.tile(vector2d,target_pts.shape[0]).reshape(vector2d.shape[0],target_pts.shape[0],2)-target_pts
    x_i_minus_x_j = np.sqrt( np.sum( x_i_minus_x_j**2,2))
    g_x_i_minus_x_j = scipy.stats.norm.pdf(x_i_minus_x_j,0,alpha)
    resvector2d = g_x_i_minus_x_j.dot(s0)+vector2d.dot(s1)+c
    iresvector2d = resvector2d.astype(int)
    resrow = iresvector2d[:,0].reshape(warped_image.shape[0],warped_image.shape[1])
    rescol = iresvector2d[:,1].reshape(warped_image.shape[0],warped_image.shape[1])
    resrown = np.clip(resrow,0,warped_image.shape[0]-1)
    rescoln = np.clip(rescol,0,warped_image.shape[1]-1)
    new_warped_image = warped_image[resrown,rescoln]
    warped_image[resrown[target_pts[0,0],target_pts[0,1]],rescoln[target_pts[0,0],target_pts[0,1]]]
    new_warped_image[source_pts[0,0],source_pts[0,1]]  
    return new_warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()

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
    source_pts = source_pts[:,::-1]
    target_pts = target_pts[:,::-1]
    warped_image = np.array(image)
    ### FILL: 基于RBF 逆变换 实现 image warping
    #每个像素点坐标
    vector2d = np.array([np.repeat(np.linspace(0,warped_image.shape[0]-1,warped_image.shape[0]),warped_image.shape[1] ),
                         np.tile(np.linspace(0,warped_image.shape[1]-1,warped_image.shape[1]),warped_image.shape[0])]).transpose()
    #调整参数,影响范围大概为所选线段平均长度
    alpha = np.sqrt(np.sum((target_pts - source_pts)**2)/target_pts.shape[0])*2
    #如果希望全局影响，可以考虑下面的参数
    #alpha = np.sqrt(warped_image.shape[0]*warped_image.shape[1])/3
    #gm是g(|x_i-x_j|),g取$mu=0$的高斯函数
    gm = scipy.stats.norm.pdf(np.sqrt(np.sum( (np.reshape(np.tile(target_pts,target_pts.shape[0]),(target_pts.shape[0],target_pts.shape[0],2) ) - target_pts)**2,2)),0,alpha)
    #偏移为 \sum_i a_i g_i(|x_from-x_i|) = x_to-x_from
    b =  source_pts - target_pts
    #gm a = b 系数a_i,计算每个点的偏移来插值
    a = np.linalg.lstsq(gm,b)[0]
    #计算新位置
    gn=scipy.stats.norm.pdf(np.sqrt(np.sum( (np.reshape(np.tile(vector2d,target_pts.shape[0]),(vector2d.shape[0],target_pts.shape[0],2) ) - target_pts)**2,2)),0,alpha)
    #系数乘以顶点位置加顶点位置就是新位置
    res = gn.dot(a) + vector2d
    #获取新图片
    resn = res.astype(int)
    resnr = resn[:,0].reshape(image.shape[0],image.shape[1])
    resnc = resn[:,1].reshape(image.shape[0],image.shape[1])
    blank = np.logical_or( np.logical_or(resnc<0 ,resnr<0) ,np.logical_or(resnc>=warped_image.shape[1],resnr>=warped_image.shape[0]))
    n_warped_image = warped_image[np.clip(resnr,0,warped_image.shape[0]-1),np.clip(resnc,0,warped_image.shape[1]-1) ]
    n_warped_image[blank] = [255,255,255]
    return n_warped_image

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

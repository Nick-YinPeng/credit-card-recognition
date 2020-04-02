import cv2


# 参数一:list结构的，轮廓信息
# 参数二:要使用的方法
# 返回值:处理过后的轮廓和矩形轮廓

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    '''
       
       cv2.boundingRect(c) 
       返回四个值，分别是x，y，w，h；
       x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
       
    '''
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w ， 并将x,y,h，w以元组形式保存，并将10个元组放在一个列表中存储
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # 将cnts ， boundingBoxes进行排序，利用zip将boundingBoxes的x坐标与cnts的y坐标交换，进行排序，之后利用zip(*)恢复排序后的cnts，和boundingBoxes，并将二者返回。
    # 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同 ， 虽然boundingBoxes是四维，但是cnts是二维，所以利用zip这个功能，相当于对boundingBoxes进行了降维处理。
    return cnts, boundingBoxes


# 重置大小，用于比较模板和图像中的数字是否一致
# 插值方法如下：
# INTER_NEAREST:最邻近插值
# INTER_LINEAR:双线性插值,默认情况下使用该方式进行插值.
# INTER_AREA:基于区域像素关系的一种重采样或者插值方式.该方法是图像抽取的首选方法,它可以产生更少的波纹,
# 但是当图像放大时,它的效果与INTER_NEAREST效果相似.
# INTER_CUBIC:4×4邻域双3次插值
# INTER_LANCZOS4:8×8邻域兰索斯插值
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 此函数调整图像大小并保持纵横比为任意高度宽度，为了保持纵横比，（假设传入想要输出参数是width并给出大小）用输出图片width除以w或h，这样就能得到纵横比例
    # 之后height乘以得到的纵横比例，就是保持纵横比例下的height
    dim = None
    (h, w) = image.shape[:2]
    # 如果不需要放大缩小就返回原图像
    if width is None and height is None:
        return image
    # 如果width是None，则resizing   height
    if width is None:
        # 计算高度的比例，构造尺寸
        r = height / float(h)
        dim = (int(w * r), height)
    else:  # 如果height 是None , resizing width
        # 计算宽度的比例，构造尺寸
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

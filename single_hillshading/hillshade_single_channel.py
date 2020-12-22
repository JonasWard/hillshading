import numpy as np
import cv2 as cv
import os

def read_img(img_path):
    print("==========================================")
    print(os.path.dirname(os.path.abspath(__file__) ) )
    print("==========================================")
    print("image path to open : " + img_path)
    img = cv.imread(img_path)

    print(img.shape)

    # cv.imshow("loading",img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    hsvImage = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h,s,v = cv.split(hsvImage)

    h = deviation_single_channel(h, True)
    h.astype(np.uint8)
    pseudocoloring(h, True)

    h,s,v = cv.split(hsvImage)
    h = np.absolute(hillshade_channel(h, 0))

    # h = h.astype(float)
    h = h.astype(np.uint8)
    pseudocoloring(h, True)

    cv.imshow("h channel",h)
    cv.waitKey(0)
    cv.destroyAllWindows()

    b,g,r = cv.split(img)

    b = hillshade_channel_one_side(b)
    r = hillshade_channel_one_side(r)
    g = hillshade_channel_one_side(g)

    img = cv.imread(img_path)

    all_channels = hillshade_channel_both_sides(img, 1)
    all_channels = np.absolute(all_channels)

    print("all_channels shape : " + str(all_channels.shape) )

    all_channels = all_channels.astype(np.uint8)
    pseudocoloring(all_channels, True)

    img = cv.imread(img_path)

    create_hsv(img, True)

def hillshade_channel_both_sides(img, num = 1):
    b,g,r = cv.split(img)

    b = hillshade_channel_one_side(b, num = num)
    r = hillshade_channel_one_side(r, num = num)
    g = hillshade_channel_one_side(g, num = num)

    all_channels_a = b + r + g

    b,g,r = cv.split(img)

    b = hillshade_channel_one_side(b, True, num = num)
    r = hillshade_channel_one_side(r, True, num = num)
    g = hillshade_channel_one_side(g, True, num = num)

    all_channels_b = b + r + g

    x, y, _ = img.shape

    x_num = x - num - 1
    y_num = y - num - 1

    print(x_num, y_num)

    all_channels = np.zeros((x_num, y_num), int)
    all_channels += all_channels_a[0: x_num, 0: y_num]
    all_channels += all_channels_b[0: x_num, 0: y_num]

    all_channels = all_channels.astype( float )
    all_channels -= np.min(all_channels)
    all_channels *= 1.0 / np.max(all_channels) * 254.999
    all_channels -= 255.0
    all_channels = np.abs(all_channels)

    return all_channels

def create_hsv(img, visualize = False):
    hsvImage = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    v,s,h = cv.split(hsvImage)

    output = s * 255.0 / np.max(s)
    output = output.astype(np.uint8)

    if visualize:
        viz = pseudocoloring(output, True)

    return output


def hillshade_channel_one_side(channel, swap = False, num = 3):
    index_range = [i for i in range(1 - num, 0, 1)]

    print(index_range)

    if swap:
        channel = np.swapaxes(channel, 0, 1)
    x, y = channel.shape

    channel_shape = np.zeros((x-1,y), int)

    print("channel.shape : " + str(x) )
    for i in range(x-num):
        if (i >= x - num):
            local_idx = index_range[:i-x-1]
        else:
            local_idx = index_range
        
        channel_shape[i] = channel[i] * (num * 2)

        for idx, val in enumerate(local_idx):
            channel_shape[i] += val * channel[i + idx + 1]

    if swap:
        channel_shape = np.swapaxes(channel_shape, 0, 1)
    print(channel_shape.shape)

    return channel_shape


def hillshade_channel_with_value(channel, num = 2):
    index_range = [i for i in range(1, num + 1, 1)]
    index_range = index_range + [i for i in range(1 - num, 0, 1)]

    print(index_range)

    channel = np.swapaxes(channel, 0, 1)
    x, y = channel.shape

    channel_shape = np.zeros((x-1,y), int)

    for i in range(x):
        if (i < x):
            local_idx = index_range[i:]
            root = num - i
        elif (i > x - num):
            local_idx = index_range[:i-x]
            root = i
        else:
            local_idx = index_range
            root = i
        
        for idx, val in enumerate(local_idx):
            channel_shape[i] += val * channel[idx + root]

    channel_shape = np.swapaxes(channel_shape, 0, 1)
    print(channel_shape.shape)

    return channel_shape

def pseudocoloring(img, visualize = False):
    img = cv.applyColorMap(img, cv.COLORMAP_JET)

    if visualize:
        cv.imshow("pseudocoloring",img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return img

def deviation_single_channel(deviation_calc, visualize = False):
    deviation_calc.astype(np.uint64)
    x, y = deviation_calc.shape
    output = np.zeros((x,y), np.double)
    
    mean_value = np.nanmean(deviation_calc)
    std_value = np.std(deviation_calc)

    output += (deviation_calc * deviation_calc - mean_value) / std_value * 7.5 + 127
    # output = np.abs(output)

    print("mean value : " + str( mean_value ) )
    print("std value : " + str( std_value ) )
    print("ouput : " + str(output) )

    output = output.astype(np.uint8)

    if visualize:
        cv.imshow("deviation_single_channel",output)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return output


def hillshade_channel(channel, swap = False, lapl = 0):
    # cv.imshow("a channel",channel)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    if swap:
        channel = np.swapaxes(channel, 0, 1)
    x, y = channel.shape

    print(channel.shape)

    channel_shape = np.zeros((x-1,y), int)

    for i in range(x-1):
        channel_shape[i] = channel[i] - channel[i+1]

    if swap:
        channel_shape = np.swapaxes(channel_shape, 0, 1)
    
    print(channel_shape.shape)

    print("mean value : " + str(np.nanmean(channel_shape) ) )
    print("median value : " + str(np.nanmedian(channel_shape) ) )
    print("std value : " + str(np.std(channel_shape) ) )

    return channel_shape

def laplacian_smooth(channel, lapl = 5):
    x,y =channel.shape
    new_channel = np.zeros( (x,y), dtype = np.int16)


if __name__ == "__main__":
    img_path = "./example.jpg"
    read_img(img_path)
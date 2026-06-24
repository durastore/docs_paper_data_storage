def crop_image(img , crop_w, crop_h):

    # Determine original image dimensions
    rows, cols, channels = img.shape
    x_cent = cols / 2
    y_cent = rows / 2
    w_x1 = int(x_cent - (crop_w / 2))
    w_x2 = int(x_cent + (crop_w / 2))
    w_y1 = int(y_cent - (crop_h / 2))
    w_y2 = int(y_cent + (crop_h / 2))

    crop_coor = [w_y1/SPR, w_y2/SPR, w_x1/SPR, w_x2/SPR]
    crop_img = img[w_y1:w_y2, w_x1:w_x2]

    return crop_img, crop_coor

#Detection of contours
def cont_det(img, pos_l, dil_it_n, dil_kernel_size, spot_size):
    #
    import cv2
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    from shapely import geometry
    import numpy as np

    #
    site_r = int(spot_size/2)
    img_gen = np.zeros((len(img), len(img)), dtype='uint8')
    for pos in pos_l:
        coor = (int(pos[1]), int(pos[0]))
        cv2.circle(img_gen, coor, radius=site_r, color=255, thickness=-1)

    # Binary thresholding after gaussian filtering
    ret3, th3 = cv2.threshold(img_gen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#10,255

    # Dilation to join nearby points
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_kernel_size, dil_kernel_size))
    dilation = cv2.dilate(th3, kernel, iterations=dil_it_n)#13/15

    # Erosion for decreasing countour size
    erosion = cv2.erode(dilation, kernel, iterations=round(dil_it_n/2))

    # Detection of contours in dilated image
    contour_points = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    Area = []
    for cnt in contour_points:
        Area.append(cv2.contourArea(cnt))

    max_cnt_ind = Area.index(max(Area))

    str_cnt = contour_points[max_cnt_ind]
    rect = cv2.minAreaRect(str_cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    polygon =Polygon(box)


    xs = list(polygon.exterior.coords.xy[0])
    ys = list(polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = geometry.Point(min(xs), min(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * 0.1
    polygon_resized = polygon.buffer(-shrink_distance)

    point_l = []
    for pos in pos_l:
        point = Point(pos[1], pos[0])
        if polygon_resized.contains(point) == True:
            point_l.append(pos)
    vert_coor = list(zip(*polygon_resized.exterior.coords.xy))

    return vert_coor, point_l

#Function to generate a cropped image from the extracted structure localization coordinates
def loc_to_img_crop(directory, loc_data_df_l, data_index, window_size, SPR, int_norm, gauss_size, index, channel, channel_rel_int):    #import packages
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import os, errno
    import cv2

    new_dir = directory + '\Images_for_prot_det'
    try:
        os.makedirs(new_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    #Make empty image
    img = np.zeros((window_size, window_size), dtype='uint8')
    img_ref = np.zeros((window_size, window_size), dtype='uint8')
    
    #Create merge coordinate list
    loc_data_coord_l = [loc_data_df['loc_coor'].values.tolist() for loc_data_df in loc_data_df_l]
    x_coord_merge_l = [item[0] for loc_data_coord in loc_data_coord_l for item in loc_data_coord]
    y_coord_merge_l = [item[1] for loc_data_coord in loc_data_coord_l for item in loc_data_coord]
    x_cent = np.mean(x_coord_merge_l)
    y_cent = np.mean(y_coord_merge_l)
    orig_x = x_cent - (window_size / (2 * SPR))
    orig_y = y_cent - (window_size / (2 * SPR))


    #Modify coordinates for centering
    loc_array_n = [[x[0] - orig_x, x[1] - orig_y] for x in loc_data_df_l[data_index]['loc_coor'].values.tolist()]
    loc_ref_array_n = [[x[0] - orig_x, x[1] - orig_y] for loc_data_coord in loc_data_coord_l for x in loc_data_coord]
    
    # Fill up image with localizations
    for loc in loc_array_n:
        # Make empty image for detection
        x = int((loc[0])*SPR)
        y = int((loc[1])*SPR)
        if 0 < x < window_size and 0 < y < window_size:
            img[x][y] = img[x][y] + 1.0*float(channel_rel_int)

    # Fill up image with localizations
    for loc in loc_ref_array_n:
        # Make empty image for detection
        x = int((loc[0]) * SPR)
        y = int((loc[1]) * SPR)
        if 0 < x < window_size and 0 < y < window_size:

            img_ref[x][y] = img_ref[x][y] + 1.0

    # Normalize intensity
    thr_min, thr_max = int_norm
    img = 255.0 * ((img - thr_min) / (thr_max - thr_min))
    img[img < 0.0] = 0.0

    #Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img, (int(gauss_size), int(gauss_size)), 0)
    cv2.imwrite(new_dir + '\Render-crop-GS-' + str(index) + '-'+channel + '.png', img_blur)
    cv2.destroyAllWindows()

    image = cv2.imread(new_dir + '\Render-crop-GS-' + str(index) + '-'+channel + '.png', 0)
    im_color = cv2.applyColorMap(image, cv2.COLORMAP_HOT)

    cv2.imwrite(new_dir + '\Render-crop-CLR-'+str(index)+ '-'+channel +'.png',im_color)
    cv2.destroyAllWindows()

    img_gs_file = new_dir + '\Render-crop-GS-' + str(index) + '-' + channel + '.png'
    img_color_file = new_dir + '\Render-crop-CLR-'+str(index)+ '-'+channel +'.png'

    return img_gs_file, img_blur, new_dir, loc_array_n

#Function for creating aligned merged image of channels
def img_merge_bc(gs_img_file_l, ch_color_l, channel_name_l, export_folder, index):
    # import packages used
    import cv2
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import copy

    # Create multichannel image
    # Transform grayscale images into color images
    gs_img_l = []
    rgb_img_l =[]
    rgb_color_l = []
    for i in range(len(gs_img_file_l)):
        ch_color = mpl.colors.to_rgb(ch_color_l[i])[::-1]
        rgb_color_l.append([x*255 for x in ch_color])

        gs_img = cv2.imread(gs_img_file_l[i], 0)
        window_size = len(gs_img)
        gs_img = cv2.cvtColor(gs_img, cv2.COLOR_GRAY2RGB)
        gs_img = gs_img.astype('float64')
        gs_img_l.append(gs_img)
        rgb_img = [[[j[0]*ch_color[0],j[1]*ch_color[1],j[2]*ch_color[2]] for j in i ] for i in gs_img]
        rgb_img_a = np.array(rgb_img)
        rgb_img_l.append(rgb_img_a)


    # Merge images
    img_merge1 = rgb_img_l[0]
    img_merge2 =  copy.deepcopy(rgb_img_l[0])
    cv2.putText(img_merge1, channel_name_l[0],(2,8),cv2.FONT_HERSHEY_SIMPLEX,0.3,rgb_color_l[0],1, cv2.LINE_AA)

    for i in range(len(rgb_img_l) - 1):
        img_merge1 = img_merge1 + rgb_img_l[i + 1]
        img_merge2 = img_merge2 + rgb_img_l[i + 1]
        cv2.putText(img_merge1, channel_name_l[i+1],(2, 8+(i+1)*8),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.3,
                   rgb_color_l[i+1],
                   1, cv2.LINE_AA)
    img_merge1[img_merge1 > 255] = 255
    img_merge2[img_merge2 > 255] = 255
    cv2.imwrite(export_folder + '\Render-crop-merge-' + str(index) + '.png', img_merge1)
    cv2.destroyAllWindows()
    img_clr_file = export_folder + '\Render-crop-merge-' + str(index) + '.png'

    gs_img_merge = gs_img_l[0]
    for i in range(len(gs_img_l)-1):
        gs_img_merge += gs_img_l[i+1]
    gs_img_merge[gs_img_merge > 255] = 255

    cv2.imwrite(export_folder + '\Render-crop-merge-gs-' + str(index) + '.png', gs_img_merge)
    cv2.destroyAllWindows()
    img_gs_file = export_folder + '\Render-crop-merge-gs-' + str(index) + '.png'

    return img_merge2, img_clr_file, gs_img_merge, img_gs_file, rgb_img_l

#Function to calculate the square of dist of two points
def Dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

#Function to calcualte the dist of two points
def Dist(p1, p2):
    import numpy as np
    return np.sqrt(Dist2(p1, p2))

#Function to merge points clustered in a certain distance to a mean point
def fuse2(points, d):
    ret_exp = []
    d2 = d * d
    len_l = [len(points)]
    check = 0
    while check == 0:
        ret = []
        n = len(points)
        taken = [False] * n
        for i in range(n):
            if not taken[i]:
                count = 1
                point = [points[i][0], points[i][1]]
                taken[i] = True
                for j in range(i+1, n):
                    if Dist2(points[i], points[j]) < d2:
                        point[0] += points[j][0]
                        point[1] += points[j][1]
                        count+=1
                        taken[j] = True
                point[0] /= count
                point[1] /= count
                ret.append((point[0], point[1]))
        ret_exp = ret
        points = ret
        len_l.append(len(points))
        if len_l[-1] == len_l[-2]:
            check = 1

    return ret_exp

#
def get_cmap(n, name):
    import matplotlib.pyplot as plt
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

#Function to calculate centroid coordinate from a set of x,y coordinates
def centroid_calc(data):
    centroid_l = []
    for i in range(len(data)):
        x = sum(h[0] for h in data[i] if h[0] is not None) / len(data[i])
        y = sum(h[1] for h in data[i] if h[1] is not None) / len(data[i])
        cent_1 = [x, y]
        centroid_l.append(cent_1)
    return centroid_l

#Function to annotate points (Find external points -> name one as first -> annotate rest by using point by point dist
def point_annot_bc(coor):
    # import packages used
    import numpy as np
    from copy import deepcopy

    #Check number of points in structure
    n_coor = len(coor)

    #Generate distance matrix
    dist_mat = np.zeros((n_coor, n_coor))
    for i in range(len(dist_mat)):
        for j in range(len(dist_mat[i])):
            dist_mat[i][j] = dist_mat[i][j] + Dist(coor[i], coor[j])

    #Annotate points based on site distances and point number

    # find indices and coordinates of terminal positions from extracting the index of the maximum distances in the dist matrix
    i = np.unravel_index(np.argmax(dist_mat, axis=None), dist_mat.shape)

    # find index and coordinates of p1 as the terminal position with lower p1-p3 dist
    term_pos_d_l = [list(dist_mat[ind]) for ind in i]
    for d_l in term_pos_d_l:
        d_l.sort()
        d_l.remove(0)
    p1_p3_d_l = [d[1] for d in term_pos_d_l]
    p1_index = i[p1_p3_d_l.index(min(p1_p3_d_l))]

    # find index and coordinate of other positions
    ord_coor = []
    p1_d_l = list(dist_mat[p1_index])
    p1_d_l_sorted = deepcopy(p1_d_l)
    p1_d_l_sorted.sort()
    
    for d in p1_d_l_sorted:
        p_index = p1_d_l.index(d)
        p_coor = coor[p_index]
        ord_coor.append(p_coor)
    
    #Creating a list of ordered coordinates and point to point distances for exporting
    dist_l = []
    for i in range(len(ord_coor)-1):
        p1 = ord_coor[i]
        p2 = ord_coor[i+1]
        dist_l.append(Dist(p1,p2))
    
    dist_tot = Dist(ord_coor[0],ord_coor[-1])

    return ord_coor, dist_l, dist_tot

#Function for determining position areas using watershed algorithm
def pos_area_det(spot_coor, img):
    #Import packages used
    import numpy as np
    from skimage.segmentation import watershed
    from scipy import ndimage as ndi
    import cv2
    import matplotlib.pyplot as plt
    import imutils

    #Create gs image with positions
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img_gs, 30, 255, cv2.THRESH_BINARY)#20-255

    distance = ndi.distance_transform_edt(th2)
    plt.imshow(-distance, cmap=plt.cm.gray)
    local_maxima = np.zeros_like(th2, dtype=bool)
    spot_coor = np.array([[int(x[0]),int(x[1])] for x in spot_coor])
    local_maxima[tuple(np.array(spot_coor).T)] = True
    markers = ndi.label(local_maxima)[0]

    # Now, mark the region of unknown with zero
    labels = watershed(-distance, markers, mask=th2)

    #
    pos_cnt_l = []
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(img_gs.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        pos_cnt_l.append(c)

    return labels, pos_cnt_l, distance

#Function to cluster all loc in d to clust center
def loc_assign(loc_l, spot_pos_l, pos_thr_l, offset_thr):
    import numpy as np

    x_min, x_max, y_min, y_max = pos_thr_l

    assign_loc_l = []
    for i in range(len(spot_pos_l)+1):
        assign_loc_l.append([])

    for loc in loc_l:
        d_l = []
        for spot_pos in spot_pos_l:
            pos_coor = [spot_pos[0]+y_min, spot_pos[1]+x_min]
            d = Dist(pos_coor, loc)
            d_l.append(d)
        pos_ind = d_l.index(min(d_l))+1

        if y_min<loc[0]<y_max and x_min<loc[1]<x_max:
            assign_loc_l[pos_ind].append(loc)
        else:
            assign_loc_l[0].append(loc)

    filt_assing_loc_l = []
    filt_assing_loc_l.append(assign_loc_l[0])

    for i in range(len(assign_loc_l)-1):
        pos = assign_loc_l[i+1]
        spot_pos = [spot_pos_l[i][0]+y_min, spot_pos_l[i][1]+x_min]
        if len(pos)>0:
            mean_pos = [np.mean([x[0] for x in pos]),np.mean([x[1] for x in pos])]
            if Dist(spot_pos,mean_pos)>offset_thr:
                filt_assing_loc_l[0] = filt_assing_loc_l[0] + assign_loc_l[i+1]
                filt_assing_loc_l.append([])
            else:
                filt_assing_loc_l.append(assign_loc_l[i+1])

    return assign_loc_l

#Function to check if distance passed
def dist_thr(dist_l, d_mean, d_std):
    #
    check = 0
    for dist in dist_l:
        if dist < d_mean - d_std or d_mean + d_std < dist:
            check = check + 1

    return check

#Function to calculate vector and normalized vector between two points
def vect_calc(p1, p2):
    #import packages used
    import numpy as np

    #Calculate vector coordinates
    x1, y1 = p1
    x2, y2 = p2
    v_x = x2-x1
    v_y = y2-y1

    #Calcualte vector length
    l = np.sqrt(v_x**2 + v_y**2)

    #Calculate normalized vector coordinates
    v = [v_x, v_y]
    v_n = [v_x/l, v_y/l]
    return(v, v_n)

#Function to calculate linearity of set of points (STD of normalized vectors between adjacent points)
def lin_check(ann_spot_coor):
    # import packages used
    import numpy as np

    #List to store point-to-point vectors and normal vectors
    v_l = []
    v_n_l = []

    # Calculate vectors from ref point to every point
    for i in range(len(ann_spot_coor) - 1):
        p1 = ann_spot_coor[i]
        p2 = ann_spot_coor[i+1]
        v, v_n = vect_calc(p1=p1, p2=p2)
        v_l.append(v)
        v_n_l.append(v_n)

    # Cal
    v_n_x, v_n_y = map(list, zip(*v_n_l))
    score = (np.std(v_n_x) + np.std(v_n_y))/2

    return score, v_l, v_n_l

#Function for calculating angle of rotation using coordinates of terminal points of the structure
def rot_angle_calc(p1,p2):
    #import packages used
    import numpy as np

    #Define reference point (p3) to define axis to rotate to
    x1,y1 = p1
    x2,y2 = p2
    p3 = [x1, y2]

    #Calculate distances between point for calculating angle between the initial axis and final
    d1 = Dist(p1, p2)
    d2 = Dist(p1, p3)
    angle = np.degrees(np.arccos(d2/d1))

    #Determine rotation angle based on relative position of original points
    if x2 > x1:
        if y2 > y1:
            rot_angle = angle * -1

        else:
            rot_angle = (180 - angle)*-1

    else:
        if y2 > y1:
            rot_angle = angle

        else:
            rot_angle = 180-angle

    return rot_angle

#Function for translating and rotating localization coordinates using the calculated transformation values
def rot_trans_loc(loc_list, SPR, trans_coor, origin, angle):
    import numpy as np

    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    mod_loc_list = []

    # Rotate
    angle = np.deg2rad(angle)

    x_trans, y_trans = trans_coor

    for loc in loc_list:
        loc_x, loc_y = loc

        #Translate

        loc_x_tr = loc_x*SPR + x_trans
        loc_y_tr = loc_y*SPR + y_trans

        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d([loc_x_tr, loc_y_tr])
        p_n = np.squeeze((R @ (p.T - o.T) + o.T).T)

        mod_loc_list.append(p_n)

    return mod_loc_list

#Function to
def pos_det (SPR, export,export_file_format, processing_dir, img_gs, img_color_file, int_thr, d_thr_site_nm, dil_it_n, dil_kernel_size, index, site_n_limit, site_spread,  pix_nm_SPR):

    # Open png files in dir
    from skimage.feature import peak_local_max
    import cv2
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    #Open color image
    img_clr = mpimg.imread(img_color_file)

    #Calculate size of image
    dim = len(img_clr)
    c_im= [dim/2,dim/2]

    #Make binary image of gs image
    ret, thresh3 = cv2.threshold(img_gs, int_thr, 255, cv2.THRESH_TOZERO)

    #Find local maxima in image
    xy = peak_local_max(thresh3)

    #Merge local maxima using a dist thr
    dist_thr_pix = d_thr_site_nm/pix_nm_SPR
    new_peak_list = fuse2(xy, dist_thr_pix)

    #Remove maxima far from center of ROI
    if len(new_peak_list)>0:
        box, filt_coor = cont_det(img=img_gs, pos_l=new_peak_list, dil_it_n = dil_it_n, dil_kernel_size=dil_kernel_size, spot_size=site_spread)
        spot_n = len(filt_coor)
    else:
        spot_n = 0

    #Cluster peaks into structures using a second distance metric
    if spot_n != site_n_limit:
        ord_coor = 'NaN'
        dist_tot_nm = 'NaN'
        dist_l_nm = 'NaN'
        lin_score = 'NaN'
        
        if export == True:
            from matplotlib import gridspec

            fig = plt.figure(figsize=(28, 4))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], height_ratios=[1])

            ax0 = plt.subplot(gs[0])
            ax0.imshow(img_clr)
            ax0.axis("off")

            ax1 = plt.subplot(gs[1])
            ax1.imshow(img_clr)
            ax1.axis("off")
            for point in xy:
                x = point[1]
                y = point[0]
                ax1.plot(x, y, 'bx')

            ax2 = plt.subplot(gs[2])
            ax2.imshow(img_clr)
            ax2.axis("off")

            for point in new_peak_list:
                x = point[1]
                y = point[0]
                ax2.plot(x, y, 'gx')

            if spot_n > 0:
                box.append(box[0])  # repeat the first point to create a 'closed loop'
                xs, ys = zip(*box)
                ax2.plot(xs, ys, linestyle='-.', color='cyan')

            plt.savefig(processing_dir + '\Protein_detection_plot-' + str(index) +'.'+ export_file_format)
            plt.close()

    else:

        ord_coor, dist_l, dist_tot = point_annot_bc(coor=filt_coor)
        lin_score, v_l, v_n_l = lin_check(ann_spot_coor=ord_coor)
        dist_tot_nm = dist_tot * pix_nm_SPR
        dist_l_nm = [d*pix_nm_SPR for d in dist_l]

        #Normalize coordinates:
        for i in range(len(filt_coor)):
            filt_coor[i]=list(filt_coor[i])
            filt_coor[i][0] = filt_coor[i][0]/SPR
            filt_coor[i][1] = filt_coor[i][1]/SPR

        # Normalize coordinates:
        for i in range(len(ord_coor)):
            ord_coor[i] = list(ord_coor[i])
            ord_coor[i][0] = ord_coor[i][0] / SPR
            ord_coor[i][1] = ord_coor[i][1] / SPR

        if export==True:

            from matplotlib import gridspec

            fig = plt.figure(figsize=(50, 4))
            gs = gridspec.GridSpec(1, 8, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1], height_ratios=[1])

            ax0 = plt.subplot(gs[0])
            ax0.imshow(img_clr)
            ax0.axis("off")

            ax1 = plt.subplot(gs[1])
            ax1.imshow(img_clr)
            ax1.axis("off")
            for point in xy:
                x = point[1]
                y = point[0]
                ax1.plot(x, y, 'bx')

            ax2 = plt.subplot(gs[2])
            ax2.imshow(img_clr)
            ax2.axis("off")

            for point in new_peak_list:
                x = point[1]
                y = point[0]
                ax2.plot(x, y, 'gx')

            ax3 = plt.subplot(gs[3])
            ax3.imshow(img_clr)
            ax3.axis("off")

            for i in range(len(new_peak_list)):
                point = new_peak_list[i]
                x = point[1]
                y = point[0]

                ax3.plot(x, y, 'x', color='g')

            for i in range(len(filt_coor)):
                point = filt_coor[i]
                x = point[1]*SPR
                y = point[0]*SPR

                ax3.plot(x, y, 'x', color='cyan')

            box.append(box[0])  # repeat the first point to create a 'closed loop'
            xs, ys = zip(*box)
            ax3.plot(xs, ys, linestyle='-.', color='cyan')

            ax4 = plt.subplot(gs[4])
            ax4.imshow(img_clr)
            ax4.axis("off")

            for i in range(len(filt_coor)):
                point = filt_coor[i]
                x = point[1]*SPR
                y = point[0]*SPR

                ax4.plot(x, y, 'x', color='cyan')

            ax5 = plt.subplot(gs[5])
            ax5.imshow(img_clr)
            ax5.axis("off")

            for i in range(len(ord_coor)):
                point = ord_coor[i]
                x = point[1] * SPR
                y = point[0] * SPR
                ax5.plot(x, y, 'x', color='cyan')
                ax5.annotate(i + 1, (x, y), color='magenta',
                             xytext=(2, 2), textcoords='offset points', )

            ax5 = plt.subplot(gs[5])
            ax5.imshow(img_clr)
            ax5.axis("off")

            for i in range(len(ord_coor)):
                point = ord_coor[i]
                x = point[1] * SPR
                y = point[0] * SPR
                ax5.plot(x, y, 'x', color='cyan')
                ax5.annotate(i + 1, (x, y), color='magenta',
                             xytext=(2, 2), textcoords='offset points', )

            ax6 = plt.subplot(gs[6])
            ax6.imshow(img_clr, zorder=0)
            ax6.axis("off")
            
            v_name_l = []
            for i in range(len(ord_coor) - 1):
                pos = i + 1
                pos_next = pos + 1
                v_name = 'p' + str(pos) + '-p' + str(pos_next)
                v_name_l.append(v_name)
            colors = plt.cm.get_cmap('tab10', len(v_name_l))
            colors_l = colors.colors

            for i in range(len(ord_coor)):
                point = ord_coor[i]
                x = point[1] * SPR
                y = point[0] * SPR
                ax6.plot(x, y, 'x', color='cyan', zorder=5)

            for i in range(len(v_l)):
                v_x, v_y = v_l[i]
                px, py = ord_coor[i]
                x = [px * SPR, px * SPR + v_x]
                y = [py * SPR, py * SPR + v_y]
                ax6.annotate("", xy=(y[1], x[1]), xytext=(y[0], x[0]),
                             arrowprops=dict(arrowstyle="->", lw=3, color=colors_l[i], shrinkA=0, shrinkB=0), zorder=10)

            ax7 = plt.subplot(gs[7])

            for i in range(len(v_n_l)):
                vec_norm = v_n_l[i]
                v_x, v_y = vec_norm
                x = [0, v_x]
                y = [0, v_y]
                ax7.arrow(y[0], x[0], y[1], x[1], label=v_name_l[i], ec=colors(i), fc=colors(i), width=0.01,
                          head_width=0.03, alpha=0.8)

            ax7.legend()
            ax7.set_xlim(-1.0, 1.0)
            ax7.set_ylim(-1.0, 1.0)
            ax7.invert_yaxis()
            
            plt.savefig(processing_dir + '\Position_detection_plot-' + str(index) + '.'+export_file_format)
            plt.close()

    return ord_coor, dist_tot_nm, dist_l_nm, lin_score

#Function for checking if site-to-site distance is within the permitted range
def pos_dist_check(ref_dist_l, pos_dist_l, std):
    
    pos_chk_l = []
    for ref_dist, dist in zip(ref_dist_l, pos_dist_l):
        if ref_dist*(1.0-std) < dist < ref_dist*(1.0+std):
            pos_chk_l.append(True)
        else:
            pos_chk_l.append(False)

    if False in pos_chk_l:
        return False
    else:
        return True

#Function to recalculate the position coordinates from the final sorted coordinates
def pos_update(sorted_loc_l):
    import numpy as np

    merged_loc_l = [[] for x in sorted_loc_l[0][1:]]
    for ch in sorted_loc_l:
        for i in range(len(ch)-1):
            merged_loc_l[i]= merged_loc_l[i] + ch[i+1]

    new_pos_list = [[np.mean([b[0] for b in a]),np.mean([b[1] for b in a])] for a in merged_loc_l]

    return new_pos_list, merged_loc_l

#
def align_and_quant(img_clr, ch_img_l, rgb_img_l, ord_coor, loc_array_l, channel_color_l, channel_name_l, SPR, pix_gap, crop_w_w, crop_w_h, site_spread, offset_thr, site_n,
                    eps_dbscan, export,export_file_format, str_index, processing_dir):

    # import packages used
    import numpy as np
    import copy
    from scipy.signal import find_peaks
    import cv2
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from collections import Counter
    from scipy.spatial import ConvexHull

    #
    str_index = str_index
    channel_n = len(loc_array_l)

    #
    results = []

    #Determine original image dimensions
    img_clr = img_clr.astype(np.uint16)

    rows, cols, channels = img_clr.shape
    x_cent = cols / 2
    y_cent = rows / 2
    p_cent = [x_cent, y_cent]

    #Fetch coordinates for first spot
    y1, x1 = [item * SPR for item in ord_coor[0]]

    #Calculate centroid for structure
    str_cent_x = np.mean([ord_coor[0][1]*SPR, ord_coor[-1][1]*SPR])
    str_cent_y = np.mean([ord_coor[0][0]*SPR, ord_coor[-1][0]*SPR])

    #Calculate vector for translation
    x_v = x_cent - str_cent_x
    y_v = y_cent - str_cent_y
    trans_coor = [y_v, x_v]
    results.append(trans_coor) #0

    #Translate image
    M_t = np.float32([
        [1, 0, x_v],
        [0, 1, y_v]])
    shifted = cv2.warpAffine(img_clr, M_t, (rows, cols))
    shifted_ch_img_l = []
    shifted_rgb_img_l = []
    for ch_img in ch_img_l:
        ch_img_shifted = cv2.warpAffine(ch_img, M_t, (rows, cols))
        shifted_ch_img_l.append(ch_img_shifted)
    for rgb_img in rgb_img_l:
        rgb_img_shifted = cv2.warpAffine(rgb_img, M_t, (rows, cols))
        shifted_rgb_img_l.append(rgb_img_shifted)
        
    results.append(shifted) #1

    x1_n = x1 + x_v
    y1_n = y1 + y_v
    p1_n = [x1_n, y1_n]

    p1_c_d = Dist(p1_n, p_cent)

    #Rotate image
    y2, x2 = [item * SPR for item in ord_coor[-1]]

    #Calculate translated coordinates
    x2_n = x2 + x_v
    y2_n = y2 + y_v
    p2_n = [x2_n, y2_n]

    #Calculate angle of rotation
    rot_angle = rot_angle_calc(p1=p1_n, p2=p2_n)
    rot_list = [rot_angle, [x_cent, y_cent]]
    results.append(rot_list) #2

    #Rotate image
    M_r = cv2.getRotationMatrix2D((x_cent, y_cent), rot_angle, 1)
    rotated = cv2.warpAffine(shifted, M_r, (cols, rows))

    ch_img_rot_l = []
    for ch_img_shifted in shifted_ch_img_l:
        ch_img_rot = cv2.warpAffine(ch_img_shifted, M_r, (cols, rows))
        ch_img_rot_l.append(ch_img_rot)

    rgb_img_rot_l = []
    for rgb_img_shifted in shifted_rgb_img_l:
        rgb_img_rot = cv2.warpAffine(rgb_img_shifted, M_r, (cols, rows))
        rgb_img_rot_l.append(rgb_img_rot)
    results.append(rotated) #3

    #Crop image
    spot_r = site_spread / 2
    crop_w = crop_w_w
    crop_h = crop_w_h

    w_x1 = int((cols / 2) - (crop_w / 2))
    w_x2 = int((cols / 2) + (crop_w / 2))
    w_y1 = int(y_cent - p1_c_d - pix_gap - spot_r)
    w_y2 = int(w_y1 + crop_h)
    results.append([w_x1, w_x2, w_y1, w_y2]) #4

    crop = rotated[w_y1:w_y2, w_x1:w_x2]

    ch_img_crop_l = []
    for ch_img_rot in ch_img_rot_l:
        ch_img_crop = ch_img_rot[w_y1:w_y2, w_x1:w_x2]
        ch_img_crop_l.append(ch_img_crop)

    rgb_img_crop_l = []
    for rgb_img_rot in rgb_img_rot_l:
        rgb_img_crop = rgb_img_rot[w_y1:w_y2, w_x1:w_x2]
        rgb_img_crop_l.append(rgb_img_crop)
        
    results.append(crop) #5

    #Transform position coordinates using calculated offset and rotation angle
    x_min, x_max, y_min, y_max = [w_x1, w_x2, w_y1, w_y2]
    rot_angle, rot_cent = rot_list
    mod_ord_coor = rot_trans_loc(loc_list=ord_coor, trans_coor=trans_coor, SPR=SPR, origin=rot_cent, angle=rot_angle)
    mod_ord_coor = [[x[0]-y_min, x[1]-x_min] for x in mod_ord_coor]
    results.append(mod_ord_coor) #6

    #Transform localization coordinates using calculated offset and rotation angle
    mod_loc_l = []
    for loc_array in loc_array_l:
        mod_loc_list = rot_trans_loc(loc_list=loc_array, trans_coor=trans_coor, SPR=SPR, origin=rot_cent,
                                    angle=rot_angle)
        mod_loc_l.append(mod_loc_list)

    #Sort localizations into sites

    #Create list to store localizations: [ch1, ch2 ...], ch1: [noise, pos1, pos2 ...]
    loc_coord_sorted = []
    for j in range(channel_n):
        channel_l = []
        for k in range(site_n+1):
            channel_l.append([])
        loc_coord_sorted.append(channel_l)

    #Sort modified localizations into storing list
    for j in range(len(mod_loc_l)):
        channel_loc_l = mod_loc_l[j]
        sorted_channel_l = loc_coord_sorted[j]
        sorted_loc_l = loc_assign(loc_l=channel_loc_l, spot_pos_l=mod_ord_coor, pos_thr_l=[w_x1, w_x2, w_y1, w_y2], offset_thr=offset_thr)
        for h in range(len(sorted_loc_l)):
            loc_l = sorted_loc_l[h]
            for loc in loc_l:
                x, y = loc
                sorted_channel_l[h].append([x / SPR, y / SPR])

    #Clustering of localization for removal of noise
    loc_coord_sorted_clust = []
    loc_coord_export = []
    loc_removed = []

    #
    for j in range(len(loc_coord_sorted)):
        loc_coord_sorted_clust.append([])
        loc_coord_export.append([])
        loc_removed.append([0,0])
        channel = loc_coord_sorted[j]

        for k in range(len(channel)):
            loc_coord_sorted_clust[j].append([])
            loc_coord_export[j].append([])

            pos = channel[k]

            #Clustering localizations
            if len(pos) > 0:
                clustering = DBSCAN(eps=eps_dbscan, min_samples=3).fit(pos)
                labels = clustering.labels_
            else:
                labels = []

            #Determine label for main cluster (highest number of loc)
            alphas_l = []
            labels_filt = [x for x in labels if x != -1]

            if len(labels_filt)>0:
                id = Counter(labels_filt).most_common(1)[0][0]

                for cluster_id in labels:
                    if cluster_id == id:
                        alphas_l.append(1.0)
                    else:
                        alphas_l.append(0.2)

            else:
                id='NaN'
                for cluster_id in labels:
                    alphas_l.append(0.2)

            if k==0:
                loc_coord_sorted_clust[j][k]= loc_coord_sorted_clust[j][k] + [1.0 for x in loc_coord_sorted[j][k]]
                loc_coord_export[j][k] = loc_coord_export[j][k] + pos
                for h in range(len(pos)):
                    loc_removed[j][1] += 1
                    loc_removed[j][0] += 1

            else:
                loc_coord_sorted_clust[j][k] = loc_coord_sorted_clust[j][k] + alphas_l
                for h in range(len(pos)):
                    if labels[h] == id:
                        loc_coord_export[j][k].append(pos[h])
                        loc_removed[j][0] += 1

                    else:
                        loc_coord_export[j][0].append(pos[h])
                        loc_removed[j][0] += 1
                        loc_removed[j][1] += 1

    new_pos_l = [[(coor[1]*SPR)-x_min, (coor[0]*SPR)-y_min] for coor in pos_update(sorted_loc_l=loc_coord_export)[0]]
    merged_loc_l = pos_update(sorted_loc_l=loc_coord_export)[1]

    results.append(new_pos_l)  # 7

    pos_hull_l = []
    for i in range(len(merged_loc_l)):
        pos = merged_loc_l[i]
        pos_hull_l.append([])
        loc_l = np.array([[x[1]*SPR,x[0]*SPR] for x in pos])

        if len(loc_l)==0:
            pos_c_point = mod_ord_coor[i]
            loc_l = np.array([[pos_c_point[0]-spot_r,pos_c_point[1]-spot_r],[pos_c_point[0]+spot_r,pos_c_point[1]-spot_r],[pos_c_point[0]-spot_r,pos_c_point[1]+spot_r],[pos_c_point[0]+spot_r,pos_c_point[1]+spot_r]])

        pos_hull = ConvexHull(loc_l)

        for simplex in pos_hull.simplices:
            pos_hull_l[-1].append([loc_l[list(simplex), 0], loc_l[list(simplex), 1]])

    results.append(pos_hull_l)  # 8
    results.append(loc_coord_sorted) #9
    results.append(loc_coord_export)  #10

    shifted = results[1]
    rotated = results[3]
    crop_w_list = results[4]
    x_min, x_max, y_min, y_max = crop_w_list
    crop = results[5]
    mod_ord_coor = results[6]
    mod_ord_coor_new = results[7]
    pos_hull_l = results[8]
    loc_coord_sorted = results[9]
    loc_coord_export = results[10]

    if export==True:
        from matplotlib import gridspec
        from matplotlib.patches import Rectangle

        fig = plt.figure(dpi=300, tight_layout=True)
        fig.set_size_inches(20, 9, forward=True)

        gs = fig.add_gridspec(2, 28, width_ratios=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], height_ratios=[1,1])


        ax0 = plt.subplot(gs[0, 0:6])
        ax0.imshow(cv2.cvtColor(img_clr, cv2.COLOR_BGR2RGB))
        ax0.axis("off")

        ax2 = plt.subplot(gs[0, 7:13])
        ax2.imshow(cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB))
        ax2.axis("off")

        ax4 = plt.subplot(gs[0, 14:20])
        ax4.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        ax4.axis("off")

        ax5 = plt.subplot(gs[0, 23:25])
        ax5.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        for point in mod_ord_coor:
            x = point[1]
            y = point[0]
            ax5.plot(x, y, 'x', color='red')
        ax5.axis("off")

        ax6 = plt.subplot(gs[1,8:10])
        ax6.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        for point in mod_ord_coor_new:
            x = point[0]
            y = point[1]
            ax6.plot(x, y, 'x', color='green')

        for j in range(len(mod_ord_coor_new)):
            ax6.annotate('Pos'+str(j+1),(mod_ord_coor_new[j][0] + spot_r, mod_ord_coor_new[j][1] + spot_r), color='green',
                             xytext=(2, 2), textcoords='offset points')

        for hull in pos_hull_l:
            for simplex in hull:
                ax6.plot(simplex[0]-x_min, simplex[1]-y_min, 'green')
        ax6.axis("off")

        ax7 = plt.subplot(gs[1, 13 :15])
        for j in range(len(channel_name_l)):

            site_names_l = ['Noise',channel_name_l[j]]
            site_colors_l = ['lightgrey',channel_color_l[j]]

            loc_l = loc_coord_sorted[j]
            exp_loc_l = loc_coord_export[j]

            for k in range(len(loc_l)):
                alphas = loc_coord_sorted_clust[j][k]
                loc_pos = loc_l[k]
                x_l = [item[0] * SPR for item in loc_pos]
                y_l = [item[1] * SPR for item in loc_pos]

                exp_loc_pos = exp_loc_l[k]
                x_l_exp = [item[0] * SPR for item in exp_loc_pos]
                y_l_exp = [item[1] * SPR for item in exp_loc_pos]
                if k == 0:
                    color = site_colors_l[0]
                    name = site_names_l[0]

                    if j == 0:

                        if len(alphas) > 0:
                            ax7.scatter(y_l, x_l, color=color, s=5, label=name, marker='o', alpha=alphas, zorder=5)

                        else:
                            ax7.scatter(y_l, x_l, color=color, s=5, label=name, marker='o', zorder=5)

                    else:
                        if len(alphas) > 0:
                            ax7.scatter(y_l, x_l, color=color, s=5, marker='o', alpha=alphas, zorder=5)

                        else:
                            ax7.scatter(y_l, x_l, color=color, s=5, marker='o', zorder=5)

                else:
                    color = site_colors_l[1]
                    name = site_names_l[1]
                    if len(alphas) > 0:
                        ax7.scatter(y_l, x_l, color=color, s=5,  marker='o', alpha=alphas, zorder=5)

                    else:
                        ax7.scatter(y_l, x_l, color=color, s=5, marker='o', zorder=5)

                if k==1:
                    ax7.scatter(y_l_exp, x_l_exp, color=color, s=1, label=name, marker='o', zorder=10)
                else:
                    ax7.scatter(y_l_exp, x_l_exp, color=color, s=1, marker='o', zorder=10)

            ax7.set_xlim([x_min, x_max])
            ax7.set_ylim([y_min, y_max])

            for hull in pos_hull_l:
                for simplex in hull:

                    ax7.plot(simplex[0],simplex[1], 'green')

            ax7.xaxis.set_ticks([])
            ax7.yaxis.set_ticks([])
            ax7.invert_yaxis()
            ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.savefig(processing_dir + '/Protein_counting_plot-' + str(str_index) +'.'+ export_file_format)

        plt.close()
    
    return crop, loc_coord_export, loc_removed, ch_img_crop_l, rgb_img_crop_l

def PAINT_func_proc_main_gui():
    # import modules used
    import PySimpleGUI as sg
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import csv
    import copy

    left_col = [
        [sg.Frame('Data information:',
                  [[sg.Text('Hdf5 file with grouped localization:'),
                    sg.In(size=(50, 1), enable_events=True, key='grouped_loc_file'),
                    sg.FileBrowse()],
                   [sg.Text('Csv file with segmentation info:'),
                    sg.In(size=(50, 1), enable_events=True, key='img_seg_csv'),
                    sg.FileBrowse()]
                   ])],
        [sg.Frame('Data acquisition parameters:',
                  [[sg.Text('Magnification:', size=(36, 1)), sg.InputText('150', key='mag', size=(8, 1))],
                   [sg.Text('Pixel_size (in nm):', size=(36, 1)), sg.InputText('86.7', key='pix_size', size=(8, 1))]
                  ])],
        [sg.Frame('Data rendering parameters:',
                  [[sg.Text('Sub-sampling:', size=(36, 1)), sg.InputText('30', key='SPR', size=(8, 1))],
                   [sg.Text('Crop window size:', size=(36, 1)), sg.InputText('90', key='Window_size', size=(8, 1))],
                   [sg.Text('Normalization intensity:', size=(36, 1)),
                    sg.InputText('0.2', key='min_int_thr', size=(8, 1)),
                    sg.InputText('2.0', key='max_int_thr', size=(8, 1))],
                   [sg.Text('Gaussian blur kernel size:', size=(36, 1)),
                    sg.InputText('9', key='gauss_kernel', size=(8, 1))],
                  ])],
        [sg.Frame('Data processing parameters:',
                   [[sg.Text('Intensity threshold for spot detection (0-255):', size=(36, 1)),
                    sg.InputText('25', key='int_thr', size=(8, 1))],
                   [sg.Text('Number of positions:', size=(36, 1)),
                    sg.InputText('5',key='site_n', size=(8, 1))],
                    [sg.Text('Spot size (in nm):', size=(36, 1)),
                     sg.InputText('30',key='spot_size_nm', size=(8, 1))],
                   [sg.Text('Distance between first site and sites (in nm values separated by ","):', size=(36, 1)),
                    sg.InputText('35,70,134,169',key='site_d_l', size=(8, 1))],
                    [sg.Text('Allowed site-to-site distance deviation (0.0-1.0):', size=(36, 1)),
                     sg.InputText('0.4', key='site_d_std', size=(8, 1))],
                   [sg.Text('Parameters for  contour detection in ROIs:', size=(36, 1)),
                    sg.InputText('20',key='dil_it_n', size=(8, 1)),
                    sg.InputText('3',key='dil_kernel_size', size=(8, 1))],
                   [sg.Text('Distance threshold for peak clustering (in nm):', size=(36, 1)),
                    sg.InputText('11',key='d_thr_site_nm', size=(8, 1))],
                    [sg.Text('Linearity score threshold (0.0-1.0):', size=(36, 1)),
                     sg.InputText('0.3', key='lin_score_thr', size=(8, 1))],
                    [sg.Text('EPS value for DBSCAN clustering (in pix):', size=(36, 1)),
                     sg.InputText('0.14', key='eps_dbscan', size=(8, 1))],
                    [sg.Text('Maximum allowed offset of channel site center from predicted (in nm):', size=(36, 1)),
                     sg.InputText('10.0', key='offset_thr', size=(8, 1))],
                   [sg.Text('Window size for cropped images (in nm):', size=(36, 1)),
                    sg.InputText('30',key='crop_win_w', size=(8, 1)), sg.InputText('90',key='crop_win_h', size=(8, 1))],
                    [sg.Text('Window offset from first position (in nm):', size=(36, 1)),
                     sg.InputText('5', key='gap_nm', size=(8, 1))]
                   ])],
        [sg.Text('Folder to save data to:'),
         sg.In(size=(50, 1), enable_events=True, key='export_folder'),
         sg.FolderBrowse()],
        [sg.Button('Load data'), sg.Button('Cancel')]
    ]
    right_col = [
        [sg.Frame('Output:', [[sg.Multiline("", size=(50, 30), key='OUTPUT')]])]
    ]

    layout = [[sg.Column(left_col, element_justification='c'),
               sg.Column(right_col, element_justification='c', vertical_alignment="top")]]
    window_single_site_occ = sg.Window('Single site occupancy calculation', layout, resizable=True)

    #
    while True:
        event, values = window_single_site_occ.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window_single_site_occ.close()
            break

        if event == 'Load data':
            window_single_site_occ['OUTPUT'].update(value='Processing data' + '\n', append=True)
            window_single_site_occ.refresh()

            try:
                mag = int(values['mag'])
                pix_nm = float(values['pix_size'])

                SPR = int(values['SPR'])
                win_size = int(values['Window_size'])
                thr_min = float(values['min_int_thr'])
                thr_max = float(values['max_int_thr'])
                gauss_kernel = int(values['gauss_kernel'])

                int_thr = int(values['int_thr'])
                spot_size_nm = float(values['spot_size_nm'])
                site_n = int(values['site_n'])
                site_d_l = [int(x) for x in values['site_d_l'].split(',')]
                site_d_std = float(values['site_d_std'])
                d_thr_site_nm = int(values['d_thr_site_nm'])
                lin_score_thr = float(values['lin_score_thr'])
                dil_it_n = int(values['dil_it_n'])
                dil_kernel_size = int(values['dil_kernel_size'])
                offset_thr = float(values['offset_thr'])
                eps_dbscan = float(values['eps_dbscan'])
                crop_w_w = int(values['crop_win_w'])
                crop_w_h = int(values['crop_win_h'])
                gap_nm = int(values['gap_nm'])


                processing_param_l = [mag, pix_nm, SPR,win_size,thr_min, thr_max,gauss_kernel,int_thr, spot_size_nm,site_n,
                                      site_d_l,site_d_std, d_thr_site_nm, lin_score_thr, dil_it_n,dil_kernel_size, offset_thr,eps_dbscan,crop_w_w,crop_w_h,gap_nm]

                window_single_site_occ['OUTPUT'].update(value='Processing parameters loaded' + '\n', append=True)
                window_single_site_occ.refresh()

                segmented_data_hdf = values['grouped_loc_file']
                if len(segmented_data_hdf) == 0:
                    sg.Popup('Segmented data h5 file not provided.')
                    break

                window_single_site_occ['OUTPUT'].update(value='Hdf5 file loaded:' + '\n', append=True)
                window_single_site_occ['OUTPUT'].update(value=segmented_data_hdf + '\n', append=True)
                window_single_site_occ.refresh()

                img_size_csv_file = values['img_seg_csv']
                if len(img_size_csv_file) == 0:
                    sg.Popup('Segmentation csv file not provided.')
                    break

                else:
                    window_single_site_occ['OUTPUT'].update(value='Csv file loaded:' + '\n', append=True)
                    window_single_site_occ['OUTPUT'].update(value=segmented_data_hdf + '\n', append=True)
                    window_single_site_occ.refresh()


                    with open(img_size_csv_file, newline='') as csvfile:
                        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
                        seg_data_info = [row for row in spamreader]
                    channel_n = int([ row for row in seg_data_info][0][1])
                    channel_name_l = [x[1] for x in [row for row in seg_data_info][2:2+channel_n]]
                    data_types_l = [ row for row in seg_data_info][-1]

                exp_folder = values['export_folder']
                if len(exp_folder) == 0:
                    sg.Popup('Export folder not specified.')
                    break

                # Exporting processing parameters to csv file
                with open(exp_folder + '/Processing_parameters.csv', 'w', newline='') as csvfile:

                    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(['Data information'])
                    spamwriter.writerow(['Data path', segmented_data_hdf])
                    spamwriter.writerow(['Number of channels', channel_n])
                    spamwriter.writerow(['Data types']+data_types_l)
                    spamwriter.writerow(['Channel names']+channel_name_l)
                    spamwriter.writerow(['Magnification', mag])
                    spamwriter.writerow(['Pixel size [nm]', pix_nm])

                    spamwriter.writerow(['Data rendering parameters'])
                    spamwriter.writerow(['Sub sampling rate', SPR])
                    spamwriter.writerow(['Render window size [pix]', win_size])
                    spamwriter.writerow(['Intensity normalization values', thr_min, thr_max])
                    spamwriter.writerow(['Gaussian blur kernel size [pix]', gauss_kernel])

                    spamwriter.writerow(['Data processing parameters'])
                    spamwriter.writerow(['Spot size [nm]', spot_size_nm])
                    spamwriter.writerow(['Intensity threshold for peak detection', int_thr])
                    spamwriter.writerow(['Number of sites', site_n])
                    spamwriter.writerow(['Site distances']+site_d_l)
                    spamwriter.writerow(['Site-to-site distance spread threshold', site_d_std])
                    spamwriter.writerow(['Distance threshold for spot clustering', d_thr_site_nm])
                    spamwriter.writerow(['Linearity score threshold', lin_score_thr])
                    spamwriter.writerow(['Parameters used for origami contour detection', dil_it_n,dil_kernel_size])
                    spamwriter.writerow(['EPS value used for DBSCAN clustering [pix]', eps_dbscan])
                    spamwriter.writerow(['Channel cluster center to site position threshold [nm]', offset_thr])
                    spamwriter.writerow(['Crop window size', crop_w_w, crop_w_h])
                    spamwriter.writerow(['Window offset [nm]', gap_nm])

                window_single_site_occ['OUTPUT'].update(value='Processing parameters exported' + '\n', append=True)
                window_single_site_occ.refresh()

                PAINT_func_proc_gui(channel_n=channel_n, channel_name_l=channel_name_l, segment_file_name=segmented_data_hdf, processing_param=processing_param_l, exp_folder=exp_folder)

            except ValueError:
                sg.Popup('Could not proceed with rendering. Check if all parameters are provided correctly.')

def PAINT_func_proc_gui(channel_n, channel_name_l, segment_file_name, processing_param, exp_folder):
    # import modules used
    import pandas as pd
    import PySimpleGUI as sg
    import csv
    from tqdm import tqdm
    from sklearn.cluster import DBSCAN
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    #Loading processing parameters from main gui
    mag, pix_nm, SPR, win_size,thr_min, thr_max, gauss_size, int_thr, spot_size_nm, site_n, site_d_l, site_d_std, d_thr_site_nm, lin_score_thr, dil_it_n, dil_kernel_size, offset_thr, eps_dbscan, crop_w_w, crop_w_h, gap_nm = processing_param

    #Convert input values to pixel values
    pix_nm_SPR = float(pix_nm/float(SPR))
    dist_tot_design = max(site_d_l)
    dist_min_design = min(site_d_l)
    ref_dist_l = [site_d_l[0]]
    for i in range(len(site_d_l)-1):
        ref_dist_l.append(site_d_l[i+1]-site_d_l[i])
    pix_gap = gap_nm/pix_nm_SPR
    spot_size_pix = int(spot_size_nm/pix_nm_SPR)
    site_d_l_pix = [int(x/pix_nm_SPR) for x in site_d_l]
    offset_thr_pix = offset_thr/pix_nm_SPR

    #
    left_col = [
        [sg.Frame('Processing parameters',
                  [
        [sg.Text('Number of probes to process:', size=(36, 1)),
         sg.InputText('', key='probe_n_limit', size=(8, 1))]
                   ])]
    ]

    for i in range(channel_n):
        channel_name = channel_name_l[i]
        left_col.append([sg.Text('Color for rendering channel ' +channel_name +':', size=(36, 1)),
         sg.Combo(['cyan',
                   'blue',
                   'green',
                   'yellow',
                   'orange',
                   'magenta',
                   'purple',
                   'black'], default_value='cyan', enable_events=True,
                  key='drop-'+str(i))])
        left_col.append([sg.Text('Relative intensity of channel ' +str(channel_name)+':', size=(36, 1)),
         sg.InputText('1', key='rel_int_ch-'+str(i), size=(8, 1))])

    left_col.append([sg.Frame('Processing parameters',
                  [[sg.Checkbox('Export images for position detection', size=(36, 1),
                     key='chk_exp_pos_det')],
        [sg.Checkbox('Export images for alignment', size=(36, 1),
                     key='chk_exp_align')],
        [sg.Text('File format to use for export:', size=(36, 1)),
         sg.Combo(['jpg', 'png', 'eps', 'pdf'], default_value='jpg', enable_events=True,
                  key='drop1')],
        [sg.Button('Process data'), sg.Button('Cancel')]])])

    right_col = [
        [sg.Frame('Output:', [[sg.Multiline("", size=(50, 30), key='OUTPUT')]])]
    ]

    layout = [[sg.Column(left_col, element_justification='c'),
               sg.Column(right_col, element_justification='c', vertical_alignment="top")]]
    window_data_proc = sg.Window('PAINT data processing', layout, resizable=True)



    while True:
        event, values = window_data_proc.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window_data_proc.close()
            break

        if event == 'Process data':
            window_data_proc['OUTPUT'].update(
                value='')
            window_data_proc.refresh()


            export_pos_det = values['chk_exp_pos_det']

            export_align = values['chk_exp_align']

            exp_file_format = values['drop1']

            channel_color_l = []
            channel_rel_int_l = []
            for i in range(channel_n):
                channel_color_l.append(values['drop-'+str(i)])
                channel_rel_int_l.append(int(values['rel_int_ch-'+str(i)]))

            ch_name_l = []
            ch_color_l = []
            for i in range(channel_n):

                ch_color_l.append(channel_color_l[i])
                ch_name_l.append(channel_name_l[i])

            # Export channel information
            window_data_proc['OUTPUT'].update(value='Exporting channel information')
            window_data_proc.refresh()
            with open(exp_folder + '/Processing_channel_info.csv', 'w', newline='') as csvfile:

                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(['Alignment information'])
                spamwriter.writerow(['Number of designed positions', site_n])
                spamwriter.writerow(['Distance between positions'] + site_d_l)

                spamwriter.writerow(['PAINT channel information'])
                spamwriter.writerow(['Channel names'] + ch_name_l)
                spamwriter.writerow(['Channel colors'] + ch_color_l)

            if values['probe_n_limit'].isdigit() == False:
                probe_proc_limit = False
            else:
                probe_proc_limit = True
                probe_n_limit = int(values['probe_n_limit'])

            # Import loc file and transform it into a np array
            data_df = pd.read_hdf(segment_file_name, key='seg_locs')

            #
            indeces = []
            for index in set(data_df['Str_index'].values):
                indeces.append(index)

            #Lists for storing data
            str_index_l = []
            str_pos_l = []
            str_clr_img_l = []
            str_spot_pos_l = []
            str_crop_img_l = []
            str_ch_img_l = []
            str_rgb_img_l = []
            str_loc_l = []
            str_removed_loc_l = []

            if probe_proc_limit == False:
                it_n = len(indeces)
            else:
                it_n = probe_n_limit

            window_data_proc['OUTPUT'].update(value='Starting processing of data',
                            append=True)
            window_data_proc.refresh()

            from tqdm import tqdm

            with tqdm(total=it_n) as pbar:
                for i in range(it_n):

                    pbar.update(1)
                    window_data_proc['OUTPUT'].update(value=pbar)
                    window_data_proc.refresh()
                    ind = indeces[i]
                    str_index_l.append(ind)
                    df_ind = data_df.loc[(ind == data_df['Str_index'])]
                    str_pos = df_ind['Str_pos'].values.tolist()[0]
                    str_pos_l.append(str_pos)

                    #
                    data_hdf_l = []
                    for i in range(channel_n):
                        channel_name = channel_name_l[i]
                        data_hdf_l.append(df_ind[channel_name].values[0])


                    #
                    img_gs_file_l = []
                    loc_array_l = []
                    ch_img_l = []
                    for i in range(channel_n):
                        img_gs_file, img_gs, processing_dir, loc_array_n = loc_to_img_crop(directory=exp_folder,
                                                                                                    loc_data_df_l=data_hdf_l,
                                                                                                    data_index = i,
                                                                                                    window_size=win_size,
                                                                                                    SPR=SPR,
                                                                                                    int_norm=[thr_min, thr_max],
                                                                                                    gauss_size=gauss_size,
                                                                                                    index=ind,channel=channel_name_l[i],
                                                                                                    channel_rel_int = channel_rel_int_l[i])
                        img_gs_file_l.append(img_gs_file)
                        loc_array_l.append(loc_array_n)
                        ch_img_l.append(img_gs)

                    img_multi, img_multi_file, img_multi_gs, img_gs_file, rgb_img_l = img_merge_bc(gs_img_file_l=img_gs_file_l, ch_color_l=channel_color_l, channel_name_l=channel_name_l, export_folder=processing_dir, index=ind)
                    str_clr_img_l.append(img_multi)

                    new_peak_list, dist_tot, dist_l_nm, lin_score = pos_det(SPR=SPR,
                                                      export=export_pos_det,
                                                      export_file_format=exp_file_format,
                                                      processing_dir=processing_dir,
                                                      img_gs=img_multi_gs,
                                                      img_color_file=img_multi_file,
                                                      int_thr=int_thr,
                                                      d_thr_site_nm=d_thr_site_nm,
                                                      dil_it_n=dil_it_n,
                                                      dil_kernel_size=dil_kernel_size,
                                                      index=ind,
                                                      site_n_limit=site_n,
                                                      site_spread=spot_size_pix,
                                                      pix_nm_SPR=pix_nm_SPR)

                    str_spot_pos_l.append(new_peak_list)
                    if dist_tot !='NaN' and 0.8*dist_tot_design < dist_tot < 1.2 * dist_tot_design and dist_l_nm !='NaN' and pos_dist_check(ref_dist_l=ref_dist_l, pos_dist_l=dist_l_nm, std=site_d_std)==True and lin_score < lin_score_thr:
                        crop, loc_coord_export, loc_removed, ch_img_crop_l, rgb_img_crop_l =align_and_quant(img_clr=img_multi, ch_img_l=ch_img_l, rgb_img_l=rgb_img_l, ord_coor=new_peak_list,
                                                                                                loc_array_l=loc_array_l,
                                                                                                channel_color_l=channel_color_l,
                                                                                                channel_name_l=channel_name_l,
                                                                                                SPR=SPR,
                                                                                                pix_gap=pix_gap, crop_w_w=crop_w_w,
                                                                                                crop_w_h=crop_w_h,
                                                                                                site_spread=spot_size_pix,
                                                                                                offset_thr=offset_thr_pix,
                                                                                                site_n=site_n,
                                                                                                eps_dbscan = eps_dbscan,
                                                                                                export=export_align,
                                                                                                export_file_format=exp_file_format,
                                                                                                str_index=ind,
                                                                                                processing_dir=processing_dir)
                        str_crop_img_l.append(crop)
                        str_ch_img_l.append(ch_img_crop_l)
                        str_loc_l.append(loc_coord_export)
                        str_removed_loc_l.append(loc_removed)
                        str_rgb_img_l.append(rgb_img_crop_l)
                        
                    else:
                        str_crop_img_l.append('NaN')
                        str_ch_img_l.append('NaN')
                        str_loc_l.append('NaN')
                        str_removed_loc_l.append('NaN')
                        str_rgb_img_l.append('NaN')

            window_data_proc['OUTPUT'].update(value='Exporting results'+ '\n',
                            append=True)
            window_data_proc.refresh()
            df1 = pd.DataFrame(list(zip(str_index_l, str_pos_l, str_clr_img_l, str_crop_img_l, str_ch_img_l, str_rgb_img_l, str_spot_pos_l, str_loc_l, str_removed_loc_l)), columns=['Str_index','Str_pos', 'Str_clr_img', 'Str_crop', 'Str_channel_img',  'Str_rgb_channel_img', 'Str_spot_pos', 'Str_loc', 'Str_loc_removed'])
            df1.to_hdf(exp_folder + '/Processing_results.h5', key='proc_res', mode='w')
            window_data_proc['OUTPUT'].update(value='Processing finished'+ '\n',append=True)
            window_data_proc.refresh()

import numpy as np
import open3d as o3d
import math
import glob

def bbox(x,y):
    points = [
        [x*5, y*5, 0],
        [(x+1)*5, y*5, 0],
        [x*5, (y+1)*5, 0],
        [(x+1)*5, (y+1)*5, 0],
        [x*5, y*5, 30],
        [(x+1)*5, y*5, 30],
        [x*5, (y+1)*5, 30],
        [(x+1)*5, (y+1)*5, 30],         #x=5,y=5,z=30でボクセルを作ってる
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[0, 0, 1] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def crop_pcd(x, y, low_pcd):
    cropped_pcd = low_pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
        np.array([[x*5],[y*5],[-6]]),
        np.array([[(x+1)*5],[(y+1)*5],[20]]),
    ))    
    #重要度領域以外を表示
    dists = low_pcd.compute_point_cloud_distance(cropped_pcd)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.01)[0]
    if ind.size > 0:
        low_pcd = low_pcd.select_by_index(ind)

    return cropped_pcd, low_pcd

def merge(pcd):
#点群を切り抜いてそれぞれに処理をしたから、複数のばらばらの点群データとして見られている。これを修正したい。
    pcd_list = []                                                   #複数の点群データとしてではなく、点の集まりとして見れるようにばらばらに分けて代入しなおす
    for pc in pcd:                                                  #複数の点群データを一つ一つで見るためにループ
        for p in np.asarray(pc.points):                             #分割した点群を1点ごとに見るためにループ
            pcd_list.append(p)                                      #一点ごとに代入しなおし
    pcd_list = np.asarray(pcd_list)                                   
    merged_pcd = o3d.geometry.PointCloud()                          #代入しなおしたtestpdcはただの配列なので点群に直す
    print(merged_pcd)
    merged_pcd.points = o3d.utility.Vector3dVector(pcd_list)  
    
    return merged_pcd

def mse_point_to_plane(pcd_A, pcd_B):                                                               #点間MSEの導出
    errar_sum = 0
    print("NA = {}".format(len(np.asarray(pcd_A.points))))
    # 法線推定
    pcd_A.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=10))
    #o3d.visualization.draw_geometries([pcd_A], point_show_normal=True)
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_B)                                                      #kdtreeを使う
    for i, point_a_j in enumerate(pcd_A.points):                                                #pcdAの一点ごとにループを回して最近傍点を探索
        n_i = pcd_A.normals[i]                                                                  #pcd_ajの法線を取得
        [k, index, vector] = pcd_tree.search_knn_vector_3d(point_a_j, 1)                            #pcdAの一点と一番近いpcdBの点情報を獲得
        errar_vector = pcd_B.points[index[0]] - point_a_j                                           #pcdajの最近傍点bjからajをひいて誤差ベクトルを算出
        
        dot_product = np.dot(errar_vector, n_i)                                                     #誤差ベクトルと法線ベクトルの内積を計算
        errar_sum += pow(dot_product,2)                                                             #計算した内積を２乗したものの合計をとる
        
    mse = errar_sum / len(np.asarray(pcd_A.points))                                                 #2乗の誤差の合計をループを回した回数で割る（pcdAの点数）
    return mse

def psnr(mse, pcd_A):
    max_dist = 0
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_A)
    for point_A in pcd_A.points:
        [k, index, vector] = pcd_tree.search_knn_vector_3d(point_A, 2)
        distance = np.linalg.norm(np.asarray(point_A) - np.asarray(pcd_A.points[index[1]]))         #最近傍点との距離を導出
        if max_dist < distance:                                                                     #点間で一番長い距離を記録
            max_dist = distance

    psnr = 10 * math.log10(pow(max_dist, 2) / mse)
    return psnr

def gametheory(R_C, frame):
    w = 0.1
    M_1 = 237
    M_2 = 163
    v = 125
    t = 0.2

    A = pow((1+w)*M_1/(w*M_2),1/3)
    R_range = R_C*A/(1+A)

    R_1 = (M_1*R_C*(1+w)+R_C*math.sqrt(w*M_1*M_2*(1+w)))/(M_1+w*M_1-w*M_2)

    if R_range >= R_1 or R_C <= R_1:
        R_1 = (M_1*R_C*(1+w)-R_C*math.sqrt(w*M_1*M_2*(1+w)))/(M_1+w*M_1-w*M_2)

    p_h = R_1 * 0.2 / 35                     #5fpsなのでほんとは0.2をかけないとbpsにならない　なんなら0.2秒で何bitの使用帯域あるかでやってる→1秒換算で５倍必要になる

    p_l = (R_C-R_1) * 0.2 / 35

    return R_1, p_h, p_l

def Rotation_xyz(pointcloud, theta_x, theta_y, theta_z):        #グランドトゥルースの図はデータセットと角度がずれているので修正
    theta_x = math.radians(theta_x)
    theta_y = math.radians(theta_y)
    theta_z = math.radians(theta_z)
    rot_x = np.array([[ 1,                 0,                  0],
                      [ 0, math.cos(theta_x), -math.sin(theta_x)],
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])

    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                      [                 0, 1,                  0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])

    rot_z = np.array([[ math.cos(theta_z), -math.sin(theta_z), 0],
                      [ math.sin(theta_z),  math.cos(theta_z), 0],
                      [                 0,                  0, 1]])

    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    rot_pointcloud = rot_matrix.dot(pointcloud.T).T
    return rot_pointcloud, rot_matrix

def tag_locate():
    #tag_locateから人の移動情報を読み込み
    tag_xy = []
    count = 0
    txt_path = glob.glob("./tag_locate/*")  #パスを指定
    for file in txt_path:
        print(file)
        count+=1
        if count == 100:
            break
        with open(file, "rb") as f:
            reader = f.readlines()
            for line in reader:
                row = line.split()
                if "Car" in str(row[0]):                                 #LiDARのデータとtagデータの向きが違ったので合わせる
                    d_rad = math.radians(-90) # 角度法を弧度法に変換
                    x_rotated = float(row[15]) * math.cos(d_rad) - float(row[17]) * math.sin(d_rad)
                    y_rotated = float(row[15]) * math.sin(d_rad) + float(row[17]) * math.cos(d_rad)
                    tag_xy.append([x_rotated, y_rotated, float(row[16])])
    t_x = np.asarray(tag_xy)
    colors = np.array([[1.0,0,0] for i in range(len(tag_xy))])

    tag_pcd = o3d.geometry.PointCloud()
    tag_pcd.points = o3d.utility.Vector3dVector(t_x)
    tag_pcd.colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([tag_pcd])
    return tag_pcd, t_x

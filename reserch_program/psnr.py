import open3d as o3d
import numpy as np
import glob
import func
from matplotlib import pyplot as plt

bin_path = glob.glob("./dataset/*")
#bin_path = glob.glob("./100data/*")

limited_bandwidth = [15000000, 12500000, 10000000, 7500000, 5000000, 2500000]
high_PSNR_list = []
PSNR_list = []
filtered_hi_voxel = []

for bandwidth in limited_bandwidth:
    hi_box =  [[26, 29], [26, 30], [26, 32], [26, 33], [26, 34], [26, 35], [27, 17], [27, 28], [27, 29], [27, 30], [27, 31], [27, 32], [27, 33], [27, 34], [27, 35], [28, 19], [28, 29], [28, 30], [28, 31], [28, 32], [28, 33], [28, 34], [28, 36], [29, 20], [29, 27], [29, 28], [29, 29], [29, 30], [29, 31], [29, 32], [29, 33], [29, 36], [30, 17], [30, 18], [30, 19], [30, 26], [30, 27], [30, 28], [30, 29], [30, 30], [30, 31], [30, 32], [30, 33], [30, 35], [30, 36], [31, 17], [31, 18], [31, 19], [31, 20], [31, 25], [31, 26], [31, 27], [31, 28], [31, 29], [31, 30], [31, 31], [31, 32], [31, 35], [32, 17], [32, 18], [32, 19], [32, 20], [32, 21], [32, 22], [32, 23], [32, 24], [32, 25], [32, 26], [32, 27], [32, 28], [32, 29], [32, 30], [32, 31], [32, 33], [33, 17], [33, 18], [33, 19], [33, 20], [33, 22], [33, 23], [33, 24], [33, 25], [33, 26], [33, 27], [33, 28], [33, 29], [33, 30], [33, 31], [33, 32], [33, 33], [34, 17], [34, 18], [34, 19], [34, 20], [34, 21], [34, 22], [34, 23], [34, 24], [34, 26], [34, 27], [34, 28], [34, 29], [34, 30], [34, 31], [34, 32], [35, 17], [35, 18], [35, 19], [35, 21], [35, 22], [35, 23], [35, 24], [35, 25], [35, 26], [35, 27], [35, 28], [35, 29], [36, 18], [36, 19], [36, 22], [36, 23], [36, 24], [36, 25], [36, 26], [36, 27], [36, 28], [36, 29], [37, 19], [37, 20], [37, 24], [37, 28], [38, 20], [38, 21], [38, 23], [38, 24], [38, 25], [38, 26], [38, 27], [38, 28], [39, 21], [39, 22], [39, 23], [39, 24], [39, 25], [39, 26], [39, 27], [39, 28], [39, 29], [39, 30], [39, 32], [40, 19], [40, 20], [40, 21], [40, 22], [40, 23], [40, 24], [40, 25], [40, 26], [40, 27], [40, 28], [40, 29], [40, 30], [40, 31], [40, 32], [40, 33], [41, 18], [41, 19], [41, 20], [41, 21], [41, 22], [41, 23], [41, 24], [41, 25], [41, 26], [41, 27], [41, 28], [41, 29], [41, 30], [41, 31], [41, 32], [41, 33], [41, 34], [41, 35], [42, 17], [42, 18], [42, 20], [42, 21], [42, 22], [42, 23], [42, 24], [42, 25], [42, 26], [42, 27], [42, 28], [42, 29], [42, 30], [42, 31], [42, 32], [42, 33], [42, 34], [42, 35], [42, 36], [43, 17], [43, 19], [43, 20], [43, 21], [43, 22], [43, 23], [43, 24], [43, 25], [43, 31], [43, 32], [43, 33], [43, 34], [43, 35], [43, 36], [44, 17], [44, 18], [44, 19], [44, 20], [44, 21], [44, 22], [44, 23], [44, 24], [44, 32], [44, 33], [44, 34], [44, 35], [44, 36], [45, 18], [45, 19], [45, 20], [45, 21], [45, 23], [45, 34], [45, 35], [45, 36]]
    for b in hi_box:                                     #100m×100mの範囲に制限した高重要度ボクセルの位置
        if b[0]-30 <= 15 and b[0]-30 >= -5 and b[1]-26 <= 10 and b[1]-26 >= -10:
            filtered_hi_voxel.append(b)


    hi_pcd = []
    high_gametheory = []
    low_gametheory = []
    num_high_pcd = []
    all_high_psnr = 0
    all_psnr = 0
    frame = 0


    for bin_file in  bin_path:
        frame += 1
        frame_sum = 0
        hi_pcd = []
        downpcd = []
        print(frame)
        pcd_ = o3d.io.read_point_cloud(bin_file)

        theta_x = 1.8
        theta_y = -1
        theta_z = 0

        points = np.asarray(pcd_.points)

        rot_pointcloud, rot_matrix = func.Rotation_xyz(points, theta_x, theta_y, theta_z)

        rot_pcd = np.asarray(rot_pointcloud)

        pcd_rotated = o3d.geometry.PointCloud()
        pcd_rotated.points = o3d.utility.Vector3dVector(rot_pcd)

        #---pcdを20×20の範囲に切り抜く---
        pcd = pcd_rotated.crop(o3d.geometry.AxisAlignedBoundingBox(
            np.array([[-20],[-50],[-5.3]]),
            np.array([[80],[50],[20]]),
        )) 
        #---------------------------------

        o3d.visualization.draw_geometries([pcd])

        low_pcd = pcd

        #圧縮率適用したいところをくりぬく
        for b in filtered_hi_voxel:
            x,y = b[0]-30, b[1]-26          #ボクセルサイズによって変える
            h_pcd, low_pcd = func.crop_pcd(x,y,low_pcd)
            hi_pcd.append(h_pcd)

        #重要度の高い領域を一つ一つくりぬいたので複数の点群になっている。まとめて重要度の高い一つの点群にする
        high_pcd = func.merge(hi_pcd)

        #重要度の高い低い領域帯域幅を渡してゲーム理論で圧縮率を求める最終的には一番多かった圧縮率を時間帯で固定したい。毎回やってると処理に時間がかかりすぎる
        R_1, comp_high_points, comp_low_points = func.gametheory(bandwidth, frame)
        print("R_1,comp_H_points,comp_L_points:[{}, {}, {}]".format(R_1, comp_high_points, comp_low_points))
        c_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        for c in c_list:
            high_down_pcd = high_pcd.voxel_down_sample(voxel_size = c)
            if comp_high_points >= len(np.asarray(high_down_pcd.points)):
                break
        for c in c_list:
            low_down_pcd = low_pcd.voxel_down_sample(voxel_size = c)
            if comp_low_points >= len(np.asarray(low_down_pcd.points)):
                break

        num_high_pcd.append(len(np.asarray(high_pcd.points)))
        
        downpcd.append(high_down_pcd)
        downpcd.append(low_down_pcd)

        #点群を切り抜いてそれぞれに処理をしたから、複数のばらばらの点群データとして見られている。これを修正したい。
        all_down_pcd = func.merge(downpcd)
        print("all_down_pcd:{}".format(all_down_pcd))

        h_mseA = func.mse_point_to_plane(high_pcd, high_down_pcd)
        h_mseB = func.mse_point_to_plane(high_down_pcd, high_pcd)

        h_MSE = max(h_mseA, h_mseB)
 
        h_PSNR = func.psnr(h_MSE, high_pcd)              
        all_high_psnr += h_PSNR

        print("high_point_all:{}\nhigh_downpcd:{}\nhigh_comp_rate:{}".format(len(high_pcd.points), len(high_down_pcd.points), len(high_down_pcd.points)/len(high_pcd.points)))

        #全体領域のPSNRを導出
        mseA = func.mse_point_to_plane(pcd, all_down_pcd)
        mseB = func.mse_point_to_plane(all_down_pcd, pcd)

        MSE = max(mseA, mseB)

        if MSE == 0:
            MSE = 0.0001
        PSNR = func.psnr(MSE, pcd)
        all_psnr += PSNR

        print("MSE:{}\nhigh_PSNR:{}\nPSNR:{}".format(MSE, h_PSNR, PSNR))
        print("point_all:{}\ndownpcd:{}\ncomp_rate:{}".format(len(pcd.points), len(all_down_pcd.points), len(all_down_pcd.points)/len(pcd.points)))

    print("--------------------------------------------------------------------")
    print("bandwidth:{}\nhigh_PSNR^-:{}".format(bandwidth, all_high_psnr/frame))
    print("bandwidth:{}\nPSNR^-:{}".format(bandwidth, all_psnr/frame))
    print("--------------------------------------------------------------------")
    high_PSNR_list.append(all_high_psnr/frame)
    PSNR_list.append(all_psnr/frame)
#グラフの表示

#帯域幅ごとのPSNRのグラフ
plt.title("high_PSNR", {"fontsize":25})
plt.xlabel("bandwidth", {"fontsize":30})
plt.ylabel("PSNR", {"fontsize":30})
plt.tick_params(labelsize=20)
plt.plot(limited_bandwidth, high_PSNR_list, ".", markersize="15")
plt.xlim(0,15000000)
plt.ylim(0, 100)
plt.show()

plt.title("all_PSNR", {"fontsize":25})
plt.xlabel("bandwidth", {"fontsize":30})
plt.ylabel("PSNR", {"fontsize":30})
plt.tick_params(labelsize=20)
plt.plot(limited_bandwidth, PSNR_list, ".", markersize ="15")
plt.xlim(0,15000000)
plt.ylim(0, 100)
plt.show()

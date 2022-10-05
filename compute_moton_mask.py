


import argparse
import numpy as np
import os
import glob
import json
import cv2
from transforms import quaternion_to_euler, to_transform_matrix
from dataloader.depthloader import load_calib_cam_to_cam, disparity_loader
from utils.flowlib import read_flow, flow_to_image
import open3d as o3d
import matplotlib.pyplot as plt

def write_to_ply(point3d, color3d, filename):
    """ Write to ply.

    point3d: [num_points, 3]
    color3d:[num_points, 3]
    
    """
    print(f"point3d {point3d.shape}, color3d {color3d.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point3d)
    pcd.colors = o3d.utility.Vector3dVector(color3d)
    o3d.io.write_point_cloud(filename, pcd)


def min_max_compute(data):
    min_d = np.min(data)
    max_d = np.max(data)
    return (data- min_d)/ (max_d -min_d)


def read_json(json_filename):
    with open(json_filename) as json_file:
        data = json.load(json_file)
        return data

def make_point_cloud_from_depth(depth, intrinsic):

    height, width = depth.shape
    x0,y0=np.meshgrid(range(width),range(height))

    hom_coord = np.concatenate((x0[np.newaxis],y0[np.newaxis],np.ones(x0.shape)[np.newaxis]),0).reshape((3,-1))

    cam_coords = np.matmul(np.linalg.inv(intrinsic), hom_coord)
    cam_coords = cam_coords / cam_coords[2]

    point_cloud = cam_coords  * depth.reshape(-1)

    return np.transpose(point_cloud)


def sampson_error_f(F, pts1, pts2):
    
    """
    F = 3x3
    pts1 = 3 x n
    pts2 = 3x n
    
    """
    
    Fp1 = np.matmul(F, pts1)
    Fp2 = np.matmul(F.T, pts2)
    p2Fp1 = np.einsum("ji, ji->i", pts2, Fp1)
    epsilon = 1e-8
    error = (p2Fp1**2) / (Fp1[0]**2 + Fp2[0]**2 + Fp1[1]**2 + Fp2[1]**2 + epsilon)
    return error

def normalize_angle(angle):
    """
    The function normalizes angle between
    -pi to pi values
    Arguments:
    angle: in radians
    Returns:
    angle: between -pi/2 to pi/2 in radians
    """
    while (angle > np.pi/2.0).any():
        mask = angle > np.pi/2.0
        angle[mask] -= 2 * np.pi
    while (angle < -np.pi/2.0).any():
        mask =  angle < -np.pi/2.0
        angle[mask] += 2 * np.pi
    return angle
def compute_sin(scene_flow, T):
    
    v1 = scene_flow / np.linalg.norm(scene_flow, axis=1)[..., None]
    v2 = -T / np.linalg.norm(T)
    dot_product = np.dot(v1, v2)
    angle = normalize_angle(np.arccos(dot_product))
    print(angle)
    return np.sin(angle)

def make_motion_masks(input_dir, output_dir):
    """Optical expansion."""
    out_expansion = os.path.join(output_dir, "expansion_visu")
    out_sampson = os.path.join(output_dir, "sampson_error")
    out_angle = os.path.join(output_dir, "angle_error")
    delta_depth = os.path.join(output_dir, "delta_depth")
    flow_3d =os.path.join(output_dir, "flow_3d")
    expansion_3d = os.path.join(output_dir, "expansion_3d")
    os.makedirs(out_expansion, exist_ok=True)
    os.makedirs(out_sampson, exist_ok=True)
    os.makedirs(out_angle, exist_ok=True)
    os.makedirs(delta_depth, exist_ok=True)
    os.makedirs(flow_3d, exist_ok=True)
    os.makedirs(expansion_3d, exist_ok=True)

    intrinsics_path = os.path.join(input_dir, "intrinsics.json")
    pose_path = os.path.join(input_dir, "poses.json")
    flow_path = os.path.join(input_dir, "optical_flow", "flow_resuls")
    expansion_path = os.path.join(input_dir, "optical_expansion", "seq")


    images_list = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    # Get intrinsic
    intrinsic_dict =read_json(intrinsics_path)
    intrinsic = np.eye(3)
    intrinsic[0,0] = intrinsic_dict["fx"]
    intrinsic[1,1] = intrinsic_dict["fy"]
    intrinsic[0,2] = intrinsic_dict["cx"]
    intrinsic[1,2] = intrinsic_dict["cy"]
    K0 = K1 = intrinsic
    # get poses
    poses_dict = read_json(pose_path)

    for idx, image_path in enumerate(images_list[:-1]):
        basename = os.path.basename(image_path)
        # load image 
        image = np.asarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), dtype=np.float32)
        height, width, _ = image.shape
        shape = [height, width]
        # Load flow
        #optical_flow = np.load(os.path.join(flow_path, basename.replace(".jpg", ".npy")))
        optical_flow2 = read_flow(os.path.join(expansion_path,"flo-" + basename.replace(".jpg", ".pfm") ))
        #print(f"optical_flow {np.min(optical_flow)}, max {np.max(optical_flow)}")
        #print(f"optical_flow {np.min(optical_flow2)}, max {np.max(optical_flow2)}")

        tau = np.exp(disparity_loader(os.path.join(expansion_path,"mid-" + basename.replace(".jpg", ".pfm") )))

        #print(f"image {image.shape}, optical_flow {optical_flow.shape}, tau {tau.shape}")

        print(f"tau min {np.min(tau)}, max {np.max(tau)}")
        # get pose
        translation = poses_dict[idx]["position"]
        translation = np.asarray([translation[key] for key in translation])
        rotation = poses_dict[idx]["heading"]
        rotation = np.asarray([rotation[key] for key in rotation])
        rotation = quaternion_to_euler(*rotation)
        world_T_camera1 = to_transform_matrix(translation, rotation)
        #R1, t1 = world_T_camera1[:3, :3], world_T_camera1[:3, 3]

        translation = poses_dict[idx+1]["position"]
        translation = np.asarray([translation[key] for key in translation])
        rotation = poses_dict[idx+1]["heading"]
        rotation = np.asarray([rotation[key] for key in rotation])
        rotation = quaternion_to_euler(*rotation)
        world_T_camera2 = to_transform_matrix(translation, rotation)

        camera1_T_camera2= np.matmul(np.linalg.inv(world_T_camera1), world_T_camera2)
        # Relative
        R, T = camera1_T_camera2[:3, :3], camera1_T_camera2[:3, 3]
        #T = T / np.linalg.norm(T)

        #R2, t2 = world_T_camera2[:3, :3], world_T_camera2[:3, 3]
        #b = t2 -t1
        S = np.asarray( [ [0.0, -T[2], T[1] ], [T[2], 0.0, -T[0]], [-T[1], T[0], 0.0]])


        # Depth from flow
        x0,y0=np.meshgrid(range(shape[1]),range(shape[0]))
        print(f"x0 {x0.shape}")
        x0=x0.astype(np.float32)
        y0=y0.astype(np.float32)
        x1=x0+optical_flow2[:,:,0]
        y1=y0+optical_flow2[:,:,1]
        hp0 = np.concatenate((x0[np.newaxis],y0[np.newaxis],np.ones(x0.shape)[np.newaxis]),0).reshape((3,-1))
        hp1 = np.concatenate((x1[np.newaxis],y1[np.newaxis],np.ones(x1.shape)[np.newaxis]),0).reshape((3,-1))

        x1_dash = np.matmul(np.eye(3), np.matmul(np.linalg.inv(K0), hp0))
        x2_dash = np.matmul(R, np.matmul(np.linalg.inv(K1), hp1))


        samson_error = sampson_error_f(S,x1_dash, x2_dash).reshape(shape)
                            



        # P_pred = cv2.triangulatePoints(K0.dot(np.concatenate( (np.eye(3),np.zeros((3,1))), -1)), 
        #                         K1.dot(np.concatenate( (R.T,-R.T.dot(T[:,np.newaxis])), -1)), 
        #                         hp0[:2],hp1[:2])
        camera1_T_world, camera2_T_world =np.linalg.inv(world_T_camera1), np.linalg.inv(world_T_camera2)
        P_pred = cv2.triangulatePoints(K0.dot(camera1_T_world[:3, :]), K1.dot(camera2_T_world[:3, :]), hp0[:2],hp1[:2])                      


        P_pred = P_pred[:3]/P_pred[-1:]  
        disp_flow = 1./P_pred[-1].reshape(shape)



        # depth from up-to-scale flow
        H01 = K0.dot(R).dot(np.linalg.inv(K1)) # remove the effect of camera rotation
        hp1 = H01.dot(hp1)
        # it can be proved that up to scale 3D flow = (ux, uy, uz)/d0(x,y) = K^{-1}(tau*p1-p0), tau = d1/d0
        p3d = np.transpose(np.linalg.inv(K0).dot(tau.flatten()[np.newaxis]*hp1-hp0),[1,0])
        disp_p3d = np.linalg.norm(p3d,2,1).reshape(shape) # forcing unit norm yields the disparity of the rigid pixels


        scale = np.median(disp_p3d) / np.median(disp_flow)

        depth_contrast = np.abs(np.log(scale * disp_flow /disp_p3d))
        depth_max  = np.max( np.where(np.isnan(depth_contrast), 0.0, depth_contrast))
        depth_contrast[np.isnan(depth_contrast)] = 0

        #print(f"depth_contrast {np.min(depth_contrast)}, max {np.max(depth_contrast)}")

        #depth_contrast = np.clip(depth_contrast, 0.0, 1.0)

        # Visu

        sin_b = compute_sin(p3d, T).reshape(shape)

        error = np.linalg.norm(p3d, axis=1).reshape(shape) * np.abs(sin_b)
        norm_error = min_max_compute(error) * 255.0
        cv2.imwrite(os.path.join(out_angle, basename), norm_error)

        samson_error_norm = min_max_compute(samson_error) *255.0
        cv2.imwrite(os.path.join(out_sampson, basename), samson_error_norm)

        tau_norm = min_max_compute(tau) * 255.0
        #scene_flow =  min_max_compute(p3d.reshape((*shape, 3))) * 255
        
        cv2.imwrite(os.path.join(out_expansion, basename), tau_norm)

        depth_contrast_norm = min_max_compute(depth_contrast) * 255.0
        cv2.imwrite(os.path.join(delta_depth, basename), depth_contrast_norm)

        color3d =np.reshape(image, (-1, 3)) / 255.0
        P_pred[:, np.linalg.norm(P_pred, axis=0)>100] = 0
        write_to_ply(np.transpose(P_pred), color3d, os.path.join(flow_3d, basename.replace(".jpg", ".ply")))

        flow_point_cloud = make_point_cloud_from_depth(1.0 / disp_p3d, intrinsic)
        flow_point_cloud[np.linalg.norm(flow_point_cloud, axis=1)>100, :] = 0

        write_to_ply(flow_point_cloud, color3d, os.path.join(expansion_3d, basename.replace(".jpg", ".ply")))

if __name__ == "__main__":
    parser  = argparse.ArgumentParser("Make motion mask")

    parser.add_argument("--input_dir")

    parser.add_argument("--output_dir")
    args = parser.parse_args()
    make_motion_masks(args.input_dir, args.output_dir)
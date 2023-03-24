import OpenEXR
import numpy as np
import torch
# from npz2pc import writefile
import os
import cv2
import matplotlib.pyplot as plt

# def readEXR(fpath):
#    readFile = OpenEXR.InputFile(fpath)
#    (r,g,b) = readFile.channels('RGB')
#    dr = np.copy(np.frombuffer(r, dtype=np.half))
#    db = np.copy(np.frombuffer(b, dtype=np.half))
#    dg = np.copy(np.frombuffer(g, dtype=np.half))
#    h = readFile.header()['displayWindow'].max.y+1-readFile.header()['displayWindow'].min.y
#    w = readFile.header()['displayWindow'].max.x+1-readFile.header()['displayWindow'].min.x
#    dr = np.reshape(dr,(h,w,1))
#    db = np.reshape(db,(h,w,1))
#    dg = np.reshape(dg,(h,w,1))
#    depth=np.append(dr,np.append(db,dg,axis=-1),axis=-1)
#    return depth.astype(np.float)
def readEXR(fpath: str) -> np.ndarray:
    """Read depth map from the .exr file

    Args:
        fpath (str): the path to the .exr depth map

    Returns:
        np.ndarray: 256*256, depth map
    """
    readFile = OpenEXR.InputFile(fpath)
    (r, g, b) = readFile.channels("RGB")
    dr = np.copy(np.frombuffer(r, dtype=np.half))
    db = np.copy(np.frombuffer(b, dtype=np.half))
    dg = np.copy(np.frombuffer(g, dtype=np.half))
    assert np.allclose(dr, db)
    assert np.allclose(db, dg)
    h = (
        readFile.header()["displayWindow"].max.y
        + 1
        - readFile.header()["displayWindow"].min.y
    )
    w = (
        readFile.header()["displayWindow"].max.x
        + 1
        - readFile.header()["displayWindow"].min.x
    )
    dr = np.reshape(dr, (h, w))
    # db = np.reshape(db,(h,w,1))
    # dg = np.reshape(dg,(h,w,1))
    # depth=np.append(dr,np.append(db,dg,axis=-1),axis=-1)
    return dr.astype(np.float)
    
def colorCodeDepth(inImg):
    givenImg = inImg[:,:,0]
    depthValues = givenImg[(givenImg==255)==False].astype('float')
    depthValues = (depthValues-depthValues.min())
    depthValues = 250.0*depthValues/depthValues.max()
    depthValues += 1
    givenImg[(givenImg==255)==False] = depthValues.astype('uint8')
    
    # givenImg = givenImg.astype('float')/(1.0*np.max(givenImg))
    # givenImg = 1/givenImg
    # givenImg = givenImg/np.max(givenImg)
    # givenImg = givenImg*255.0
    givenImg = (givenImg).astype('uint8')
    colorCodedImage =  cv2.applyColorMap(givenImg, cv2.COLORMAP_PLASMA)
    colorCodedImage[inImg[:,:,0]==255,:] = 255
    return colorCodedImage


def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.
    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True
    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor

def transform_to_world(pixels, depth, camera_mat, world_mat, scale_mat,
                       invert=True):
    ''' Transforms pixel positions p with given depth value d to world coordinates.
    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)

    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    p_world = scale_mat @ world_mat @ camera_mat @ pixels

    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world

def getPointCloudFromDepths(scanPath):
    fpath=os.path.join(scanPath,"depth")
    imgPath=os.path.join(scanPath,"image")
    camPath=os.path.join(scanPath,"cameras.npz")
    camMats = np.load(camPath)
    read_color = True
    if read_color:
        pc = np.zeros((0,6))
    else:
        pc = np.zeros((0,3))

    print(scanPath + ": Getting pc with colors")
    for imageName in os.listdir(fpath):
        fnum = int(imageName[:-8])
        # 00230001.exr

        camName = 'camera_mat_'+str(fnum)
        worldName = 'world_mat_'+str(fnum)

        camera_mat = camMats[camName]
        world_mat = camMats[worldName]

        exrFilePath = os.path.join(fpath,imageName)
        depth = readEXR(exrFilePath)
        depth = depth[:,:,0]
        depth = np.expand_dims(depth,axis=2)
        r = np.linspace(0,depth.shape[0]-1,depth.shape[0])
        c = np.linspace(0,depth.shape[1]-1,depth.shape[1])

        r = 2*(r/np.max(r)) -1
        c = 2*(c/np.max(c)) -1
        p1,p2 = np.meshgrid(r,c)
        pixels = np.append(np.expand_dims(p1,axis=2),np.expand_dims(p2,axis=2),axis=2)


        worldCoords = transform_to_world(pixels, depth, camera_mat, world_mat, np.eye(4), invert=True)
        coords = (worldCoords==-np.inf) | (worldCoords==np.inf) | np.isnan(worldCoords,where=True)
        coords = coords == False
        coords = np.sum(coords,axis=2)
        worldCoords = worldCoords[coords==3,:]
        if read_color:
            currImgPath = os.path.join(imgPath,"00"+str(fnum).zfill(2)+".png")
            colorPixels = cv2.imread(currImgPath)
            colorPixels = colorPixels[coords==3,:]
        # import pdb; pdb.set_trace()
            pc = np.append(pc,np.append(worldCoords,colorPixels,axis=1),axis=0)
        else:
            pc = np.append(pc,worldCoords,axis=0)
    return pc


if __name__ == "__main__":
    # scanPath = "/storage/user/yenamand/one/differentiable_volumetric_rendering/data/ShapeNet/02691156/9ff7d7d71bcf50ff4fb6842b3610149/"
    #pc = getPointCloudFromDepths(scanPath)
    # writefile("debug.obj",pc)
    scanPath = '/storage/user/yenamand/one/differentiable_volumetric_rendering/data/ShapeNet/02958343/a3f8fa2d571276a596f33e8908dbb2a2/depth/00000001.exr'
    depth = readEXR(scanPath)
    # depth = colorCodeDepth(depth)
    # plt.imshow(depth)
    # plt.show()
    cv2.imshow('depth', depth)
    k=cv2.waitKey(0)
    if k == ord("q"):
        cv2.destroyAllWindows()
    # print(sorted(keys))

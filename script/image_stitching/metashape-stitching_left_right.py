import Metashape
import os
import logging

metashape_doc = Metashape.app.document
mask_path = '/blue/cli2/zhengkun.li/peanut_project/Analysis_result/SFM_MetaShape/mask.png'
root_foder = "/blue/cli2/zhengkun.li/peanut_project/Analysis_result/SFM_MetaShape"

dir_list = [directory for directory in os.listdir(root_foder+"/left") if os.path.isdir(os.path.join(root_foder,"left",directory))]

logging.info(dir_list)

for dir in dir_list:
    project_path = os.path.join(root_foder,'metashape_projects', dir+'.psx')
    if (os.path.isfile(project_path)):
        continue
    metashape_doc.save(project_path)
    metashape_doc.open(project_path, ignore_lock=True)
    metashape_doc.clear()
    
    chunk = metashape_doc.addChunk()
    left_camera_group = chunk.addCameraGroup()
    left_camera_group.label="left"
    right_camera_group = chunk.addCameraGroup()
    right_camera_group.label="right"

    left_camera_path = os.path.join(root_foder, "left", dir)
    left_images = [os.path.join(left_camera_path, file)
        for file in os.listdir(left_camera_path)
        if file.endswith('.jpg') or file.endswith('.JPG')]
    
    right_camera_path = os.path.join(root_foder, "right", dir)

    if not os.path.exists(right_camera_path):
        continue
    
    right_images = [os.path.join(right_camera_path, file)
        for file in os.listdir(right_camera_path)
        if file.endswith('.jpg') or file.endswith('.JPG')]
    
    chunk.addPhotos(left_images, group=0)
    chunk.addPhotos(right_images, group=1)

    #chunk.generateMasks(path="", masking_mode=Metashape.MaskingModeFile)
    #mask = Metashape.Mask()
    #mask.load(mask_path)
    #for camera in chunk.cameras:
    #    camera.mask=mask
        
    chunk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=False)
    chunk.alignCameras()
    chunk.buildDepthMaps(downscale=4, filter_mode=Metashape.ModerateFiltering)
    chunk.buildModel(surface_type=Metashape.Arbitrary,face_count=Metashape.CustomFaceCount, face_count_custom=200000)
    chunk.smoothModel(strength=100)
    metashape_doc.save(project_path)
    metashape_doc.open(project_path, ignore_lock=True)
    chunk=metashape_doc.chunks[0]
    chunk.buildOrthomosaic()
    metashape_doc.save(project_path)
    chunk.exportRaster(path=os.path.join(root_foder,'ortho', dir+'.jpg'), image_format=Metashape.ImageFormatJPEG)

    try:
        metashape_doc.save(project_path)
    except RuntimeError:
        Metashape.app.messageBox("Can't save project: "+ project_path)



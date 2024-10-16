import Metashape
import os
import logging

metashape_doc = Metashape.app.document
mask_path = '/blue/cli2/zhengkun.li/peanut_project/Analysis_result/SFM_MetaShape/mask.png'
# frame_path = "/blue/cli2/zhengkun.li/peanut_project/plot-scale_analysis/citra/frames/23501_all"
# save_path = "/blue/cli2/zhengkun.li/peanut_project/plot-scale_analysis/citra/SFM_MetaShape"

frame_path = "/blue/cli2/zhengkun.li/peanut_project/plot-scale_analysis/citra_20230706/frames/right"
save_path = "/blue/cli2/zhengkun.li/peanut_project/plot-scale_analysis/citra_20230706/SFM_MetaShape/stitching/right"

# logging.basicConfig(filename=os.path.join(save_path, 'metashape.log'), level=logging.INFO)

# extract all the directories in the root folder
dir_list = [dir for dir in os.listdir(frame_path) if os.path.isdir(os.path.join(frame_path, dir))]

logging.info(dir_list)

for dir in dir_list:
    project_path = os.path.join(save_path,'metashape_projects', dir+'.psx')

    if (os.path.isfile(project_path)):
        continue

    metashape_doc.save(project_path)
    metashape_doc.open(project_path, ignore_lock=True)
    metashape_doc.clear()
    
    chunk = metashape_doc.addChunk()
    camera_group = chunk.addCameraGroup()
    camera_group.label="camera_group"

    camera_path = os.path.join(frame_path, dir)
    images = [os.path.join(camera_path, file)
        for file in os.listdir(camera_path)
        if file.endswith('.jpg') or file.endswith('.JPG')]
    
    chunk.addPhotos(images, group=0)

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
    chunk.exportRaster(path=os.path.join(save_path,'ortho', dir+'.jpg'), image_format=Metashape.ImageFormatJPEG)

    try:
        metashape_doc.save(project_path)
    except RuntimeError:
        Metashape.app.messageBox("Can't save project: "+ project_path)



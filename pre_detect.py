from libs import *
from align_imgs import align_face


def pre_detect(source):
    #config
    save_img=False
    weights = 'checkpoints/whole_passport.pt' 
    save_dir = 'runs/detect'       
    name = 'stage_1' #'save results to project/name'  
    augment = False   
    imgsz = 416 #img_size
    conf_thres = 0.66
    iou_thres = 0.5
    save_txt = True
    save_conf = False
    # source = '/Users/duc/Downloads/passport/yolov5/ho chieu test'

    # Directories
    save_dir = Path(increment_path(Path(save_dir) / name, exist_ok=False))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16


    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t2 = time_synchronized()



        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0 = Path(path), '', im0s

            save_path = str(save_dir / p.name)
            
            txt_path = str(save_dir / 'labels' / p.stem) 
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = get_coor(xyxy)
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        print(xyxy)
                        #print(xywh)
                        
                        print(c1)
                        print(c2)
                        im0 = im0[c1[1]:c2[1], c1[0]:c2[0]]
                        # print(img_crop.shape)


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
                    
            im0 = align_face(im0)

            if save_img:
                cv2.imwrite(save_path, im0)


    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        print(save_dir)

    print('Done. (%.3fs)' % (time.time() - t0))

    return save_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--source', type=str, default='DB1C47D9-2A68-4AFF-B332-45599D28C022.jpg', help='source')  # file/folder, 0 for webcam
    opt = parser.parse_args()
    source = '/Users/duc/Downloads/passport/yolov5/Test passport2/IMG_1420.jpg'
    with torch.no_grad():
        start = timeit.default_timer()
        # img = cv2.imread('/Users/duc/Downloads/passport/yolov5/images/result_test_datahc_Y_Ty.jpg')
        # mask = segment(img)
        # img = perspective_transform(img, mask)
        # cv2.imwrite('12.png',img)
        print(pre_detect(source))
        stop = timeit.default_timer()

        print('Time: ', stop - start)

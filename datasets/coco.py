def get_coco_train(args):
    from symimdb.coco import coco
    if not args.imageset:
        args.imageset = 'train2017'
    args.rcnn_num_classes = len(coco.classes)

    isets = args.imageset.split('+')
    roidb = []
    for iset in isets:
        imdb = coco(iset, 'data', 'data/coco')
        imdb.filter_roidb()
        imdb.append_flipped_images()
        roidb.extend(imdb.roidb)
    return roidb

def get_coco_test(args):
    from symimdb.coco import coco
    if not args.imageset:
        args.imageset = 'val2017'
    args.rcnn_num_classes = len(coco.classes)
    return coco(args.imageset, 'data', 'data/coco')
def get_voc_train(args):
    from symimdb.pascal_voc import PascalVOC
    if not args.imageset:
        args.imageset = '2007_trainval'
    args.rcnn_num_classes = len(PascalVOC.classes)

    isets = args.imageset.split('+')
    roidb = []
    for iset in isets:
        imdb = PascalVOC(iset, 'data', 'data/VOCdevkit')
        imdb.filter_roidb()
        imdb.append_flipped_images()
        roidb.extend(imdb.roidb)
    return roidb

def get_voc_test(args):
    from symimdb.pascal_voc import PascalVOC
    if not args.imageset:
        args.imageset = '2007_test'
    args.rcnn_num_classes = len(PascalVOC.classes)
    return PascalVOC(args.imageset, 'data', 'data/VOCdevkit')

#
# def voc(args, type):
#     if type == 'test':
#         # get_voc(args)
#         pass
#     elif type == 'train':
#         get_voc_train(args)

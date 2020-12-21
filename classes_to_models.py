
classes = ['aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle','building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup','curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard','light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road', 'rock','sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree','truck', 'tvmonitor', 'wall', 'water', 'window', 'wood']

models = ['cityscapes_label2photo_pretrained','cityscapes_photo2label_pretrained','horse2zebra_pretrained','iphone2dslr_flower_pretrained','map2sat_pretrained','sat2map_pretrained','monet2photo_pretrained','style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained','winter2summer_yosemite_pretrained']

classes_to_models = {}

classes_to_models['background'] = ['winter2summer_yosemite_pretrained']

classes_to_models['dog'] = ['horse2zebra_pretrained']
classes_to_models['bird'] = ['horse2zebra_pretrained']
classes_to_models['cat'] = ['horse2zebra_pretrained']
classes_to_models['cow'] = ['horse2zebra_pretrained']
classes_to_models['horse'] = ['horse2zebra_pretrained']

classes_to_models['motorbike'] = ['winter2summer_yosemite_pretrained', 'style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['aeroplane'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['bag'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['bed'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['bedclothes'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['bench'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['bicycle'] = ['cityscapes_label2photo_pretrained','cityscapes_photo2label_pretrained','style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['boat'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['book'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['bottle'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['building'] = ['cityscapes_label2photo_pretrained','cityscapes_photo2label_pretrained','style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['bus'] = ['cityscapes_label2photo_pretrained','cityscapes_photo2label_pretrained','style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['cabinet'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['car'] = ['cityscapes_label2photo_pretrained','cityscapes_photo2label_pretrained','style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['ceiling'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['chair'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['cloth'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['computer'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['cup'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['curtain'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['door'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['fence'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['floor'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['flower'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['food'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['grass'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['ground'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['keyboard'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['light'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['mountain'] = ['cityscapes_label2photo_pretrained', 'style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['mouse'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['person'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['plate'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['platform'] = ['cityscapes_label2photo_pretrained','cityscapes_photo2label_pretrained','style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['pottedplant'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['road'] = ['cityscapes_label2photo_pretrained','cityscapes_photo2label_pretrained','map2sat_pretrained','sat2map_pretrained','style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['rock'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['sheep'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['shelves'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['sidewalk'] = ['cityscapes_label2photo_pretrained','cityscapes_photo2label_pretrained','style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['sign'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['sky'] = ['cityscapes_label2photo_pretrained', 'style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained', 'winter2summer_yosemite_pretrained']

classes_to_models['snow'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained','winter2summer_yosemite_pretrained']

classes_to_models['sofa'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['table'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['track'] = ['cityscapes_label2photo_pretrained','cityscapes_photo2label_pretrained','map2sat_pretrained','sat2map_pretrained', 'style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['train'] = ['cityscapes_label2photo_pretrained','cityscapes_photo2label_pretrained','style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['tree'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['truck'] = ['cityscapes_label2photo_pretrained','cityscapes_photo2label_pretrained','style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['tvmonitor'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['wall'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['water'] = ['map2sat_pretrained','sat2map_pretrained','style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['window'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained']

classes_to_models['wood'] = ['style_cezanne_pretrained','style_monet_pretrained','style_ukiyoe_pretrained','style_vangogh_pretrained','winter2summer_yosemite_pretrained']

def get_mapping():
    return classes, models, classes_to_models
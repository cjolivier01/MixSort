import json
import os

def process(split="train"):

    video_list = list()
    category_list = [
        {'id': 1, 'name': 'person'},
        {'id': 2, 'name': 'referee'},
        {'id': 3, 'name': 'player'},
    ]

    all_image_ids = set()
    all_annotation_ids = set()
    all_video_ids = set()
    video_id_mapping = dict()

    max_img = 100000
    max_ann = 2000000
    max_video = 1000

    base_dir = os.path.join(os.getcwd(), 'datasets')

    split_json = json.load(open(f'{base_dir}/crowdhuman/annotations/{split}.json','r'))

    img_list = list()
    ann_list = list()
    
    #
    # CH (image dataset)
    #
    img_id_count = 0
    for img in split_json['images']:
        img_id_count += 1
        img['file_name'] = f'{base_dir}/crowdhuman/Crowdhuman_{split}/' + img['file_name']
        img['frame_id'] = img_id_count
        # No prev for non-sequence-video
        img['prev_image_id'] = img['id'] + max_img
        img['next_image_id'] = img['id'] + max_img
        image_id = img['id'] + max_img
        assert image_id not in all_image_ids
        all_image_ids.add(image_id)
        img['id'] = image_id
        img['video_id'] = max_video
        img_list.append(img)

    anno_count = 0
    for ann in split_json['annotations']:
        anno_count += 1
        annotation_id = ann['id'] + max_ann
        assert annotation_id not in all_annotation_ids
        all_annotation_ids.add(annotation_id)
        ann['id'] = annotation_id
        ann['image_id'] = ann['image_id'] + max_img
        assert ann['category_id'] == 1
        ann_list.append(ann)


    video_list.append({
        'id': max_video,
        'file_name': f'crowdhuman_{split}'
    })
    video_id_mapping[max_video] = max_video

    #
    # HDS (video dataset)
    #
    split_json = json.load(open(f'{base_dir}/hockeyTrackingDataset/annotations/{split}.json','r'))

    max_img += int(img_id_count * 2)
    max_ann += int(anno_count * 2)
    max_video += 1

    img_id_count = 0
    anno_count = 0
    
    for vid in split_json['videos']:
        old_video_id = vid['id']
        new_video_id = vid['id'] + max_video 
        assert new_video_id not in all_video_ids
        assert old_video_id not in video_id_mapping
        all_video_ids.add(new_video_id)
        video_id_mapping[old_video_id] = new_video_id
    
    for img in split_json['images']:
        img['file_name'] = f'{base_dir}/hockeyTrackingDataset/{split}/' + img['file_name']
        # No prev for non-sequence-video
        image_id = img['id'] + max_img
        assert image_id not in all_image_ids
        all_image_ids.add(image_id)
        img['id'] = image_id
        if img['prev_image_id'] >= 0:
            img['prev_image_id'] = img['prev_image_id'] + max_img
        img['next_image_id'] = img['next_image_id'] + max_img
        img['video_id'] = video_id_mapping[img['video_id']]
        img_list.append(img)
    
    for ann in split_json['annotations']:
        annotation_id = ann['id'] + max_ann
        assert annotation_id not in all_annotation_ids
        all_annotation_ids.add(annotation_id)
        ann['id'] = annotation_id
        ann['image_id'] = ann['image_id'] + max_img
        assert ann['category_id'] == 1
        # Set new catagory as 'player'
        ann['category_id'] = 3
        ann_list.append(ann)
    

    print(f'hockeyTrackingDataset-{split}')

    mix_json = dict()
    mix_json['images'] = img_list
    mix_json['annotations'] = ann_list
    mix_json['videos'] = video_list
    mix_json['categories'] = category_list
    json.dump(mix_json, open(f'datasets/hockeyTraining/annotations/mix_{split}.json','w'))

if __name__ == '__main__':
    process(split='train')

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

    max_img = 100000
    max_ann = 2000000
    max_video = 100

    base_dir = os.path.join(os.getcwd(), 'datasets')

    train_json = json.load(open(f'{base_dir}/crowdhuman/annotations/{split}.json','r'))

    img_list = list()
    img_id_count = 0
    for img in train_json['images']:
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

    ann_list = list()
    for ann in train_json['annotations']:
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

    max_video += 1

    print(f'crowdhuman_{split}')

    mix_json = dict()
    mix_json['images'] = img_list
    mix_json['annotations'] = ann_list
    mix_json['videos'] = video_list
    mix_json['categories'] = category_list
    json.dump(mix_json, open(f'datasets/hockeyTraining/annotations/mix_{split}.json','w'))

if __name__ == '__main__':
    process(split='train')

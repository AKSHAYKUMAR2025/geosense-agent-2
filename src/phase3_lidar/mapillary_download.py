# mapillary_download.py
# Purpose: Download 360-degree street-level images for a bounding box using the Mapillary API v4

import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('MAPILLARY_TOKEN')
BASE_URL = 'https://graph.mapillary.com/images'


def find_images_in_bbox(min_lon, min_lat, max_lon, max_lat, limit=50):
    """Query Mapillary for image metadata inside a bounding box."""

    params = {
        'access_token': TOKEN,
        'fields': 'id,thumb_2048_url,geometry,compass_angle,captured_at',
        'bbox': f'{min_lon},{min_lat},{max_lon},{max_lat}',
        'limit': limit,
    }

    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()

    return resp.json().get('data', [])


def download_image(image_meta, save_dir='data/streetview/images'):
    """Download a single image to disk."""

    os.makedirs(save_dir, exist_ok=True)

    url = image_meta['thumb_2048_url']
    image_id = image_meta['id']

    filepath = os.path.join(
        save_dir,
        f'{image_id}.jpg'
    )

    img_data = requests.get(url).content

    with open(filepath, 'wb') as f:
        f.write(img_data)

    return filepath


if __name__ == '__main__':

    # Example: small area in Chennai
    images = find_images_in_bbox(
        80.24, 13.05,
        80.25, 13.06,
        limit=20
    )

    print(f'Found {len(images)} images')

    for img in images:
        path = download_image(img)
        print('Saved:', path)
import os

# Resolve relative to project root so moving the repo does not break paths.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_PATH_DICT = {
    'KittiDataset12': os.path.join(_PROJECT_ROOT, 'data', 'KITTI12').replace('\\', '/'),
    'KittiDataset15': os.path.join(_PROJECT_ROOT, 'data', 'KITTI15').replace('\\', '/'),
}
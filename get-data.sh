#!/usr/bin/bash

set -xe

# Create directories
mkdir ./data
mkdir ./models

echo "Downloading ultralytcs-timm..."
kaggle datasets download sersasj/ultralytcs-timm-rsna
unzip ultralytcs-timm-rsna.zip


echo "Downloading competition data"
kaggle competitions download -c rsna-intracranial-aneurysm-detection
unzip ./rsna-intracranial-aneurysm-detection.zip -d ./data/
